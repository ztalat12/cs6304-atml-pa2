from __future__ import annotations

"""Task 1 DPO training loop.

Pipeline for every optimizer step
---------------------------------
  for each of `grad_accum_steps` micro-batches:
      1. policy forward on chosen + rejected   -> log pi_theta(y+), log pi_theta(y-)
      2. reference forward (adapter OFF, no grad) -> log pi_ref(y+), log pi_ref(y-)
      3. DPO loss, divided by grad_accum_steps, then backward (gradients ADD UP)
  then: unscale -> clip gradient norm -> optimizer step -> zero gradients -> log
"""

import argparse
import math
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.models import (
    clear_gpu,
    count_parameters,
    load_policy,
    load_tokenizer,
    reference_mode,
    resolve_dtype,
    trainable_parameters,
)
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    """Turn a list of raw dataset rows into two padded tensor batches.

    For each row we build TWO sequences that share the same prompt:
        chosen   = prompt tokens + y+ tokens
        rejected = prompt tokens + y- tokens
    `encode_prompt_response` also returns a `response_mask` (1 on response tokens,
    0 on prompt tokens) so later sums run over response tokens only.
    `pad_batch` left-pads every sequence to the longest one in the batch and marks
    padding with 0 in attention_mask / response_mask.
    """
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


# A pair is usable only if its prompt leaves room for a meaningful response.
# encode_prompt_response() RAISES when the prompt alone fills max_length (54/1500 train
# prompts do), and a prompt that leaves only a handful of tokens would compare the two
# responses on their first few tokens only. One rule, used identically by training AND
# evaluation, with the dropped IDs saved so the exclusion is reproducible and reportable.
MIN_RESPONSE_TOKENS = 64


def filter_fitting(rows, tokenizer, max_length, min_response=MIN_RESPONSE_TOKENS):
    """Return (kept_rows, kept_original_indices, dropped_records)."""
    kept, kept_idx, dropped = [], [], []
    for i, row in enumerate(rows):
        n = len(tokenizer.apply_chat_template(prompt_messages_from_preference(row),
                                              tokenize=True, add_generation_prompt=True))
        if n <= max_length - min_response:
            kept.append(row); kept_idx.append(i)
        else:
            dropped.append({"index": i, "prompt_id": row.get("prompt_id"), "prompt_tokens": n})
    return kept, kept_idx, dropped


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    # Fixing the seed makes the LoRA initialisation AND the shuffle order identical
    # across runs, so beta forks differ ONLY in beta.
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    tokenizer = load_tokenizer(cfg["base_model"])
    rows, kept_idx, dropped = filter_fitting(rows, tokenizer, int(cfg["max_sequence_length"]))
    if max_examples is not None:
        # First N USABLE rows in file order -> the same fixed subset for every fork.
        rows, kept_idx = rows[: int(max_examples)], kept_idx[: int(max_examples)]

    # fresh_lora=True: start from the ORIGINAL Qwen weights with a brand-new
    # (untrained) LoRA adapter. The reference policy is this same model with the
    # adapter switched off.
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
        generator=torch.Generator().manual_seed(int(cfg["seed"])),  # reproducible shuffle
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
        "dataset_path": str(path),
        "dropped": dropped,
        "kept_indices": kept_idx,
    }


def _to_device(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    model, loader, optimizer = bundle["model"], bundle["loader"], bundle["optimizer"]
    beta = bundle["beta"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    results_dir = repo_path(cfg["results_dir"]) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()  # start a clean log for this run (append_jsonl appends)

    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"

    # ---------------------------------------------------------------------
    # Mixed precision setup.
    # The base model is stored in fp16 (config dtype: float16). fp16 can only
    # represent numbers down to ~6e-8, so small gradients round to zero and
    # Adam's tiny updates get lost. Two standard fixes:
    #   (a) keep the TRAINABLE LoRA weights in fp32 (they are tiny, ~1M params)
    #   (b) use a GradScaler: multiply the loss by a large factor before
    #       backward so gradients stay representable, divide back before the step.
    # If your GPU supports bf16 you could switch dtype instead; record which you used.
    # ---------------------------------------------------------------------
    for p in trainable_parameters(model):
        p.data = p.data.float()
    amp_dtype = resolve_dtype(cfg.get("dtype", "float16"))   # fp16 on a T4
    use_cuda = use_cuda and amp_dtype != torch.float32       # fp32 config -> no autocast needed
    use_scaler = use_cuda and amp_dtype == torch.float16     # bf16 has fp32's range -> no scaler
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    accum = int(cfg["grad_accum_steps"])
    max_grad_norm = float(cfg["max_grad_norm"])
    epochs = int(cfg["epochs"])
    micro_per_epoch = len(loader)
    total_opt_steps = epochs * math.ceil(micro_per_epoch / accum)

    total, trainable = count_parameters(model)
    print(f"[{run_name}] pairs={len(bundle['rows'])} (dropped {len(bundle['dropped'])} too-long prompts) beta={beta} "
          f"micro-batches/epoch={micro_per_epoch} optimizer steps={total_opt_steps} "
          f"trainable params={trainable:,}/{total:,}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    opt_step = 0
    window = []          # diagnostics from the micro-batches of the current accumulation window

    for epoch in range(epochs):
        for micro_idx, (chosen, rejected) in enumerate(loader):
            chosen, rejected = _to_device(chosen, device), _to_device(rejected, device)

            # ---- 1. Policy log-probs (WITH gradients) --------------------
            # response_sequence_logprobs shifts by one position (token t is
            # predicted from tokens < t), takes log-softmax, picks the log-prob
            # of the actual next token, and sums ONLY where response_mask == 1.
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_cuda):
                pol_c, _, _ = response_sequence_logprobs(model, chosen)
                pol_r, _, _ = response_sequence_logprobs(model, rejected)

            # ---- 2. Reference log-probs (NO gradients) -------------------
            # reference_mode() temporarily disables the LoRA adapter -> the
            # model behaves exactly like the original Qwen = pi_ref.
            # no_grad(): the reference is frozen, so we skip gradient bookkeeping
            # (saves a lot of memory).
            with torch.no_grad(), reference_mode(model), torch.autocast("cuda", dtype=amp_dtype, enabled=use_cuda):
                ref_c, _, _ = response_sequence_logprobs(model, chosen)
                ref_r, _, _ = response_sequence_logprobs(model, rejected)

            # ---- 3. Loss + backward --------------------------------------
            loss, diag = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)

            # Divide by the number of micro-batches in this window so the summed
            # gradient equals the gradient of the AVERAGE loss (as if one big batch).
            # The last window of an epoch may be shorter -> divide by its real size.
            window_start = (micro_idx // accum) * accum
            window_size = min(accum, micro_per_epoch - window_start)
            scaler.scale(loss / window_size).backward()

            window.append({
                "loss": float(loss.item()),
                **{k: float(v.item()) for k, v in diag.items()},
                "chosen_len": float(chosen["response_mask"].sum(-1).mean().item()),
                "rejected_len": float(rejected["response_mask"].sum(-1).mean().item()),
            })

            # ---- 4. Optimizer step at the end of each window -------------
            end_of_window = (micro_idx + 1) % accum == 0 or (micro_idx + 1) == micro_per_epoch
            if end_of_window:
                scaler.unscale_(optimizer)   # undo the loss scaling so clipping sees true gradients
                # clip_grad_norm_ returns the norm BEFORE clipping -> good stability diagnostic
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
                scaler.step(optimizer)       # skips the step automatically if grads are inf/NaN
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                record = {"run": run_name, "epoch": epoch, "opt_step": opt_step,
                          "beta": beta, "grad_norm": float(grad_norm),
                          "elapsed_s": time.perf_counter() - t0}
                # average each diagnostic over the micro-batches in this window
                for key in window[0]:
                    record[key] = sum(w[key] for w in window) / len(window)
                append_jsonl(log_path, record)
                window = []

                if opt_step % 10 == 0 or opt_step == total_opt_steps:
                    print(f"[{run_name}] step {opt_step}/{total_opt_steps} "
                          f"loss={record['loss']:.4f} acc={record['preference_accuracy']:.3f} "
                          f"margin={record['margin_mean']:.3f} gnorm={record['grad_norm']:.3f}")

    wall = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

    # save_pretrained on a PEFT model saves ONLY the LoRA adapter (a few MB),
    # not the 1.5B base weights. Load it later with load_policy(adapter_path=...).
    model.save_pretrained(str(output))
    save_json(results_dir / "data_used.json", {
        "dataset": bundle["dataset_path"], "min_response_tokens": MIN_RESPONSE_TOKENS,
        "used_indices": bundle["kept_indices"], "dropped": bundle["dropped"],
    })
    save_json(results_dir / "train_summary.json", {
        "run": run_name, "beta": beta, "dataset": bundle["dataset_path"],
        "num_pairs": len(bundle["rows"]), "num_dropped_too_long": len(bundle["dropped"]),
        "epochs": epochs,
        "batch_size": int(cfg["batch_size"]), "grad_accum_steps": accum,
        "effective_batch": int(cfg["batch_size"]) * accum,
        "optimizer_steps": opt_step, "learning_rate": float(cfg["learning_rate"]),
        "seed": int(cfg["seed"]), "precision": "fp16+GradScaler, fp32 LoRA" if use_scaler else str(amp_dtype),
        "wall_clock_s": wall, "peak_vram_gb": peak_gb, "output": str(output),
    })
    print(f"[{run_name}] done in {wall/60:.1f} min, peak VRAM {peak_gb:.2f} GB -> {output}")

    # Free GPU memory so the next fork / evaluation in the same process fits.
    del model, optimizer, loader, bundle
    clear_gpu()
    return str(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
