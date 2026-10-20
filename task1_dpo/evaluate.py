from __future__ import annotations

"""Task 1 evaluation: one function, `evaluate_adapter`, used for EVERY DPO condition
(standard, the three beta forks, length-balanced, and optionally the SFT baseline),
so all conditions are measured with an identical protocol.

Metrics produced
----------------
A. On held-out PREFERENCE PAIRS (teacher forcing, no generation):
   - DPO loss at the run's beta
   - preference accuracy  = fraction of pairs with m_theta > 0
   - mean margin m_theta
B. On GENERATED responses to a fixed set of held-out prompts:
   - KL from the reference (course sampled estimator: mean over response tokens of
     log pi_theta - log pi_ref, i.e. common.metrics.sampled_kl)
   - reward-model score (fixed course reward model)
   - response length: mean, std, median, IQR
C. On the 10 fixed word-limit prompts: word-limit compliance rate.
All per-example records are saved so you can pick qualitative examples later.
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from common.data import load_yaml, prompt_messages, prompt_messages_from_preference, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_sequence_logprobs, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import word_count, word_limit_compliance
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.train import filter_fitting, make_collate


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def length_stats(lengths):
    """Mean plus two dispersion statistics (std and inter-quartile range).
    IQR = 75th percentile - 25th percentile: the spread of the middle half,
    robust to a few extremely long responses."""
    a = np.asarray(lengths, dtype=float)
    if len(a) == 0:
        return {}
    q25, q50, q75 = np.percentile(a, [25, 50, 75])
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "median": float(q50), "iqr": float(q75 - q25), "n": int(len(a))}


@torch.no_grad()
def preference_metrics(model, tokenizer, rows, beta, max_length, batch_size=4, indices=None):
    """Teacher-forced scoring of held-out (x, y+, y-) pairs.

    For every pair we compute the four sequence log-probs and the margin
        m = [log pi_theta(y+) - log pi_ref(y+)] - [log pi_theta(y-) - log pi_ref(y-)]
    exactly as in the manual. For the untouched SFT model the adapter-off and
    adapter-on models are identical, so m = 0 and accuracy = 0 (strict "> 0").
    """
    collate = make_collate(tokenizer, max_length)
    device = next(model.parameters()).device
    indices = list(range(len(rows))) if indices is None else indices   # original positions in the file
    records = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        chosen, rejected = (_to_device(b, device) for b in collate(chunk))
        pc, _, _ = response_sequence_logprobs(model, chosen)
        pr, _, _ = response_sequence_logprobs(model, rejected)
        with reference_mode(model):                      # adapter OFF = pi_ref
            rc, _, _ = response_sequence_logprobs(model, chosen)
            rr, _, _ = response_sequence_logprobs(model, rejected)
        margin = (pc - rc) - (pr - rr)
        clen = chosen["response_mask"].sum(-1)
        rlen = rejected["response_mask"].sum(-1)
        for i, row in enumerate(chunk):
            records.append({
                "index": indices[start + i],              # position in the ORIGINAL eval file
                "prompt_id": row.get("prompt_id", row.get("id")),
                "margin": float(margin[i]),
                "chosen_logratio": float(pc[i] - rc[i]),
                "rejected_logratio": float(pr[i] - rr[i]),
                "chosen_len": int(clen[i]), "rejected_len": int(rlen[i]),
            })
    m = torch.tensor([r["margin"] for r in records])
    return {
        "dpo_loss": float(-F.logsigmoid(beta * m).mean()),
        "preference_accuracy": float((m > 0).float().mean()),
        "margin_mean": float(m.mean()),
        "n_pairs": len(records),
    }, records


@torch.no_grad()
def generation_metrics(model, tokenizer, rm, rm_tok, prompts, cfg, batch_size=4, seed=None):
    """Generate one response per prompt, then measure KL, reward and length.

    KL estimator (course convention): for each generated response token,
        log pi_theta(token) - log pi_ref(token)
    then average over ALL valid response tokens in the evaluation set
    (token-level mean = what common.metrics.sampled_kl computes). We also save the
    per-sequence SUM so you can report it if you prefer, but use ONE convention for
    every condition.
    """
    gen = cfg["generation"]
    # If a prompt is longer than max_prompt_length, cut from the LEFT (start of the
    # prompt). The default (right) would cut off the "<|im_start|>assistant" header at
    # the end, and the model would no longer know it is supposed to answer.
    tokenizer.truncation_side = "left"
    if seed is not None:
        set_seed(seed)        # same seed before every model's generation -> fair comparison
    out_rows = []
    kl_num, kl_den = 0.0, 0.0
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        out = batch_generate(model, tokenizer, chunk,
                             max_prompt_length=int(cfg.get("max_prompt_length", cfg["max_sequence_length"])),
                             max_new_tokens=int(cfg["max_generation_tokens"]),
                             temperature=float(gen["temperature"]), top_p=float(gen["top_p"]),
                             do_sample=bool(gen["do_sample"]))
        args = (out["sequences"], out["attention_mask"], out["prompt_width"], out["response_ids"])
        pol_lp, _ = response_token_logprobs(model, *args)
        with reference_mode(model):
            ref_lp, _ = response_token_logprobs(model, *args)
        mask = out["response_mask"]
        diff = (pol_lp - ref_lp) * mask
        kl_num += float(diff.sum()); kl_den += float(mask.sum())
        rewards = score_reward_pairs(rm, rm_tok, chunk, out["responses"],
                                     max_length=int(cfg.get("reward_max_length", 1024)))
        for i, p in enumerate(chunk):
            out_rows.append({
                "prompt": p[-1]["content"] if isinstance(p, list) else p,
                "response": out["responses"][i],
                "length_tokens": out["response_lengths"][i],
                "truncated": out["truncated"][i],
                "reward": float(rewards[i]),
                "kl_seq_sum": float(diff[i].sum()),
            })
        del out, pol_lp, ref_lp
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {
        "kl_token_mean": kl_num / max(kl_den, 1.0),
        "kl_seq_sum_mean": float(np.mean([r["kl_seq_sum"] for r in out_rows])),
        "reward_mean": float(np.mean([r["reward"] for r in out_rows])),
        "reward_std": float(np.std([r["reward"] for r in out_rows], ddof=1)),
        "truncation_rate": float(np.mean([r["truncated"] for r in out_rows])),
        "length_tokens": length_stats([r["length_tokens"] for r in out_rows]),
    }, out_rows


def word_limit_metrics(model, tokenizer, rm, rm_tok, cfg, seed):
    rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    prompts = [prompt_messages(r) for r in rows]
    summary, gens = generation_metrics(model, tokenizer, rm, rm_tok, prompts, cfg, seed=seed)
    for r, g in zip(rows, gens):
        g["prompt_id"] = r.get("prompt_id")
        g["words"] = word_count(g["response"])
        g["compliant"] = word_limit_compliance(g["prompt"], g["response"])
    valid = [g["compliant"] for g in gens if g["compliant"] is not None]
    return {"word_limit_compliance": float(np.mean(valid)) if valid else None,
            "mean_words": float(np.mean([g["words"] for g in gens])),
            "length_tokens": summary["length_tokens"], "n": len(gens)}, gens


def evaluate_adapter(config_path, adapter, name, beta=None, pair_rows=None, gen_prompts=200, batch_size=4):
    """Evaluate one condition and save everything under results/task1_dpo/<name>/.

    adapter=None or "none" -> the untouched SFT baseline (useful reference row).
    gen_prompts: how many held-out prompts to generate for (first N in file order).
    Keep this number IDENTICAL for every condition you compare.
    """
    cfg = load_yaml(config_path)
    beta = float(cfg["beta"] if beta is None else beta)
    seed = int(cfg["seed"])
    adapter = None if adapter in (None, "none", "None") else adapter
    out_dir = repo_path(cfg["results_dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm, rm_tok = load_reward_model(cfg)

    all_rows = pair_rows if pair_rows is not None else read_jsonl(cfg["paths"]["dpo_standard_eval"])
    # Same too-long-prompt rule as training (see train.filter_fitting); dropped IDs are saved.
    rows, kept_idx, dropped = filter_fitting(all_rows, tokenizer, int(cfg["max_sequence_length"]))
    save_json(out_dir / "eval_dropped.json", {"dropped": dropped, "n_kept": len(rows)})

    pref, pair_records = preference_metrics(model, tokenizer, rows, beta,
                                            int(cfg["max_sequence_length"]), batch_size, indices=kept_idx)
    write_jsonl(out_dir / "pairs.jsonl", pair_records)

    # Generation prompts: the first N USABLE held-out prompts, never truncated
    # (every kept prompt fits within max_sequence_length).
    prompts = [prompt_messages_from_preference(r) for r in rows[:gen_prompts]]
    gen, gen_records = generation_metrics(model, tokenizer, rm, rm_tok, prompts, cfg, batch_size, seed=seed)
    for i, g in enumerate(gen_records):
        g["index"] = kept_idx[i]             # position in the ORIGINAL eval file
    write_jsonl(out_dir / "generations.jsonl", gen_records)

    wl, wl_records = word_limit_metrics(model, tokenizer, rm, rm_tok, cfg, seed)
    write_jsonl(out_dir / "word_limit.jsonl", wl_records)

    metrics = {"name": name, "adapter": adapter, "beta": beta, "seed": seed,
               "eval_pairs_dropped_too_long": len(dropped),
               "gen_prompts": len(prompts), "max_generation_tokens": int(cfg["max_generation_tokens"]),
               "decoding": cfg["generation"],
               "preference": pref, "generation": gen, "word_limit": wl}
    save_json(out_dir / "metrics.json", metrics)
    print(f"[{name}] acc={pref['preference_accuracy']:.3f} loss={pref['dpo_loss']:.4f} "
          f"KL={gen['kl_token_mean']:.4f} reward={gen['reward_mean']:.3f} "
          f"len={gen['length_tokens']['mean']:.1f} wl={wl['word_limit_compliance']}")

    del model, rm
    clear_gpu()
    return metrics, pair_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help='adapter folder, or "none" for the SFT baseline')
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta used in training (for the held-out DPO loss)")
    ap.add_argument("--gen-prompts", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    evaluate_adapter(args.config, args.adapter, args.name, args.beta,
                     gen_prompts=args.gen_prompts, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
