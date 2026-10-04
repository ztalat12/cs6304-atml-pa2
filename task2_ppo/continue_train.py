from __future__ import annotations

"""Task 2: PPO continuation from the supplied midpoint.

One UPDATE (repeated `updates` times):
  1. ROLLOUT   sample `prompts_per_update` prompts (fixed seeded schedule) and let the
               policy generate up to `max_response_length` tokens.
  2. SCORE     reward model -> one number per answer; subtract `missing_eos_penalty`
               if the answer was cut off before EOS.
  3. LOG-PROBS log pi_old (policy now, frozen snapshot), log pi_ref (adapter OFF),
               V(s_t) from the critic -- all without gradients.
  4. SHAPE     per-token reward = -beta_KL (log pi_old - log pi_ref) + score at last token.
  5. GAE       advantages + returns (gamma, lambda), then whiten the advantages.
  6. OPTIMISE  `ppo_epochs` passes over the same rollout: clipped policy loss -> policy
               step; value MSE -> critic step. Diagnostics logged per pass.

Dropout is OFF for the whole run (policy.eval()): otherwise pi_new != pi_old even before
any update, and the epoch-0 ratio would not be exactly 1 (a built-in correctness test).
"""

import argparse
import math
import time

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import masked_mean, sample_entropy, sampled_kl
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    resolve_dtype,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import (
    active_clip_fraction,
    approx_kl,
    compute_gae,
    explained_variance,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    token_entropy,
    value_mse_loss,
)


def fork_name(eps: float, beta: float) -> str:
    """One name per (epsilon, beta_KL) fork, so the (0.2, 0.1) fork is shared by both studies."""
    return f"fork_eps{eps:g}_kl{beta:g}"


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    tokenizer.truncation_side = "left"      # an over-long prompt loses its START, never the assistant header
    # Every run (standard and every fork) starts from the SAME supplied midpoint adapter.
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    value_model = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"],
                                   train_mode=cfg.get("value_train_mode", "head_only"))
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    # fp16 storage underflows small Adam updates -> keep only the TRAINABLE weights in fp32
    # (forward passes still run in fp16 under autocast; see GradScaler in run_ppo).
    for p in trainable_parameters(policy):
        p.data = p.data.float()
    for p in trainable_parameters(value_model):
        p.data = p.data.float()

    policy_optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["policy_learning_rate"]))
    value_optimizer = AdamW(
        value_parameter_groups(value_model, lora_lr=float(cfg["value_lora_learning_rate"]),
                               head_lr=float(cfg["value_head_learning_rate"])),
        weight_decay=0.0,
    )
    return {"cfg": cfg, "tokenizer": tokenizer, "policy": policy, "value_model": value_model,
            "reward_model": reward_model, "reward_tokenizer": reward_tokenizer, "prompt_rows": prompts,
            "policy_optimizer": policy_optimizer, "value_optimizer": value_optimizer}


def prompt_schedule(n_rows, n_updates, per_update, seed):
    """Fixed, seeded prompt order. Update u always sees the same prompt(s) in every run,
    so forks differ only in the setting being studied."""
    order = np.random.RandomState(seed).permutation(n_rows)
    flat = [int(order[i % n_rows]) for i in range(n_updates * per_update)]
    return [flat[u * per_update:(u + 1) * per_update] for u in range(n_updates)]


def _amp(cfg):
    dtype = resolve_dtype(cfg.get("dtype", "float16"))
    on = torch.cuda.is_available() and dtype != torch.float32
    return dtype, on


@torch.no_grad()
def collect_rollout(bundle, prompt_rows, beta_kl, rollout_seed):
    """Steps 1-5 of one update. Returns everything the optimisation passes need."""
    cfg, tok = bundle["cfg"], bundle["tokenizer"]
    policy, critic = bundle["policy"], bundle["value_model"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    amp_dtype, amp_on = _amp(cfg)
    gen = cfg["generation"]
    msgs = [prompt_messages(r) for r in prompt_rows]

    policy.eval(); critic.eval()
    set_seed(rollout_seed)                         # same sampling randomness for update u in every fork
    roll = batch_generate(policy, tok, msgs, max_prompt_length=int(cfg["max_prompt_length"]),
                          max_new_tokens=int(cfg["max_response_length"]), temperature=float(gen["temperature"]),
                          top_p=float(gen["top_p"]), do_sample=bool(gen["do_sample"]))
    # batch_generate runs under torch.inference_mode(); tensors made there can never take part
    # in a backward pass. .clone() (outside inference mode) turns them into ordinary tensors,
    # which the gradient passes in ppo_pass need.
    seq, attn = roll["sequences"].clone(), roll["attention_mask"].clone()
    pw, rids = roll["prompt_width"], roll["response_ids"].clone()
    mask = roll["response_mask"].clone().float()
    T = rids.shape[1]

    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
        old_lp, logits = response_token_logprobs(policy, seq, attn, pw, rids)   # log pi_old, logits for entropy
        ent_exact = token_entropy(logits, mask)
        del logits
        with reference_mode(policy):                                         # adapter OFF = pi_ref
            ref_lp, ref_logits = response_token_logprobs(policy, seq, attn, pw, rids)
            del ref_logits
        # The critic reads the same token ids (Qwen2.5 models share one tokenizer).
        # V(s_t) for generated token t is read at position (prompt_width - 1 + t): the last
        # position BEFORE token t is produced -- the same alignment as the policy logits.
        all_values = token_values(critic, seq, attn).float()
    values = all_values[:, pw - 1: pw - 1 + T] * mask
    old_lp, ref_lp = old_lp.float(), ref_lp.float()

    raw_reward = score_reward_pairs(rm, rm_tok, msgs, roll["responses"],
                                    max_length=int(cfg.get("reward_max_length", 1024))).float().to(seq.device)
    truncated = torch.tensor(roll["truncated"], dtype=torch.float32, device=seq.device)
    task_reward = raw_reward - float(cfg.get("missing_eos_penalty", 0.0)) * truncated

    rewards = shaped_rewards(task_reward, old_lp, ref_lp, mask, beta_kl)
    adv, returns = compute_gae(rewards, values, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    adv_w = normalize_advantages(adv, mask)

    return {"msgs": msgs, "rows": prompt_rows, "seq": seq, "attn": attn, "pw": pw, "rids": rids, "mask": mask,
            "old_lp": old_lp, "ref_lp": ref_lp, "values": values, "returns": returns, "adv": adv, "adv_w": adv_w,
            "raw_reward": raw_reward, "task_reward": task_reward, "truncated": truncated,
            "responses": roll["responses"], "lengths": roll["response_lengths"], "entropy_exact": float(ent_exact),
            "kl_ref": float(sampled_kl(old_lp, ref_lp, mask)),              # course estimator, token mean
            "kl_ref_seq": float(((old_lp - ref_lp) * mask).sum(-1).mean()),
            "entropy_sampled": float(sample_entropy(old_lp, mask)),
            "explained_variance": explained_variance(values, returns, mask)}


def ppo_pass(bundle, ro, eps, scalers):
    """One optimisation pass (step 6) over a collected rollout: one policy step + one critic step."""
    cfg, policy, critic = bundle["cfg"], bundle["policy"], bundle["value_model"]
    pol_opt, val_opt = bundle["policy_optimizer"], bundle["value_optimizer"]
    amp_dtype, amp_on = _amp(cfg)
    max_gn = float(cfg["max_grad_norm"])
    mask = ro["mask"]

    # ---- policy: clipped surrogate ------------------------------------------------
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
        new_lp, logits = response_token_logprobs(policy, ro["seq"], ro["attn"], ro["pw"], ro["rids"])
    del logits
    new_lp = new_lp.float()
    pol_loss, ratio, clip_frac = ppo_policy_loss(new_lp, ro["old_lp"], ro["adv_w"], mask, eps)
    act_frac = active_clip_fraction(ratio, ro["adv_w"], mask, eps)
    a_kl = approx_kl(new_lp.detach(), ro["old_lp"], mask)
    max_dev = float(((ratio - 1).abs() * mask).max())
    pol_opt.zero_grad(set_to_none=True)
    scalers["policy"].scale(pol_loss).backward()
    scalers["policy"].unscale_(pol_opt)
    gnorm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_gn)   # norm BEFORE clipping
    scalers["policy"].step(pol_opt)            # skipped automatically if the gradient overflowed
    scalers["policy"].update()

    # ---- critic: regress V(s_t) onto the GAE returns -------------------------------
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
        v_all = token_values(critic, ro["seq"], ro["attn"])
    T = ro["rids"].shape[1]
    v_new = v_all.float()[:, ro["pw"] - 1: ro["pw"] - 1 + T]
    v_loss = float(cfg.get("value_coef", 0.5)) * value_mse_loss(v_new, ro["returns"], mask)
    val_opt.zero_grad(set_to_none=True)
    scalers["value"].scale(v_loss).backward()
    scalers["value"].unscale_(val_opt)
    vgnorm = torch.nn.utils.clip_grad_norm_([p for g in val_opt.param_groups for p in g["params"]], max_gn)
    scalers["value"].step(val_opt)
    scalers["value"].update()

    return {"policy_loss": float(pol_loss), "value_loss": float(v_loss), "clip_fraction": float(clip_frac),
            "active_clip_fraction": float(act_frac), "approx_kl_old_new": float(a_kl),
            "max_abs_ratio_minus_1": max_dev, "grad_norm": float(gnorm), "value_grad_norm": float(vgnorm)}


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None,
            kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    eps, beta_kl, n_updates = float(cfg["clip_epsilon"]), float(cfg["kl_beta"]), int(cfg["updates"])
    seed, per_update = int(cfg["seed"]), int(cfg["prompts_per_update"])

    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    res = repo_path(cfg["results_dir"]) / run_name
    res.mkdir(parents=True, exist_ok=True)
    for f in ["train_log.jsonl", "rollouts.jsonl"]:
        if (res / f).exists():
            (res / f).unlink()

    amp_dtype, amp_on = _amp(cfg)
    use_scaler = amp_on and amp_dtype == torch.float16
    scalers = {"policy": torch.amp.GradScaler("cuda", enabled=use_scaler),
               "value": torch.amp.GradScaler("cuda", enabled=use_scaler)}
    schedule = prompt_schedule(len(bundle["prompt_rows"]), n_updates, per_update, seed)
    print(f"[{run_name}] PPO continuation: {n_updates} updates x {per_update} prompt(s), "
          f"{cfg['ppo_epochs']} passes each | eps={eps} beta_KL={beta_kl} | from {cfg['paths']['ppo_midpoint_policy']}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    gen_tokens = 0
    for u in range(n_updates):
        rows = [bundle["prompt_rows"][i] for i in schedule[u]]
        ro = collect_rollout(bundle, rows, beta_kl, rollout_seed=seed + u)
        gen_tokens += int(sum(ro["lengths"]))
        if ro["mask"].sum() == 0:
            print(f"[{run_name}] update {u+1}: empty response, skipped"); continue
        passes = [ppo_pass(bundle, ro, eps, scalers) for _ in range(int(cfg["ppo_epochs"]))]

        rec = {"run": run_name, "update": u + 1, "eps": eps, "beta_kl": beta_kl, "prompt_indices": schedule[u],
               "reward": float(ro["raw_reward"].mean()), "reward_penalized": float(ro["task_reward"].mean()),
               "truncated": float(ro["truncated"].mean()), "response_length": float(np.mean(ro["lengths"])),
               "kl_ref": ro["kl_ref"], "kl_ref_seq": ro["kl_ref_seq"],
               "entropy": ro["entropy_sampled"], "entropy_exact": ro["entropy_exact"],
               "value_mean": float(masked_mean(ro["values"], ro["mask"])),
               "return_mean": float(masked_mean(ro["returns"], ro["mask"])),
               "explained_variance": ro["explained_variance"],
               "adv_raw_mean": float(masked_mean(ro["adv"], ro["mask"])),
               # per-pass lists (pass 0 must have ratio == 1 exactly) ...
               "passes": passes,
               # ... and the per-update summaries the manual asks for
               "policy_loss": passes[-1]["policy_loss"], "value_loss": passes[-1]["value_loss"],
               "clip_fraction": float(np.mean([p["clip_fraction"] for p in passes])),
               "clip_fraction_last": passes[-1]["clip_fraction"],
               "active_clip_fraction_last": passes[-1]["active_clip_fraction"],
               "approx_kl_old_new": max(p["approx_kl_old_new"] for p in passes),
               "grad_norm": float(np.mean([p["grad_norm"] for p in passes])),
               "elapsed_s": time.perf_counter() - t0}
        append_jsonl(res / "train_log.jsonl", rec)
        for i, r in enumerate(rows):
            append_jsonl(res / "rollouts.jsonl", {"update": u + 1, "prompt_id": r.get("prompt_id"),
                                                  "prompt": ro["msgs"][i][-1]["content"], "response": ro["responses"][i],
                                                  "reward": float(ro["raw_reward"][i]), "truncated": bool(ro["truncated"][i]),
                                                  "length": int(ro["lengths"][i])})
        print(f"[{run_name}] update {u+1:2d}/{n_updates} reward={rec['reward']:+.3f} KL={rec['kl_ref']:.4f} "
              f"H={rec['entropy']:.3f} len={rec['response_length']:.0f}{' (cut)' if rec['truncated'] else ''} "
              f"clip={rec['clip_fraction_last']:.3f} vloss={rec['value_loss']:.3f} EV={rec['explained_variance']:+.2f} "
              f"gnorm={rec['grad_norm']:.2f} pass0|rho-1|={passes[0]['max_abs_ratio_minus_1']:.1e}")
        del ro
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    bundle["policy"].save_pretrained(str(out))        # adapter only (a few MB)
    save_json(res / "train_summary.json", {
        "run": run_name, "updates": n_updates, "prompts_per_update": per_update, "ppo_epochs": int(cfg["ppo_epochs"]),
        "eps": eps, "beta_kl": beta_kl, "gamma": float(cfg["gamma"]), "gae_lambda": float(cfg["gae_lambda"]),
        "policy_lr": float(cfg["policy_learning_rate"]), "value_lora_lr": float(cfg["value_lora_learning_rate"]),
        "value_head_lr": float(cfg["value_head_learning_rate"]), "seed": seed,
        "max_response_length": int(cfg["max_response_length"]), "generated_tokens": gen_tokens,
        "precision": "fp16 autocast + GradScaler, fp32 trainable weights" if use_scaler else str(amp_dtype),
        "wall_clock_s": wall, "peak_vram_gb": peak, "start": cfg["paths"]["ppo_midpoint_policy"], "output": str(out)})
    print(f"[{run_name}] done: {wall/60:.1f} min, peak VRAM {peak:.2f} GB, {gen_tokens} generated tokens -> {out}")
    del bundle
    clear_gpu()
    return str(out)


def stability_stats(run_name, results_dir="results/task2_ppo"):
    """The clearly defined stability statistics used in the clipping study (from the train log):
       max_approx_kl : largest KL(old || new) moved by any single update (step-size blow-up)
       grad_norm_std : standard deviation of the per-update policy gradient norm
       clip_last_mean: mean fraction of tokens outside [1-eps, 1+eps] on the final pass"""
    import json
    log = [json.loads(l) for l in open(repo_path(results_dir) / run_name / "train_log.jsonl") if l.strip()]
    g = [r["grad_norm"] for r in log if math.isfinite(r["grad_norm"])]
    return {"max_approx_kl": max(r["approx_kl_old_new"] for r in log),
            "grad_norm_std": float(np.std(g)) if g else float("nan"),
            "clip_last_mean": float(np.mean([r["clip_fraction_last"] for r in log])),
            "nonfinite_grad_updates": sum(not math.isfinite(r["grad_norm"]) for r in log),
            "train_reward_mean": float(np.mean([r["reward"] for r in log]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
