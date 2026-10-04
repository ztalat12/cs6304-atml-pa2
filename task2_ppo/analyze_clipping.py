from __future__ import annotations

"""Task 2 Step 2: clipping study, epsilon in {0.05, 0.20, 0.50}.

PART A -- the supplied cached rollout batch (no generation; deterministic):
  1. rebuild every cached rollout as token ids (prompt + response) aligned with the cached
     old/reference log-probs;
  2. compute advantages exactly as training does: reward-model score (or the cached one),
     missing-EOS penalty, KL-shaped per-token reward from the CACHED old/ref log-probs, critic
     values from the supplied midpoint value model, GAE, whitening over the whole batch;
  3. ratio rho = pi_midpoint / pi_old(cache) on every token, then for each epsilon:
       clip_fraction   tokens with rho outside [1-eps, 1+eps]          (manual's definition)
       affected        tokens whose gradient the min actually zeroes   (same response mask)
       surrogate       clipped vs unclipped objective, and their gap
  4. extra (labelled as such): `inner_steps` further gradient passes on the cached batch for
     each epsilon, starting from the identical midpoint, to show how epsilon bounds how far
     token probabilities can move within one batch.
PART B -- matched short forks: `fork_updates` PPO updates from the identical midpoint for each
  epsilon (beta_KL fixed at the config value), then the common held-out evaluation.
"""

import argparse
import csv
import json

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs, score_reward_pairs
from common.logging_utils import load_json, save_json
from common.models import (clear_gpu, load_policy, load_reward_model, load_tokenizer, load_value_model,
                           token_values, trainable_parameters)
from task2_ppo.continue_train import fork_name, run_ppo, stability_stats
from task2_ppo.evaluate import evaluate_policy
from task2_ppo.ppo import active_clip_fraction, approx_kl, compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards

# Release cache schema (seen on Kaggle): clipped_at_max, effective_terminal_reward, old_logprobs,
# prompt_id, raw_terminal_reward, ref_logprobs, response (TEXT), response_tokens (a COUNT),
# source_index, terminated_with_eos, values (rollout-time critic values, one per response token).
ID_KEYS = ["response_ids", "response_token_ids", "response_tokens", "completion_ids"]   # used only if a SEQUENCE
# Terminal reward, in priority order. "effective" already includes the missing-EOS penalty.
REWARD_KEYS = [("effective_terminal_reward", True), ("raw_terminal_reward", False), ("reward", False),
               ("score", False), ("task_reward", False), ("rm_score", False)]


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)
    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def _is_seq(x):
    if torch.is_tensor(x):
        return x.dim() > 0 and x.numel() > 1
    return isinstance(x, (list, tuple))


def _as_list(x):
    if torch.is_tensor(x):
        return x.flatten().tolist()
    return list(x)


def _scalar(x):
    """A cached reward stored as a plain number or a 1-element tensor; anything else -> None."""
    if torch.is_tensor(x) and x.numel() == 1:
        return float(x)
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    return None


def find_prompt(row, pools):
    """Prompt messages for a cached row: stored directly if present, else looked up by
    source_index / prompt_id in the RL prompt pools (train first, then eval)."""
    if any(k in row for k in ["messages", "prompt"]):
        return prompt_messages(row), "stored in cache"
    for pool_name, pool in pools.items():
        for key in ["prompt_id", "source_index"]:
            if key in row:
                hit = [r for r in pool if r.get(key) == row[key]]
                if hit:
                    return prompt_messages(hit[0]), f"{pool_name} by {key}"
    raise KeyError(f"Cannot locate the prompt for cached row with keys {sorted(row)}")


def rebuild(rows, tok, pools, max_prompt_length):
    """Token ids aligned with the cached log-probs, plus a reconstruction report."""
    eos = tok.eos_token_id
    out, report = [], {"prompt_source": {}, "ids_from": {}, "length_mismatch_fixed": 0, "length_mismatch_cut": 0}
    for row in rows:
        msgs, src = find_prompt(row, pools)
        report["prompt_source"][src] = report["prompt_source"].get(src, 0) + 1
        old = torch.tensor(_as_list(row["old_logprobs"]), dtype=torch.float32)
        ref = torch.tensor(_as_list(row["ref_logprobs"]), dtype=torch.float32)
        key = next((k for k in ID_KEYS if k in row and _is_seq(row[k])), None)
        if key is not None:
            ids, how = [int(i) for i in _as_list(row[key])], key
        elif not isinstance(row["response"], str):
            ids, how = [int(i) for i in _as_list(row["response"])], "response (token ids)"
        else:
            ids, how = tok(row["response"], add_special_tokens=False)["input_ids"], "re-tokenised text"
            if len(ids) + 1 == len(old) and eos is not None:
                ids = ids + [eos]; report["length_mismatch_fixed"] += 1
        report["ids_from"][how] = report["ids_from"].get(how, 0) + 1
        n = min(len(ids), len(old), len(ref))
        if not (len(ids) == len(old) == len(ref)):
            report["length_mismatch_cut"] += 1
        ids, old, ref = ids[:n], old[:n], ref[:n]
        mask = torch.tensor(_as_list(row["response_mask"])[:n], dtype=torch.float32) if "response_mask" in row else torch.ones(n)
        p_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)[-max_prompt_length:]
        cached_r, penalized = None, False
        for k, pen_included in REWARD_KEYS:
            if k in row and _scalar(row[k]) is not None:
                cached_r, penalized = _scalar(row[k]), pen_included
                break
        cnt = _scalar(row["response_tokens"]) if "response_tokens" in row and not _is_seq(row["response_tokens"]) else None
        if cnt is not None and int(cnt) != n:
            report["count_mismatch"] = report.get("count_mismatch", 0) + 1
        vals = None
        if "values" in row and _is_seq(row["values"]):
            vv = torch.tensor(_as_list(row["values"]), dtype=torch.float32)
            if len(vv) >= n:
                vals = vv[:n]
        ends_eos = bool(row["terminated_with_eos"]) if "terminated_with_eos" in row else bool(ids and ids[-1] == eos)
        out.append({"msgs": msgs, "prompt_ids": p_ids, "ids": ids, "old": old, "ref": ref, "mask": mask,
                    "text": tok.decode(ids, skip_special_tokens=True), "ends_eos": ends_eos,
                    "cached_reward": cached_r, "reward_includes_penalty": penalized, "cached_values": vals})
    return out, report


def _logp(policy, r, device):
    seq = torch.tensor([r["prompt_ids"] + r["ids"]], device=device)
    attn = torch.ones_like(seq)
    rids = torch.tensor([r["ids"]], device=device)
    lp, logits = response_token_logprobs(policy, seq, attn, len(r["prompt_ids"]), rids)
    del logits
    return lp[0].float(), seq, attn


def cached_batch_analysis(cfg, inner_steps=6):
    tok = load_tokenizer(cfg["base_model"])
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    pools = {"rl_prompt_train": read_jsonl(cfg["paths"]["rl_prompt_train"]),
             "rl_prompt_eval": read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    R, report = rebuild(rows, tok, pools, int(cfg["max_prompt_length"]))
    report.update({"n_rollouts": len(R), "cache_keys": sorted(rows[0].keys())})
    print("cache reconstruction:", json.dumps(report, indent=1))

    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    policy.eval()                                       # dropout off, as in training
    critic = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode="frozen")
    rm, rm_tok = load_reward_model(cfg)
    device = next(policy.parameters()).device
    beta, pen = float(cfg["kl_beta"]), float(cfg.get("missing_eos_penalty", 0.0))

    # ---- advantages, exactly as in training (but from the CACHED old/ref log-probs) ----
    with torch.no_grad():
        for r in R:
            lp, seq, attn = _logp(policy, r, device)
            r["mid"] = lp.cpu()
            pw, T = len(r["prompt_ids"]), len(r["ids"])
            if r["cached_values"] is not None:
                v = r["cached_values"]                      # rollout-time critic values from the cache
            else:
                v = token_values(critic, seq, attn).float()[0, pw - 1: pw - 1 + T].cpu()
            score = r["cached_reward"]
            if score is None:
                score = float(score_reward_pairs(rm, rm_tok, [r["msgs"]], [r["text"]],
                                                 max_length=int(cfg.get("reward_max_length", 1024)))[0])
            r["score"] = score
            already = r["cached_reward"] is not None and r["reward_includes_penalty"]
            task = torch.tensor([score - (0.0 if already else pen * (not r["ends_eos"]))])
            rew = shaped_rewards(task, r["old"][None], r["ref"][None], r["mask"][None], beta)
            adv, ret = compute_gae(rew, v[None] * r["mask"][None], r["mask"][None], float(cfg["gamma"]), float(cfg["gae_lambda"]))
            r["adv_raw"] = adv[0]
    del critic, rm
    clear_gpu()
    mask = torch.cat([r["mask"] for r in R])
    adv_w = normalize_advantages(torch.cat([r["adv_raw"] for r in R])[None], mask[None])[0]   # whiten over the batch
    old = torch.cat([r["old"] for r in R]); mid = torch.cat([r["mid"] for r in R])
    ratio0 = torch.exp(mid - old)
    m = mask.bool()
    ratio_stats = {"n_tokens": int(m.sum()), "mean": float(ratio0[m].mean()), "min": float(ratio0[m].min()),
                   "max": float(ratio0[m].max()), "p05": float(ratio0[m].quantile(0.05)), "p95": float(ratio0[m].quantile(0.95)),
                   "frac_within_1e-3_of_1": float(((ratio0[m] - 1).abs() < 1e-3).float().mean()),
                   "rewards_from": "cache" if R[0]["cached_reward"] is not None else "reward model (re-scored)",
                   "values_from": "cache (rollout-time critic)" if R[0]["cached_values"] is not None else "midpoint critic (recomputed)",
                   "mean_score": float(np.mean([r["score"] for r in R]))}

    # ---- static geometry per epsilon -----------------------------------------------
    static = []
    for eps in [float(e) for e in cfg["clip_values"]]:
        loss, ratio, clip = ppo_policy_loss(mid, old, adv_w, mask, eps)
        static.append({"eps": eps, "clip_fraction": float(clip),
                       "affected_token_fraction": float(active_clip_fraction(ratio, adv_w, mask, eps)),
                       "clipped_surrogate": float(-loss), "unclipped_surrogate": float((ratio * adv_w * mask).sum() / mask.sum()),
                       })
        static[-1]["surrogate_gap"] = static[-1]["unclipped_surrogate"] - static[-1]["clipped_surrogate"]

    # ---- extra: further inner passes on the cached batch, per epsilon ---------------
    snapshot = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}
    for p in trainable_parameters(policy):
        p.data = p.data.float()
    inner = []
    n_tok = float(mask.sum())
    offs = np.cumsum([0] + [len(r["ids"]) for r in R])
    for eps in [float(e) for e in cfg["clip_values"]]:
        with torch.no_grad():
            for n, p in policy.named_parameters():
                if n in snapshot:
                    p.data = snapshot[n].clone().float()
        opt = torch.optim.AdamW(trainable_parameters(policy), lr=float(cfg["policy_learning_rate"]))
        for k in range(inner_steps + 1):
            opt.zero_grad(set_to_none=True)
            news = []
            for j, r in enumerate(R):
                a = adv_w[offs[j]:offs[j + 1]].to(device); o = r["old"].to(device); mk = r["mask"].to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    lp, _, _ = _logp(policy, r, device)
                rho = torch.exp(lp - o)
                obj = torch.minimum(rho * a, rho.clamp(1 - eps, 1 + eps) * a)
                if k < inner_steps:
                    (-(obj * mk).sum() / n_tok).backward()
                news.append(lp.detach().cpu())
            new = torch.cat(news)
            _, ratio, clip = ppo_policy_loss(new, old, adv_w, mask, eps)
            inner.append({"eps": eps, "step": k, "clip_fraction": float(clip),
                          "affected_token_fraction": float(active_clip_fraction(ratio, adv_w, mask, eps)),
                          "mean_abs_log_ratio_vs_midpoint": float(((new - mid).abs() * mask).sum() / n_tok),
                          "approx_kl_midpoint_new": float(approx_kl(new, mid, mask))})
            if k < inner_steps:
                torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"]))
                opt.step()
    del policy
    clear_gpu()
    return {"reconstruction": report, "ratio_midpoint_vs_cache_old": ratio_stats, "static": static,
            "inner_passes": inner, "inner_steps": inner_steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--inner-steps", type=int, default=6)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--cache-only", action="store_true", help="Part A only (no forks)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    print("Required epsilon values:", cfg["clip_values"])

    cpath = res / "clip_cached_batch.json"
    if args.skip_existing and cpath.exists():
        A = load_json(cpath)
    else:
        A = cached_batch_analysis(cfg, args.inner_steps)
        save_json(cpath, A)
    for s in A["static"]:
        print(f"  eps={s['eps']:<5g} clip fraction {s['clip_fraction']:.4f} | affected {s['affected_token_fraction']:.4f} | "
              f"surrogate clipped {s['clipped_surrogate']:+.4f} vs unclipped {s['unclipped_surrogate']:+.4f}")
    if args.cache_only:
        return

    beta = float(cfg["kl_beta"]); rows = []
    for eps in [float(e) for e in cfg["clip_values"]]:
        name = fork_name(eps, beta)
        out = f"outputs/task2_ppo/{name}"
        if not (args.skip_existing and (repo_path(out) / "adapter_config.json").exists()):
            run_ppo(args.config, output=out, updates=int(cfg["fork_updates"]), clip_epsilon=eps, kl_beta=beta, run_name=name)
        mpath = res / name / "eval_metrics.json"
        m = load_json(mpath) if (args.skip_existing and mpath.exists()) else evaluate_policy(args.config, out, name, args.n_prompts)
        rows.append({"condition": name, "eps": eps, "beta_kl": beta, "updates": int(cfg["fork_updates"]),
                     "heldout_reward": m["reward_mean"], "heldout_reward_penalized": m["reward_penalized_mean"],
                     "heldout_kl": m["kl_token_mean"], "heldout_len": m["length_tokens"]["mean"],
                     "heldout_len_std": m["length_tokens"]["std"], "heldout_trunc": m["truncation_rate"],
                     "heldout_entropy": m["entropy_sampled"], **stability_stats(name, cfg["results_dir"]),
                     "generated_tokens": load_json(res / name / "train_summary.json")["generated_tokens"]})
    save_json(res / "clip_forks.json", rows)
    with open(res / "clip_forks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
