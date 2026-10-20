from __future__ import annotations

"""Task 3 Step 3: canonical GRPO (divide by each completion's length T_k) vs Dr.-GRPO-style
normalisation (divide by the constant max_completion_length), as matched short forks.

Both forks: identical supplied midpoint, prompt schedule, per-update sampling seed, reward model,
beta, eps, K, generation settings and `fork_updates` updates. Only `loss_type` differs.

Length-conditioned evidence (the direct test of the normalisation effect):
  every update logs, for EACH completion, its length, advantage and its OWN gradient norm (each
  completion is back-propagated separately). We report, per loss type:
    corr_len_gradnorm      correlation between completion length and its gradient norm
    long_over_short        mean gradient norm of long vs short completions (split at the pooled
                           median length), and the same separately for negative-advantage ("wrong")
                           and positive-advantage completions
    update1_paired         at update 1 both forks start from the SAME midpoint with the SAME seed,
                           so they sample IDENTICAL completions: the per-completion gradient ratio
                           dr_grpo / grpo vs length isolates the normalisation exactly
  Expected from the maths: canonical GRPO gives every completion total weight |A| (length-free);
  Dr. GRPO gives total weight |A| * T / max_len, so gradient norm should grow with length.
Then the common held-out evaluation for both forks.
"""

import argparse
import csv
import json

import numpy as np

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_policy

FORKS = {"grpo": "fork_grpo", "dr_grpo": "fork_dr_grpo"}


def completions(run, results_dir):
    out = []
    for l in open(repo_path(results_dir) / run / "train_log.jsonl"):
        if not l.strip():
            continue
        r = json.loads(l)
        for c in r["per_completion"]:
            if not c["masked"] and c["grad_norm"] > 0:
                out.append({**c, "update": r["update"]})
    return out


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else float("nan")


def length_conditioned(cfg):
    C = {lt: completions(name, cfg["results_dir"]) for lt, name in FORKS.items()}
    med = float(np.median([c["length"] for cs in C.values() for c in cs]))
    out = {"median_length_split": med}
    for lt, cs in C.items():
        L = np.array([c["length"] for c in cs]); G = np.array([c["grad_norm"] for c in cs]); A = np.array([c["adv"] for c in cs])
        def ratio(sel):
            s, l_ = G[sel & (L <= med)], G[sel & (L > med)]
            return float(l_.mean() / s.mean()) if len(s) and len(l_) and s.mean() > 0 else float("nan")
        allm = np.ones_like(L, dtype=bool)
        out[lt] = {"n_completions": int(len(cs)), "corr_len_gradnorm": corr(L, G),
                   "corr_len_gradnorm_per_unit_adv": corr(L, G / np.maximum(np.abs(A), 1e-6)),
                   "long_over_short": ratio(allm), "long_over_short_negative_adv": ratio(A < 0),
                   "long_over_short_positive_adv": ratio(A > 0),
                   "mean_grad_norm_short": float(G[L <= med].mean()) if (L <= med).any() else float("nan"),
                   "mean_grad_norm_long": float(G[L > med].mean()) if (L > med).any() else float("nan"),
                   "mean_length": float(L.mean())}
    u1 = {lt: [c for c in cs if c["update"] == 1] for lt, cs in C.items()}
    pairs = [(a["length"], b["grad_norm"] / a["grad_norm"]) for a, b in zip(u1["grpo"], u1["dr_grpo"])
             if a["length"] == b["length"] and a["grad_norm"] > 0]
    out["update1_paired"] = {"n": len(pairs), "identical_samples": len(pairs) == len(u1["grpo"]) > 0,
                             "length_and_ratio_dr_over_grpo": pairs,
                             "corr_len_ratio": corr([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) > 2 else float("nan")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    print("Fork updates:", cfg["fork_updates"], "| loss types:", list(FORKS))
    rows = []
    for lt, name in FORKS.items():
        out = f"outputs/task3_grpo/{name}"
        if not (args.skip_existing and (repo_path(out) / "adapter_config.json").exists()):
            run_grpo(args.config, output=out, updates=int(cfg["fork_updates"]), loss_type=lt, run_name=name)
        mpath = res / name / "eval_metrics.json"
        m = load_json(mpath) if (args.skip_existing and mpath.exists()) else evaluate_policy(args.config, out, name, args.n_prompts)
        s = load_json(res / name / "train_summary.json")
        rows.append({"condition": name, "loss_type": lt, "updates": s["updates"], "generated_tokens": s["generated_tokens"],
                     "wasted_token_fraction": s["wasted_token_fraction"], "heldout_reward": m["reward_mean"],
                     "heldout_kl": m["kl_token_mean"], "heldout_entropy": m["entropy_sampled"], "heldout_len": m["length_tokens"]["mean"],
                     "heldout_len_std": m["length_tokens"]["std"], "heldout_trunc": m["truncation_rate"], "heldout_distinct2": m["distinct2_mean"]})
    LC = length_conditioned(cfg)
    save_json(res / "normalization_study.json", {"forks": rows, "length_conditioned": LC})
    with open(res / "normalization_forks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for lt in FORKS:
        d = LC[lt]
        print(f"  {lt:8s} corr(length, grad norm) {d['corr_len_gradnorm']:+.3f} | long/short grad norm {d['long_over_short']:.2f} "
              f"(wrong answers {d['long_over_short_negative_adv']:.2f})")
    print(f"  update-1 paired (identical samples: {LC['update1_paired']['identical_samples']}): corr(length, dr/grpo ratio) {LC['update1_paired']['corr_len_ratio']:+.3f}")


if __name__ == "__main__":
    main()
