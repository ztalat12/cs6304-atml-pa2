from __future__ import annotations

"""Task 3 Step 2: group-size study at an EQUAL generation budget (no training, no GPU).

Budget rule (documented choice): the cache holds 8 completions per prompt. For every prompt we
take its first 8 completions in generation_index order and partition them into 8/K DISJOINT
consecutive groups of size K:
    K = 8 -> 1 group per prompt   K = 4 -> 2 groups   K = 2 -> 4 groups
so all three conditions use exactly the same P x 8 completions (equal total generations), and
differ only in how many siblings each completion is compared with.

Per K we report:
  informative_rate     groups with reward std > tol (1e-6, the advantage eps); also the share of
                       groups whose spread is merely SMALL (std < 0.05 / 0.1 / 0.25 reward units)
  mean_group_std       mean within-group reward std (population). Small groups underestimate it:
                       E[sigma_K^2] = (K-1)/K * sigma^2
  var_centered         variance of the raw relative signal r - mu_group (before dividing by sigma)
  var_adv, mean_abs_adv  of the normalised advantages A = (r - mu)/(sigma + eps)
  frac_adv_pm1         share of advantages that are exactly +-1 (always 100% when K = 2: two
                       distinct rewards give z-scores of +1 and -1 however close they are)
  sign_agreement       share of completions whose advantage sign agrees with the sign of
                       r - mu_8 (the verdict using ALL 8 siblings) -- how often a small group
                       gives the same "better / worse" signal as the full group (1 by definition at K=8)
  baseline_mse         mean (mu_group - mu_8)^2: noise in the group baseline relative to the 8-sample mean
Difficulty bins (defined once): terciles of each prompt's 8-completion mean reward -> hard / medium / easy.
"""

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json

REWARD_KEYS = ["reward", "raw_reward", "raw_terminal_reward", "score", "rm_score", "terminal_reward", "effective_terminal_reward"]
TRUNC_KEYS = ["truncated", "clipped_at_max", "hit_max_length", "is_truncated"]
TOL, SMALL = 1e-6, [0.05, 0.1, 0.25]


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def reward_key(row):
    for k in REWARD_KEYS:
        if k in row and isinstance(row[k], (int, float)):
            return k
    raise KeyError(f"No numeric reward field in cache row; keys: {sorted(row)}")


def trunc_flag(row):
    for k in TRUNC_KEYS:
        if k in row:
            return bool(row[k])
    if "terminated_with_eos" in row:
        return not bool(row["terminated_with_eos"])
    return None


def regroup_equal_generation_budget(by_prompt, k: int):
    """[(prompt_id, [rows of one K-group]), ...] using each prompt's first 8 completions, split into
    8/K disjoint consecutive groups (generation_index order)."""
    if 8 % k:
        raise ValueError("K must divide 8")
    out = []
    for pid in sorted(by_prompt, key=lambda x: (len(x), x)):
        first8 = by_prompt[pid][:8]
        out += [(pid, first8[j:j + k]) for j in range(0, 8, k)]
    return out


def z(r, eps=1e-6):
    r = np.asarray(r, dtype=float)
    return (r - r.mean()) / (r.std() + eps)


def stats_for(groups, rkey, mu8):
    stds, cent, adv, agree, bmse, n_signal, n_comp = [], [], [], [], [], 0, 0
    for pid, g in groups:
        r = np.array([x[rkey] for x in g], dtype=float)
        s = r.std(); a = z(r)
        stds.append(s); cent += list(r - r.mean()); adv += list(a)
        ref = r - mu8[pid]
        agree += [np.sign(ai) == np.sign(ri) for ai, ri in zip(a, ref) if abs(ri) > 1e-9]
        bmse.append((r.mean() - mu8[pid]) ** 2)
        tr = [trunc_flag(x) for x in g]
        n_comp += len(g)
        n_signal += sum(1 for ai, t in zip(a, tr) if abs(ai) > 1e-9 and not t)   # unmasked AND non-zero advantage
    stds, adv = np.array(stds), np.array(adv)
    return {"n_groups": int(len(groups)), "n_completions": int(n_comp),
            "informative_rate": float((stds > TOL).mean()), "uninformative_rate": float((stds <= TOL).mean()),
            **{f"small_spread_rate_lt{t:g}": float((stds < t).mean()) for t in SMALL},
            "mean_group_std": float(stds.mean()), "median_group_std": float(np.median(stds)),
            "var_centered": float(np.var(cent)), "var_adv": float(np.var(adv)), "mean_abs_adv": float(np.abs(adv).mean()),
            "max_abs_adv": float(np.abs(adv).max()), "frac_adv_pm1": float((np.abs(np.abs(adv) - 1) < 1e-3).mean()),
            "sign_agreement": float(np.mean(agree)) if agree else float("nan"), "baseline_mse": float(np.mean(bmse)),
            "frac_completions_with_signal_after_masking": n_signal / max(n_comp, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    first = next(iter(by_prompt.values()))
    rkey = reward_key(first[0])
    print("Cached prompts:", len(by_prompt), "| completions per prompt used: 8 | reward field:", rkey)
    print("Cache row keys:", sorted(first[0].keys()))
    print("Group sizes to analyze:", cfg["group_sizes"])

    mu8 = {pid: float(np.mean([x[rkey] for x in g[:8]])) for pid, g in by_prompt.items()}
    q1, q2 = np.quantile(list(mu8.values()), [1 / 3, 2 / 3])
    bin_of = {pid: ("hard" if m <= q1 else "medium" if m <= q2 else "easy") for pid, m in mu8.items()}
    trunc_all = [trunc_flag(x) for g in by_prompt.values() for x in g[:8]]
    summary = {"reward_field": rkey, "n_prompts": len(by_prompt), "completions_per_prompt": 8,
               "total_completions": 8 * len(by_prompt), "tolerance": TOL,
               "difficulty_rule": "terciles of each prompt's 8-completion mean reward",
               "difficulty_cutoffs": [float(q1), float(q2)],
               "bin_sizes": {b: sum(v == b for v in bin_of.values()) for b in ["hard", "medium", "easy"]},
               "truncated_fraction": (float(np.mean(trunc_all)) if None not in trunc_all else None),
               "cache_generation_cap": cfg.get("cache_generation_cap"), "per_k": {}, "per_k_bin": {}}
    rows_csv = []
    for k in [int(x) for x in cfg["group_sizes"]]:
        groups = regroup_equal_generation_budget(by_prompt, k)
        summary["per_k"][str(k)] = st = stats_for(groups, rkey, mu8)
        rows_csv.append({"K": k, "bin": "all", **st})
        for b in ["hard", "medium", "easy"]:
            gb = [(p, g) for p, g in groups if bin_of[p] == b]
            summary["per_k_bin"][f"{k}_{b}"] = sb = stats_for(gb, rkey, mu8)
            rows_csv.append({"K": k, "bin": b, **sb})
        print(f"  K={k}: {st['n_groups']} groups | informative {st['informative_rate']:.3f} | mean std {st['mean_group_std']:.3f} | "
              f"var(r-mu) {st['var_centered']:.3f} | +-1 advantages {st['frac_adv_pm1']:.2f} | sign agreement {st['sign_agreement']:.3f}")
    res = repo_path(cfg["results_dir"]); res.mkdir(parents=True, exist_ok=True)
    save_json(res / "group_size_study.json", summary)
    with open(res / "group_size_study.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0])); w.writeheader(); w.writerows(rows_csv)
    print("saved ->", res / "group_size_study.json")


if __name__ == "__main__":
    main()
