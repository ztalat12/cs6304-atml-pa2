from __future__ import annotations

"""Task 2 Step 3: reward-overoptimisation study, beta_KL in {0, 0.10, 0.20}.

Each condition: `fork_updates` PPO updates from the IDENTICAL supplied midpoint (same prompt
schedule, same per-update sampling seed, epsilon fixed at the config value), then the common
held-out evaluation. beta_KL = 0 removes the leash to the reference entirely.

What to look for: does held-out reward rise while the text gets worse (longer, cut off,
repetitive -> low distinct-2), entropy falls and KL rises? That pattern is reward
overoptimisation. The per-update train logs show which quantity moves FIRST (RQ2).
The (eps = config value, beta = 0.10) fork is shared with the clipping study (same name),
so it is trained once.
"""

import argparse
import csv

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from task2_ppo.continue_train import fork_name, run_ppo, stability_stats
from task2_ppo.evaluate import evaluate_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    eps = float(cfg["clip_epsilon"])
    print("KL beta conditions:", cfg["kl_values"], "| fork update budget:", cfg["fork_updates"], "| eps:", eps)

    rows = []
    for beta in [float(b) for b in cfg["kl_values"]]:
        name = fork_name(eps, beta)
        out = f"outputs/task2_ppo/{name}"
        if not (args.skip_existing and (repo_path(out) / "adapter_config.json").exists()):
            run_ppo(args.config, output=out, updates=int(cfg["fork_updates"]), clip_epsilon=eps, kl_beta=beta, run_name=name)
        mpath = res / name / "eval_metrics.json"
        m = load_json(mpath) if (args.skip_existing and mpath.exists()) else evaluate_policy(args.config, out, name, args.n_prompts)
        rows.append({"condition": name, "eps": eps, "beta_kl": beta, "updates": int(cfg["fork_updates"]),
                     "heldout_reward": m["reward_mean"], "heldout_reward_penalized": m["reward_penalized_mean"],
                     "heldout_kl": m["kl_token_mean"], "heldout_entropy": m["entropy_sampled"],
                     "heldout_entropy_exact": m["entropy_exact"], "heldout_len": m["length_tokens"]["mean"],
                     "heldout_len_std": m["length_tokens"]["std"], "heldout_trunc": m["truncation_rate"],
                     "heldout_distinct2": m["distinct2_mean"], **stability_stats(name, cfg["results_dir"]),
                     "generated_tokens": load_json(res / name / "train_summary.json")["generated_tokens"]})
    save_json(res / "kl_forks.json", rows)
    with open(res / "kl_forks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"  beta={r['beta_kl']:<4g} reward {r['heldout_reward']:+.3f} | KL {r['heldout_kl']:.4f} | "
              f"H {r['heldout_entropy']:.3f} | len {r['heldout_len']:.0f} | cut {r['heldout_trunc']:.0%} | distinct2 {r['heldout_distinct2']:.3f}")


if __name__ == "__main__":
    main()
