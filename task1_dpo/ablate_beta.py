from __future__ import annotations
 
"""Task 1 Step 2: beta (regularisation-strength) study.
 
Controlled design: for each beta in {0.03, 0.10, 0.30}
  * start from the ORIGINAL Qwen init with a fresh LoRA (train.prepare_dpo_run),
  * use the SAME first `short_ablation_examples` (=600) training pairs,
  * the SAME seed (=> identical LoRA init and identical shuffle order),
  * the SAME optimiser, learning rate, batch size and LoRA config.
Only beta changes, so differences are attributable to beta.
 
What to expect (hypothesis, not a result): beta multiplies the margin inside the
sigmoid. Large beta -> the sigmoid saturates after only a small margin -> the loss is
"satisfied" early -> less movement from pi_ref (lower KL). Small beta -> the model
must build a big margin -> more drift. RQ1 asks whether your data agree and whether
the trend is monotonic across the three values.
 
CAUTION for the report: the held-out DPO LOSS depends on beta itself (beta is inside
the formula), so losses are NOT directly comparable across beta. Compare preference
accuracy, mean margin, KL, reward and length instead. Also note these are short
600-pair runs, a different budget from the one-epoch standard run.
"""
 
import argparse
import csv
 
from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from task1_dpo.evaluate import evaluate_adapter
from task1_dpo.train import run_training
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--gen-prompts", type=int, default=200)
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse adapters that already exist (handy after a Colab disconnect)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    betas = [float(b) for b in cfg["betas"]]
    n = int(cfg["short_ablation_examples"])
    print("Required beta values:", betas, "| short-run examples per condition:", n)
 
    rows = []
    for beta in betas:
        name = f"beta_{beta:g}"
        out = f"outputs/task1_dpo/{name}"
        if not (args.skip_existing and (repo_path(out) / "adapter_config.json").exists()):
            run_training(args.config, run_name=name, output_path=out, beta=beta, max_examples=n)
        mpath = repo_path(cfg["results_dir"]) / name / "metrics.json"
        if args.skip_existing and mpath.exists():
            metrics = load_json(mpath)          # already evaluated in an earlier session
        else:
            metrics, _ = evaluate_adapter(args.config, out, name, beta=beta, gen_prompts=args.gen_prompts)
        g, p = metrics["generation"], metrics["preference"]
        rows.append({
            "condition": name, "beta": beta, "train_pairs": n, "budget": "short",
            "pref_acc": p["preference_accuracy"], "margin_mean": p["margin_mean"],
            "dpo_loss_at_own_beta": p["dpo_loss"],
            "kl_token_mean": g["kl_token_mean"], "reward_mean": g["reward_mean"],
            "len_mean": g["length_tokens"]["mean"], "len_std": g["length_tokens"]["std"],
            "len_iqr": g["length_tokens"]["iqr"],
            "word_limit_compliance": metrics["word_limit"]["word_limit_compliance"],
        })
 
    res = repo_path(cfg["results_dir"])
    save_json(res / "beta_summary.json", rows)
    with open(res / "beta_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
 
    # Simple monotonicity check for RQ1: does each metric move in one direction as beta grows?
    for key in ["pref_acc", "kl_token_mean", "reward_mean", "len_mean"]:
        vals = [r[key] for r in rows]
        diffs = [b - a for a, b in zip(vals, vals[1:])]
        trend = "increasing" if all(d > 0 for d in diffs) else "decreasing" if all(d < 0 for d in diffs) else "NON-monotonic"
        print(f"{key:>15}: {['%.4f' % v for v in vals]} -> {trend} in beta")
 
 
if __name__ == "__main__":
    main()
 
