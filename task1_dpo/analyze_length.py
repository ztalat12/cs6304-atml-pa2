from __future__ import annotations

"""Task 1 Step 3: length-confounding study.

Question: is DPO learning "content quality", or partly "longer = preferred"?

Three pieces of evidence, kept SEPARATE as the manual asks:
 1. DATASET property  - in the training pairs, how often is the chosen response
    longer than the rejected one? (measured on the data, no model involved)
 2. LEARNED preference - per length stratum of the held-out stratified set
    (preferred-longer / matched / rejected-longer), how often does each model rank
    the chosen response higher (m_theta > 0)?
    Telltale sign of length bias: high accuracy on "preferred-longer" but low
    accuracy on "rejected-longer" (there, "pick the longer one" picks the WRONG one).
 3. POLICY behaviour  - generated length and word-limit compliance on the COMMON
    prompt set (same held-out prompts + same 10 word-limit prompts as the standard
    evaluation), so standard and length-balanced are directly comparable.
"""

import argparse
import csv
from collections import defaultdict

import numpy as np

from common.data import load_yaml, preference_responses, read_jsonl, repo_path, write_jsonl
from common.logging_utils import load_json, save_json
from common.models import clear_gpu, load_policy, load_tokenizer
from task1_dpo.evaluate import evaluate_adapter, preference_metrics
from task1_dpo.train import filter_fitting, run_training

# The stratified file labels each pair with its stratum. The exact key name is set by
# the course release; we try common names and fail loudly (printing the keys) if none match.
STRATUM_KEYS = ["stratum", "length_stratum", "length_bucket", "bucket", "strata", "length_group"]


def find_stratum_key(row):
    for k in STRATUM_KEYS:
        if k in row:
            return k
    raise KeyError(f"No stratum field found. Row keys are: {sorted(row)} -> add the right one to STRATUM_KEYS")


def dataset_length_stats(tokenizer, rows):
    """Evidence piece 1: length structure of a preference DATASET (no model involved)."""
    diffs = []
    for r in rows:
        yc, yr = preference_responses(r)
        lc = len(tokenizer(yc, add_special_tokens=False)["input_ids"])
        lr = len(tokenizer(yr, add_special_tokens=False)["input_ids"])
        diffs.append(lc - lr)
    d = np.asarray(diffs)
    return {"n": int(len(d)), "frac_chosen_longer": float((d > 0).mean()),
            "frac_rejected_longer": float((d < 0).mean()),
            "mean_len_diff_chosen_minus_rejected": float(d.mean()),
            "median_len_diff": float(np.median(d))}


def stratified_preference(cfg, adapter, beta, rows, batch_size=4):
    """Evidence piece 2 (teacher forcing only, no generation): margins on the stratified pairs."""
    tok = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    kept, kept_idx, _ = filter_fitting(rows, tok, int(cfg["max_sequence_length"]))   # same rule as training
    pref, recs = preference_metrics(model, tok, kept, beta, int(cfg["max_sequence_length"]), batch_size, indices=kept_idx)
    del model
    clear_gpu()
    return pref, recs


def per_stratum(pair_records, eval_rows, key):
    """Accuracy and margin broken down by stratum."""
    groups = defaultdict(list)
    for rec in pair_records:
        groups[eval_rows[rec["index"]][key]].append(rec)
    out = {}
    for stratum, recs in sorted(groups.items()):
        m = np.asarray([r["margin"] for r in recs])
        out[str(stratum)] = {"n": len(recs), "pref_acc": float((m > 0).mean()),
                             "margin_mean": float(m.mean()),
                             "chosen_len_mean": float(np.mean([r["chosen_len"] for r in recs])),
                             "rejected_len_mean": float(np.mean([r["rejected_len"] for r in recs]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--gen-prompts", type=int, default=200, help="must match the standard evaluation")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    beta = float(cfg["beta"])
    res = repo_path(cfg["results_dir"])
    study_dir = res / "length_study"
    study_dir.mkdir(parents=True, exist_ok=True)

    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified))
    key = find_stratum_key(stratified[0])

    # ---- 1. dataset-level length structure ---------------------------------------
    tok = load_tokenizer(cfg["base_model"])
    data_stats = {
        "standard_train": dataset_length_stats(tok, read_jsonl(cfg["paths"]["dpo_standard_train"])),
        "balanced_train": dataset_length_stats(tok, balanced),
        "stratified_eval": dataset_length_stats(tok, stratified),
    }

    # ---- train the length-balanced model (same init, seed, optimiser, beta, 1 epoch)
    std_out, bal_out = cfg["standard_output"], cfg["length_output"]
    if not (repo_path(std_out) / "adapter_config.json").exists():
        raise FileNotFoundError(f"{std_out} missing: run the standard DPO training first")
    if not (args.skip_existing and (repo_path(bal_out) / "adapter_config.json").exists()):
        run_training(args.config, run_name="length_balanced",
                     dataset_path=cfg["paths"]["dpo_length_train"], output_path=bal_out)

    # ---- 3. policy behaviour on the COMMON prompt set ----------------------------
    # Standard model: reuse its main evaluation if it exists (same protocol), else run it.
    gen_metrics = {}
    for name, adapter in [("standard", std_out), ("length_balanced", bal_out)]:
        mpath = res / name / "metrics.json"
        if args.skip_existing and mpath.exists():
            m = load_json(mpath)
        else:
            m, _ = evaluate_adapter(args.config, adapter, name, gen_prompts=args.gen_prompts)
        if m["gen_prompts"] != args.gen_prompts:
            print(f"WARNING: {name} was evaluated on {m['gen_prompts']} prompts, not {args.gen_prompts}")
        gen_metrics[name] = m

    # ---- 2. per-stratum preference accuracy (teacher forcing) ---------------------
    summary = {"stratum_key": key, "dataset_length_stats": data_stats, "models": {}}
    rows_csv = []
    for name, adapter in [("standard", std_out), ("length_balanced", bal_out)]:
        pref, recs = stratified_preference(cfg, adapter, beta, stratified)
        for r in recs:
            r["stratum"] = stratified[r["index"]][key]
        write_jsonl(study_dir / f"{name}_stratified_pairs.jsonl", recs)
        strata = per_stratum(recs, stratified, key)
        g = gen_metrics[name]
        summary["models"][name] = {
            "per_stratum": strata,
            "stratified_overall_pref_acc": pref["preference_accuracy"],
            "gen_length_tokens": g["generation"]["length_tokens"],
            "word_limit": g["word_limit"],
        }
        for s, v in strata.items():
            rows_csv.append({"model": name, "stratum": s, **v,
                             "gen_len_mean": g["generation"]["length_tokens"]["mean"],
                             "word_limit_compliance": g["word_limit"]["word_limit_compliance"]})
        print(f"[{name}] " + "  ".join(f"{s}: acc={v['pref_acc']:.3f}" for s, v in strata.items()))

    save_json(res / "length_study_summary.json", summary)
    with open(res / "length_study_per_stratum.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader(); w.writerows(rows_csv)
    print("Dataset: fraction chosen-longer (standard train) =",
          round(data_stats["standard_train"]["frac_chosen_longer"], 3))


if __name__ == "__main__":
    main()
