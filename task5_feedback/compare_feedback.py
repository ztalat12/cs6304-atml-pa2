from __future__ import annotations

"""Task 5: combine in-domain (GSM8K), controlled-diagnostic and out-of-domain (SVAMP) results into the
final RLVR-vs-RLAIF comparison (CPU only).

Outputs (results/task5_feedback/): task5_summary.json, task5_main_table.csv, task5_diagnostic_table.csv
  main table  : per policy x dataset - accuracy (+95% CI), format compliance, win rate vs SFT,
                length, at-cap rate, failure types; OOD drop = in-domain minus out-of-domain
  diagnostics : per pair type x mechanism - better / tie / wrong rates, S_reason, S_outcome,
                judge order consistency, RLAIF group reward per variant
  feedback-source comparison numbers for the discussion: coverage (share of verifier ties the judge
  separates), noise (judge order inconsistency), exploitability (filler / distractor wrong-preference),
  inference cost (judge seconds per call vs verifier)
If the probe outputs exist (probe.py), the tables also carry, NEXT TO the official numbers:
  lenient_accuracy                 final answer read leniently (format vs maths)
  win_rate_vs_sft_echo_as_tie      the judge's echo outputs ("A, B" = no choice) counted as ties
  judge_echoes_vs_sft              how many verdicts in that pair were echoes
  diagnostic rows 'judge_echo_as_tie', and echo-as-tie versions of the feedback-source numbers
"""

import argparse
import time

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from task5_feedback.rlvr import exact_reward

POLICIES = ["sft", "rlvr", "rlaif"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    base = repo_path(cfg["results_dir"]) / "task5_feedback"
    M = {d: load_json(base / d / "metrics.json") for d in ["gsm", "transfer"] if (base / d / "metrics.json").exists()}
    D = load_json(base / "diagnostics" / "summary.json") if (base / "diagnostics" / "summary.json").exists() else None
    A = load_json(base / "probe" / "judge_audit.json") if (base / "probe" / "judge_audit.json").exists() else None
    L = pd.read_csv(base / "probe" / "lenient_accuracy.csv") if (base / "probe" / "lenient_accuracy.csv").exists() else None
    if not M:
        raise FileNotFoundError("run evaluate_math for --dataset gsm and --dataset transfer first")

    rows = []
    for d, m in M.items():
        for p in POLICIES:
            q = m["policies"][p]
            wr = m["pairs"].get(f"{p}_vs_sft", {}).get("win_rate", 0.5 if p == "sft" else float("nan"))
            extra = {}
            if L is not None:
                hit = L[(L.dataset == d) & (L.policy == p)]
                extra["lenient_accuracy"] = float(hit.lenient_accuracy.iloc[0]) if len(hit) else float("nan")
            if A is not None:
                pp = A["policy_pairs"].get(f"{d}/{p}_vs_sft")
                extra["win_rate_vs_sft_echo_as_tie"] = pp["echo_as_tie"]["win_rate"] if pp else (0.5 if p == "sft" else float("nan"))
                extra["judge_echoes_vs_sft"] = pp["echoes"] if pp else 0
            rows.append({"dataset": d, "policy": p, "n": m["n"], "accuracy": q["accuracy"], "acc_ci_lo": q["accuracy_ci95"][0],
                         "acc_ci_hi": q["accuracy_ci95"][1], "format_compliance": q["format_compliance"], "win_rate_vs_sft": wr,
                         "len_mean": q["len_mean"], "len_std": q["len_std"], "at_cap_rate": q["at_cap_rate"],
                         **{f"ft_{k}": v for k, v in q["failure_types"].items()}, "gold_mentioned_not_final": q["gold_mentioned_not_final"], **extra})
    T = pd.DataFrame(rows)
    if {"gsm", "transfer"} <= set(M):
        for col in [c for c in ["accuracy", "lenient_accuracy", "win_rate_vs_sft", "win_rate_vs_sft_echo_as_tie", "format_compliance", "len_mean"] if c in T.columns]:
            g = T[T.dataset == "gsm"].set_index("policy")[col]; t = T[T.dataset == "transfer"].set_index("policy")[col]
            T.loc[T.dataset == "transfer", f"drop_{col}"] = T[T.dataset == "transfer"].policy.map(g - t).values
    T.to_csv(base / "task5_main_table.csv", index=False)

    out = {"main": T.to_dict("records"), "pairs": {d: m["pairs"] for d, m in M.items()}}
    if D:
        drows = []
        for ptype, v in D["by_pair"].items():
            for mech in ["verifier", "judge"]:
                if mech in v:
                    drows.append({"pair": ptype, "mechanism": mech, **{k: v[mech][k] for k in ["better", "tie", "wrong", "n"]},
                                  "order_consistency": v.get("judge_order_consistency") if mech == "judge" else None})
            if A is not None and ptype in A["diagnostics"]:
                a = A["diagnostics"][ptype]
                drows.append({"pair": ptype, "mechanism": "judge_echo_as_tie", **{k: a["echo_as_tie"][k] for k in ["better", "tie", "wrong", "n"]},
                              "order_consistency": a.get("order_consistency_echo_as_tie")})
        pd.DataFrame(drows).to_csv(base / "task5_diagnostic_table.csv", index=False)
        out["diagnostics"] = D

    # numbers for the feedback-source discussion
    t0 = time.perf_counter()
    for _ in range(2000):
        exact_reward("... so the answer is 42.\n#### 42", "42")
    verifier_sec = (time.perf_counter() - t0) / 2000
    judge_secs = [m.get("judge_cost", {}).get("sec_per_new_call") for m in M.values()] + ([D["judge_cost"]["sec_per_new_call"]] if D else [])
    judge_secs = [s for s in judge_secs if s]
    gsm_pairs = M.get("gsm", {}).get("pairs", {})
    disc = {"verifier_sec_per_call": verifier_sec, "judge_sec_per_call": (sum(judge_secs) / len(judge_secs)) if judge_secs else None,
            "coverage_judge_separates_verifier_ties_gsm": {k: v["verifier_judge"].get("judge_separates_when_verifier_ties") for k, v in gsm_pairs.items()},
            "judge_picks_correct_when_verifier_decisive_gsm": {k: v["verifier_judge"].get("judge_picks_correct") for k, v in gsm_pairs.items()}}
    if D and "judge" in D["by_pair"]["filler"]:
        disc.update({"judge_order_consistency_mean": sum(D["by_pair"][p]["judge_order_consistency"] for p in D["by_pair"]) / len(D["by_pair"]),
                     "exploitability_filler_wrong_pref": {"verifier": D["by_pair"]["filler"]["verifier"]["wrong"], "judge": D["by_pair"]["filler"]["judge"]["wrong"]},
                     "exploitability_distractor_wrong_pref": {"verifier": D["by_pair"]["distractor"]["verifier"]["wrong"], "judge": D["by_pair"]["distractor"]["judge"]["wrong"]}})
    if A is not None:
        dg = A["diagnostics"]
        disc.update({"judge_ambiguous_output_rate": A["outputs"]["echo_rate"], "judge_ambiguous_outputs": A["outputs"]["classes"].get("ECHO", 0),
                     "judge_audit_fidelity": A["outputs"]["fidelity_matches_cache"],
                     "judge_order_consistency_mean_echo_as_tie": sum(v.get("order_consistency_echo_as_tie", float("nan")) for v in dg.values()) / len(dg),
                     "exploitability_filler_wrong_pref_echo_as_tie": dg["filler"]["echo_as_tie"]["wrong"] if "filler" in dg else None,
                     "exploitability_distractor_wrong_pref_echo_as_tie": dg["distractor"]["echo_as_tie"]["wrong"] if "distractor" in dg else None,
                     "coverage_judge_separates_verifier_ties_gsm_echo_as_tie": {k.split("/", 1)[1]: v["verifier_judge_echo_as_tie"].get("judge_separates_when_verifier_ties")
                                                                               for k, v in A["policy_pairs"].items() if k.startswith("gsm/")},
                     "judge_picks_correct_when_verifier_decisive_gsm_echo_as_tie": {k.split("/", 1)[1]: v["verifier_judge_echo_as_tie"].get("judge_picks_correct")
                                                                                   for k, v in A["policy_pairs"].items() if k.startswith("gsm/")}})
        if "group_reward" in A:
            disc["rlaif_group_reward_by_variant"] = {"official": A["group_reward"]["official"], "echo_as_tie": A["group_reward"]["echo_as_tie"]}
        out["judge_audit"] = A
    if disc["judge_sec_per_call"]:
        disc["judge_vs_verifier_cost_ratio"] = disc["judge_sec_per_call"] / disc["verifier_sec_per_call"]
    out["feedback_source"] = disc
    save_json(base / "task5_summary.json", out)
    cols = [c for c in ["dataset", "policy", "accuracy", "lenient_accuracy", "format_compliance", "win_rate_vs_sft", "win_rate_vs_sft_echo_as_tie",
                        "judge_echoes_vs_sft", "len_mean"] if c in T.columns]
    print(T[cols].round(3).to_string(index=False))
    print("feedback-source numbers:", {k: v for k, v in disc.items() if not isinstance(v, dict)})
    print("saved ->", base / "task5_summary.json")


if __name__ == "__main__":
    main()
