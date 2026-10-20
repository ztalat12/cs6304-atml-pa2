from __future__ import annotations

"""Task 5 Step 2: reward-sensitivity and robustness on the supplied Controlled Reward Diagnostic Set.

For each of the 20 problems, each perturbed variant is paired with `clean_correct`, which is the
diagnostically BETTER response in every pair:
  reasoning  : clean_correct vs corrupt_reasoning_correct_final   (final answer held correct)  -> S_reason
  outcome    : clean_correct vs good_reasoning_wrong_final         (reasoning held ~fixed)      -> S_outcome
  filler     : clean_correct vs persuasive_filler_correct          (style / verbosity probe)
  distractor : clean_correct vs gold_distractor_wrong_final        (verifier-specification probe)
Each mechanism's verdict on a pair: BETTER (prefers clean), TIE, or WRONG (prefers the perturbed one).
  verifier : exact_reward(clean) vs exact_reward(perturbed)
  judge    : PairwiseAIJudge.compare(problem, clean, perturbed)   (supplied protocol, position-balanced, cached)
Extras: (i) order consistency - every pair also judged as compare(problem, perturbed, clean);
        (ii) RLAIF group reward over all 5 variants (the training-time formula, K = 5);
        (iii) per-variant verifier reward (the "all five categories" view).
Outputs: results/task5_feedback/diagnostics/{pairs.jsonl, group_rewards.jsonl, summary.json}
"""

import argparse
import time
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlvr import exact_reward, extract_designated_final
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}
PAIR_TYPES = {"reasoning": "corrupt_reasoning_correct_final", "outcome": "good_reasoning_wrong_final",
              "filler": "persuasive_filler_correct", "distractor": "gold_distractor_wrong_final"}
VARIANT_ORDER = ["clean_correct", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final",
                 "persuasive_filler_correct", "gold_distractor_wrong_final"]
EXPECTED_VERIFIER = {"clean_correct": 1.0, "corrupt_reasoning_correct_final": 1.0, "good_reasoning_wrong_final": 0.0,
                     "persuasive_filler_correct": 1.0, "gold_distractor_wrong_final": 0.0}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def field(row, names, what):
    for k in names:
        if isinstance(row.get(k), (str, int, float)) and str(row[k]).strip() != "":
            return str(row[k])
    raise KeyError(f"no {what} field among {names}; row keys: {sorted(row)}")


def response_of(row):
    return field(row, ["response", "completion", "text", "solution", "candidate"], "response")


def problem_of(row):
    """Plain problem text for the judge -- the same convention as evaluate_math.problem_text."""
    from task5_feedback.evaluate_math import problem_text
    t = problem_text(row)
    if not t:
        raise KeyError(f"no problem text; row keys: {sorted(row)}")
    return t


def gold_of(row):
    return field(row, ["gold_final", "gold", "gold_answer", "answer_final"], "gold").replace(",", "")


def verdict_from_scores(better: float, worse: float) -> str:
    return "BETTER" if better > worse else "TIE" if better == worse else "WRONG"


def verdict_from_judge(result_clean_first: str) -> str:
    """result of compare(problem, a=clean, b=perturbed)."""
    return {"A": "BETTER", "TIE": "TIE", "B": "WRONG"}[result_clean_first]


def rates(verdicts):
    n = len(verdicts)
    return {"n": n, "better": sum(v == "BETTER" for v in verdicts) / n if n else float("nan"),
            "tie": sum(v == "TIE" for v in verdicts) / n if n else float("nan"),
            "wrong": sum(v == "WRONG" for v in verdicts) / n if n else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--no-judge", action="store_true", help="verifier only (no GPU)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    out = repo_path(cfg["results_dir"]) / "task5_feedback" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)

    # verifier on every variant
    vrows = []
    for pid, V in sorted(groups.items()):
        gold = gold_of(V["clean_correct"])
        for vt in VARIANT_ORDER:
            resp = response_of(V[vt])
            r = exact_reward(resp, gold)
            vrows.append({"problem_id": pid, "variant": vt, "verifier": r, "expected": EXPECTED_VERIFIER[vt],
                          "pred": extract_designated_final(resp), "gold": gold, "tokens_approx": len(resp.split())})
    judge = None if args.no_judge else PairwiseAIJudge(cfg, repo_path(cfg["results_dir"]) / "task5_feedback" / "judge_cache.json")
    pairs, groups_out, new_calls, t_new = [], [], 0, 0.0

    def cmp(problem, a, b):
        nonlocal new_calls, t_new
        cached = judge._key(problem, a, b) in judge.cache
        t0 = time.perf_counter(); res = judge.compare(problem, a, b)
        if not cached:
            new_calls += 1; t_new += time.perf_counter() - t0
        return res

    for pid, V in sorted(groups.items()):
        gold, problem = gold_of(V["clean_correct"]), problem_of(V["clean_correct"])
        clean = response_of(V["clean_correct"])
        for ptype, vt in PAIR_TYPES.items():
            pert = response_of(V[vt])
            rec = {"problem_id": pid, "pair": ptype, "variant": vt,
                   "verifier": verdict_from_scores(exact_reward(clean, gold), exact_reward(pert, gold)),
                   "len_clean_words": len(clean.split()), "len_perturbed_words": len(pert.split())}
            if judge is not None:
                j1 = cmp(problem, clean, pert)            # supplied protocol: headline verdict
                j2 = cmp(problem, pert, clean)            # reversed argument order: consistency check
                rec.update({"judge": verdict_from_judge(j1), "judge_raw": j1, "judge_reversed_raw": j2,
                            "judge_order_consistent": verdict_from_judge(j1) == {"A": "WRONG", "TIE": "TIE", "B": "BETTER"}[j2]})
            pairs.append(rec)
        if judge is not None:                               # RLAIF training-time group reward, K = 5
            resp5 = [response_of(V[vt]) for vt in VARIANT_ORDER]
            rw = judge.group_rewards(problem, resp5)
            groups_out += [{"problem_id": pid, "variant": vt, "rlaif_group_reward": float(r)} for vt, r in zip(VARIANT_ORDER, rw)]
        print(f"  problem {pid}: " + ", ".join(f"{p['pair']} V:{p['verifier'][0]}" + (f" J:{p['judge'][0]}" if 'judge' in p else "")
                                               for p in pairs if p["problem_id"] == pid), flush=True)

    write_jsonl(out / "verifier_by_variant.jsonl", vrows)
    write_jsonl(out / "pairs.jsonl", pairs)
    if groups_out:
        write_jsonl(out / "group_rewards.jsonl", groups_out)
    S = {"n_problems": len(groups), "verifier_matches_expected": float(np.mean([r["verifier"] == r["expected"] for r in vrows])),
         "verifier_by_variant": {vt: float(np.mean([r["verifier"] for r in vrows if r["variant"] == vt])) for vt in VARIANT_ORDER},
         "by_pair": {}, "judge_cost": {"new_calls": new_calls, "sec_per_new_call": (t_new / new_calls) if new_calls else None}}
    for ptype in PAIR_TYPES:
        d = [p for p in pairs if p["pair"] == ptype]
        S["by_pair"][ptype] = {"verifier": rates([p["verifier"] for p in d])}
        if judge is not None:
            S["by_pair"][ptype]["judge"] = rates([p["judge"] for p in d])
            S["by_pair"][ptype]["judge_order_consistency"] = float(np.mean([p["judge_order_consistent"] for p in d]))
    S["S_reason"] = {m: S["by_pair"]["reasoning"][m]["better"] for m in S["by_pair"]["reasoning"] if m in ("verifier", "judge")}
    S["S_outcome"] = {m: S["by_pair"]["outcome"][m]["better"] for m in S["by_pair"]["outcome"] if m in ("verifier", "judge")}
    if groups_out:
        S["rlaif_group_reward_by_variant"] = {vt: float(np.mean([g["rlaif_group_reward"] for g in groups_out if g["variant"] == vt])) for vt in VARIANT_ORDER}
    save_json(out / "summary.json", S)
    print("verifier matches the expected pattern on", f"{S['verifier_matches_expected']:.0%}", "of variants")
    for ptype, v in S["by_pair"].items():
        line = f"  {ptype:10s} verifier B/T/W {v['verifier']['better']:.2f}/{v['verifier']['tie']:.2f}/{v['verifier']['wrong']:.2f}"
        if "judge" in v:
            line += f" | judge B/T/W {v['judge']['better']:.2f}/{v['judge']['tie']:.2f}/{v['judge']['wrong']:.2f} (order-consistent {v['judge_order_consistency']:.2f})"
        print(line)
    print("S_reason:", S["S_reason"], "| S_outcome:", S["S_outcome"])


if __name__ == "__main__":
    main()
