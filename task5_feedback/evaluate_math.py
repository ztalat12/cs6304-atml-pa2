from __future__ import annotations

"""Task 5 Steps 1 and 3: SFT vs RLVR vs RLAIF on GSM8K (in-domain) and SVAMP (out-of-domain).

Phase 1 - generation: every policy answers every problem with greedy decoding (deterministic), the
          same max_new_tokens (`math_max_new_tokens`), prompts in the fixed file order, one batch size.
Phase 2 - exact verifier (supplied rlvr.exact_reward): accuracy, format compliance, failure types.
Phase 3 - fixed pairwise AI judge (supplied rlaif.PairwiseAIJudge, cached, position-balanced):
          rlvr vs sft, rlaif vs sft, rlvr vs rlaif on every problem.
          win = 1, tie = 0.5, loss = 0 (manual). Word-for-word identical answers are scored as a tie
          WITHOUT calling the judge (same text cannot be better than itself); their number is reported.
Outputs: results/task5_feedback/<dataset>/generations_<policy>.jsonl, pairs.jsonl, metrics.json
"""

import argparse
import math
import re
import time

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import load_json, save_json
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

POLICIES = ["sft", "rlvr", "rlaif"]
PAIRS = [("rlvr", "sft"), ("rlaif", "sft"), ("rlvr", "rlaif")]
FORMAT_INSTRUCTION = "\n\nShow your reasoning and end your response with exactly `#### <number>`."
_GOLD_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


# ------------------------------------------------------------------ row accessors (schema-tolerant)
def problem_text(row) -> str:
    """The plain problem statement given to the AI judge."""
    for k in ["question", "problem", "prompt_text"]:
        if isinstance(row.get(k), str) and row[k].strip():
            return row[k].strip()
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        return str(msgs[-1].get("content", "")).replace(FORMAT_INSTRUCTION, "").strip()
    return str(row.get("prompt", "")).strip()


def gold_final(row) -> str:
    for k in ["gold_final", "gold", "final_answer"]:
        if row.get(k) is not None and str(row[k]).strip() != "":
            return str(row[k]).replace(",", "").strip()
    m = _GOLD_RE.findall(str(row.get("answer", "")))
    if m:
        return m[-1].replace(",", "")
    raise KeyError(f"no gold answer field in row with keys {sorted(row)}")


def policy_messages(row):
    """Chat messages for the policy. Uses the file's own `messages` (which carry the `#### <number>`
    instruction); if a row only has a question, the course's standard instruction is appended."""
    if isinstance(row.get("messages"), list) and row["messages"]:
        return row["messages"]
    return [{"role": "user", "content": problem_text(row) + FORMAT_INSTRUCTION}]


def row_id(row, i):
    return str(row.get("prompt_id", row.get("source_index", row.get("id", i))))


# ------------------------------------------------------------------ small pure functions (unit-tested)
def pref_score(result: str) -> float:
    """Judge verdict for compare(problem, a=X, b=Y) -> score for X: A = 1, TIE = 0.5, B = 0."""
    return {"A": 1.0, "TIE": 0.5, "B": 0.0}[result]


def failure_type(response: str, gold: str, truncated: bool) -> str:
    pred = extract_designated_final(response)
    if pred is not None:
        return "correct" if exact_reward(response, gold) == 1.0 else "wrong_final"
    if truncated:
        return "no_final_truncated"
    return "no_final_completed"


def gold_mentioned_not_final(response: str, gold: str) -> bool:
    """The gold number appears somewhere in the text, but the designated final is missing or different."""
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", str(response))
    has = any(_num_eq(n.replace(",", ""), gold) for n in nums)
    return has and exact_reward(response, gold) == 0.0


def _num_eq(a, b):
    try:
        return abs(float(a) - float(b)) <= 1e-9
    except Exception:
        return False


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def agreement_stats(v_scores, j_scores):
    """v_scores: verifier score of X minus Y in {-1, 0, 1}; j_scores: judge score for X in {0, .5, 1}.
    Returns overall agreement and the two informative breakdowns."""
    v = np.asarray(v_scores, float); j = np.sign(np.asarray(j_scores, float) - 0.5)
    out = {"n": int(len(v)), "agreement": float((np.sign(v) == j).mean()) if len(v) else float("nan")}
    dec = v != 0
    out["verifier_decisive_n"] = int(dec.sum())
    if dec.any():
        out["judge_picks_correct"] = float((j[dec] == np.sign(v[dec])).mean())
        out["judge_ties_when_decisive"] = float((j[dec] == 0).mean())
        out["judge_picks_wrong"] = float((j[dec] == -np.sign(v[dec])).mean())
    tie = v == 0
    out["verifier_tie_n"] = int(tie.sum())
    if tie.any():
        out["judge_separates_when_verifier_ties"] = float((j[tie] != 0).mean())
    return out


# ------------------------------------------------------------------ phases
def generate_all(cfg, rows, out_dir, batch_size, skip_existing):
    tok = load_tokenizer(cfg["base_model"])
    tok.truncation_side = "left"
    cap = int(cfg["math_max_new_tokens"])
    for name in POLICIES:
        dst = out_dir / f"generations_{name}.jsonl"
        if skip_existing and dst.exists() and len(read_jsonl(dst)) == len(rows):
            print(f"[{name}] generations exist -> skipping"); continue
        model = load_frozen_policy(cfg, name)
        recs, t0 = [], time.perf_counter()
        nb = (len(rows) + batch_size - 1) // batch_size
        for b, s in enumerate(range(0, len(rows), batch_size)):
            chunk = rows[s:s + batch_size]
            g = batch_generate(model, tok, [policy_messages(r) for r in chunk], max_prompt_length=512,
                               max_new_tokens=cap, temperature=0.0, top_p=1.0, do_sample=False)
            for i, r in enumerate(chunk):
                resp, gold = g["responses"][i], gold_final(r)
                recs.append({"id": row_id(r, s + i), "policy": name, "problem": problem_text(r), "gold": gold, "response": resp,
                             "response_tokens": int(g["response_lengths"][i]), "truncated": bool(g["truncated"][i]),
                             "pred": extract_designated_final(resp), "correct": exact_reward(resp, gold),
                             "format_ok": extract_designated_final(resp) is not None,
                             "failure_type": failure_type(resp, gold, bool(g["truncated"][i])),
                             "gold_mentioned_not_final": gold_mentioned_not_final(resp, gold)})
            if (b + 1) % 5 == 0 or b + 1 == nb:
                el = time.perf_counter() - t0
                print(f"  [{name}] batch {b+1}/{nb} | {el/60:.1f} min, ~{el/(b+1)*(nb-b-1)/60:.1f} min left", flush=True)
        write_jsonl(dst, recs)
        print(f"[{name}] accuracy {np.mean([r['correct'] for r in recs]):.3f} | format {np.mean([r['format_ok'] for r in recs]):.3f} | "
              f"length {np.mean([r['response_tokens'] for r in recs]):.0f} | at cap {np.mean([r['truncated'] for r in recs]):.2f}")
        del model
        clear_gpu()


def judge_pairs(cfg, out_dir, cache_path):
    G = {p: read_jsonl(out_dir / f"generations_{p}.jsonl") for p in POLICIES}
    judge = PairwiseAIJudge(cfg, cache_path)
    recs, new_calls, t_new = [], 0, 0.0
    for x, y in PAIRS:
        for gx, gy in zip(G[x], G[y]):
            assert gx["id"] == gy["id"]
            if gx["response"] == gy["response"]:
                res, identical = "TIE", True
            else:
                key = judge._key(gx["problem"], gx["response"], gy["response"])
                cached = key in judge.cache
                t0 = time.perf_counter()
                res = judge.compare(gx["problem"], gx["response"], gy["response"])
                if not cached:
                    new_calls += 1; t_new += time.perf_counter() - t0
                identical = False
            recs.append({"id": gx["id"], "x": x, "y": y, "judge": res, "x_score": pref_score(res), "identical": identical,
                         "x_correct": gx["correct"], "y_correct": gy["correct"], "verifier_diff": gx["correct"] - gy["correct"],
                         "x_tokens": gx["response_tokens"], "y_tokens": gy["response_tokens"]})
        done = [r for r in recs if r["x"] == x and r["y"] == y]
        print(f"  judged {x} vs {y}: win rate {np.mean([r['x_score'] for r in done]):.3f} | ties {sum(r['judge'] == 'TIE' for r in done)} "
              f"(identical {sum(r['identical'] for r in done)})", flush=True)
    write_jsonl(out_dir / "pairs.jsonl", recs)
    del judge
    clear_gpu()
    return {"new_judge_calls": new_calls, "sec_per_new_call": t_new / new_calls if new_calls else None}


def summarize(cfg, out_dir, dataset, cost):
    G = {p: read_jsonl(out_dir / f"generations_{p}.jsonl") for p in POLICIES}
    P = read_jsonl(out_dir / "pairs.jsonl")
    m = {"dataset": dataset, "n": len(G["sft"]), "max_new_tokens": int(cfg["math_max_new_tokens"]), "decoding": "greedy", "policies": {}, "pairs": {},
         "judge_cost": cost}
    for p in POLICIES:
        g = G[p]; n = len(g); k = int(sum(r["correct"] for r in g))
        L = np.array([r["response_tokens"] for r in g], float)
        ft = {t: sum(r["failure_type"] == t for r in g) / n for t in ["correct", "wrong_final", "no_final_truncated", "no_final_completed"]}
        m["policies"][p] = {"accuracy": k / n, "accuracy_ci95": list(wilson(k, n)), "format_compliance": float(np.mean([r["format_ok"] for r in g])),
                            "len_mean": float(L.mean()), "len_std": float(L.std(ddof=1)), "len_median": float(np.median(L)),
                            "len_iqr": float(np.percentile(L, 75) - np.percentile(L, 25)), "at_cap_rate": float(np.mean([r["truncated"] for r in g])),
                            "failure_types": ft, "gold_mentioned_not_final": float(np.mean([r["gold_mentioned_not_final"] for r in g]))}
    for x, y in PAIRS:
        d = [r for r in P if r["x"] == x and r["y"] == y]
        s = np.array([r["x_score"] for r in d])
        gx, gy = {r["id"]: r["correct"] for r in G[x]}, {r["id"]: r["correct"] for r in G[y]}
        b = sum(1 for i in gx if gx[i] == 1 and gy[i] == 0); c = sum(1 for i in gx if gx[i] == 0 and gy[i] == 1)
        m["pairs"][f"{x}_vs_{y}"] = {
            "win_rate": float(s.mean()), "win_rate_se": float(s.std(ddof=1) / math.sqrt(len(s))) if len(s) > 1 else float("nan"),
            "wins": int((s == 1).sum()), "ties": int((s == 0.5).sum()), "losses": int((s == 0).sum()),
            "identical_answers": int(sum(r["identical"] for r in d)), "judge_ties_excl_identical": int(sum(r["judge"] == "TIE" and not r["identical"] for r in d)),
            "verifier_judge": agreement_stats([r["verifier_diff"] for r in d], [r["x_score"] for r in d]),
            "accuracy_only_x": b, "accuracy_only_y": c, "mcnemar_p": mcnemar_exact(b, c)}
    save_json(out_dir / "metrics.json", m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, help="first N problems only (smoke tests)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()
    cfg, rows, _ = load_math_evaluation(args.config, args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))
    out_dir = repo_path(cfg["results_dir"]) / "task5_feedback" / (args.dataset if not args.limit else f"{args.dataset}_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_all(cfg, rows, out_dir, args.batch_size, args.skip_existing)
    if args.no_judge:
        return
    cache = repo_path(cfg["results_dir"]) / "task5_feedback" / "judge_cache.json"
    cost = judge_pairs(cfg, out_dir, cache)
    mt = summarize(cfg, out_dir, args.dataset, cost)
    for p in POLICIES:
        q = mt["policies"][p]
        print(f"  {p:6s} accuracy {q['accuracy']:.3f} | format {q['format_compliance']:.3f} | length {q['len_mean']:.0f} +/- {q['len_std']:.0f}")
    for k_, q in mt["pairs"].items():
        print(f"  {k_:15s} win rate {q['win_rate']:.3f} (W/T/L {q['wins']}/{q['ties']}/{q['losses']}, identical {q['identical_answers']}) | "
              f"verifier-judge agreement {q['verifier_judge']['agreement']:.3f}")
    print("saved ->", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
