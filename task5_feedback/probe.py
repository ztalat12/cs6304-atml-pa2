from __future__ import annotations

"""Task 5 judge-output audit + follow-up probes.

The supplied verifier and judge are NOT modified, and the numbers they produce stay the OFFICIAL ones. Everything
here is reported NEXT TO them (the manual: "report the number of judge ties or ambiguous outputs separately").

(a) judge-output audit. The supplied PairwiseAIJudge.compare() asks for "A", "B" or "TIE", lets the judge write at
    most 4 tokens, reads the text with re.search(r"\\b(A|B|TIE)\\b") (the FIRST match wins; no match -> TIE), undoes
    its hash-based A/B swap, and caches only the verdict - the raw text is thrown away. This re-runs that exact
    procedure with the raw text kept, for EVERY comparison the official numbers are built from:
      diagnostics : 80 headline pairs (clean vs perturbed), their 80 reversed-order checks, and the 200 comparisons
                    inside the K = 5 RLAIF group reward (10 per problem; the 80 clean-first ones ARE the headline
                    pairs, so 120 are new)
      GSM8K/SVAMP : every non-identical policy pair (identical answers never reach the judge)
    Each raw output gets a class:
      TIE / A / B : a real answer (names exactly one option)
      ECHO        : names more than one option - the judge repeated the option list ("A, B", "A,B,TIE") and was cut
                    off by the 4-token cap. The regex reads "A, B" as "A" = "the answer SHOWN FIRST wins": a coin
                    flip set by the swap, recorded as if it were a preference.
      OTHER       : names no option (the regex falls back to TIE)
    echo-as-tie numbers: an echo is no verdict, so it is scored 0.5 like a tie; every other verdict is unchanged.
    Fidelity check: every re-run verdict must equal the cached official verdict (the judge is deterministic).
    Resumable: comparisons already in probe/judge_raw.jsonl (matched by the judge's own cache key) are not re-run.
(b) adapter magnitude - how far each LoRA adapter moves the base weights: for every adapted matrix,
    dW = (lora_alpha / r) * B @ A and the relative size ||dW||_F / ||W||_F. Run on RLVR, RLAIF, the PPO/GRPO
    midpoints, and the Task 1-3 standard adapters if present under outputs/ (one scale for the synthesis).
(c) lenient final-answer reader - how much of the exact-accuracy gap is FORMAT: accepts '#### $N',
    '#### Answer: N', '\\boxed{N}', 'the answer is N'. Reported next to, never instead of, the verifier.
Outputs: results/task5_feedback/probe/{judge_raw.jsonl, judge_audit.json, adapter_norms.csv, lenient_accuracy.csv}
Run:     python -m task5_feedback.probe --config configs/feedback.yaml [--only judge|lenient|adapters, comma-separated]
"""

import argparse
import collections
import itertools
import json
import math
import re

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json

NUM = r"[-+]?\$?\d[\d,]*(?:\.\d+)?"
FLIP = {"A": "B", "B": "A", "TIE": "TIE"}


# ------------------------------------------------------------------ (c) lenient reader (pure, unit-tested)
def lenient_final(text: str):
    t = str(text)
    m = re.findall(r"####\s*(?:\**\s*answer\s*:?\s*\**)?\s*(" + NUM + ")", t, re.I)
    if m:
        return m[-1].replace(",", "").replace("$", ""), "hash_lenient"
    m = re.findall(r"\\boxed\{\s*(" + NUM + ")", t)
    if m:
        return m[-1].replace(",", "").replace("$", ""), "boxed"
    m = re.findall(r"answer is[:\s]*\**\s*(" + NUM + ")", t, re.I)
    if m:
        return m[-1].replace(",", "").replace("$", ""), "answer_is"
    return None, "none"


def _eq(a, b):
    try:
        return abs(float(a) - float(b)) <= 1e-9
    except Exception:
        return False


def lenient_table(base):
    rows = []
    for ds in ["gsm", "transfer"]:
        for p in ["sft", "rlvr", "rlaif"]:
            f = base / ds / f"generations_{p}.jsonl"
            if not f.exists():
                continue
            G = read_jsonl(f); n = len(G)
            how = collections.Counter()
            ok = 0
            for r in G:
                if r["correct"] == 1:
                    ok += 1; continue
                v, h = lenient_final(r["response"])
                if v is not None and _eq(v, r["gold"]):
                    ok += 1; how[h] += 1
            rows.append({"dataset": ds, "policy": p, "n": n, "exact_accuracy": sum(r["correct"] for r in G) / n, "lenient_accuracy": ok / n,
                         "rescued_hash_lenient": how["hash_lenient"], "rescued_boxed": how["boxed"], "rescued_answer_is": how["answer_is"],
                         "hash_present_but_unparsed": sum(1 for r in G if "####" in r["response"] and r["pred"] is None)})
    return rows


# ------------------------------------------------------------------ (a) judge-output audit: pure helpers (unit-tested)
def raw_class(raw: str) -> str:
    """'TIE' / 'A' / 'B' when the output names exactly one option, 'ECHO' when it names more than one, 'OTHER' when none.
    Uses the same token rule as the supplied parser (\\b(A|B|TIE)\\b on the upper-cased text)."""
    names = set(re.findall(r"\b(A|B|TIE)\b", str(raw).strip().upper()))
    if len(names) == 1:
        return names.pop()
    return "ECHO" if len(names) > 1 else "OTHER"


def corrected(row) -> str:
    """Verdict in the caller's (a, b) orientation with an echo counted as no verdict (TIE); all else unchanged."""
    return "TIE" if row["cls"] == "ECHO" else row["result"]


def shown_first(row) -> bool:
    """Did the parser's verdict go to the answer DISPLAYED first? (result is in caller order; undo the swap)."""
    disp = FLIP[row["result"]] if row["swap"] else row["result"]
    return disp == "A"


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def group_reward(g: dict, K: int, verdict) -> np.ndarray:
    """The supplied RLAIF group reward ((wins + 0.5 ties) / (K - 1)) from {(i, j): row} with i < j, a = response i."""
    wins, ties = np.zeros(K), np.zeros(K)
    for (i, j), r in g.items():
        v = verdict(r)
        if v == "A":
            wins[i] += 1
        elif v == "B":
            wins[j] += 1
        else:
            ties[i] += 1; ties[j] += 1
    return (wins + 0.5 * ties) / max(K - 1, 1)


def _winrate(s):
    s = np.asarray(s, float)
    return {"win_rate": float(s.mean()), "win_rate_se": float(s.std(ddof=1) / math.sqrt(len(s))) if len(s) > 1 else float("nan"),
            "wins": int((s == 1).sum()), "ties": int((s == 0.5).sum()), "losses": int((s == 0).sum())}


def _rates(verdicts):
    """verdicts in clean-first orientation: A = prefers clean = BETTER, B = WRONG."""
    n = len(verdicts); c = collections.Counter(verdicts)
    return {"n": n, "better": c["A"] / n, "tie": c["TIE"] / n, "wrong": c["B"] / n} if n else {"n": 0}


# ------------------------------------------------------------------ (a) judge-output audit: GPU part
def raw_compare(judge, problem, a, b):
    """Exactly PairwiseAIJudge.compare's steps, returning the raw decoded text as well (no caching)."""
    import torch
    from task5_feedback.rlaif import PAIRWISE_RUBRIC
    key = judge._key(problem, a, b)
    swap = int(key[:8], 16) % 2 == 1
    aa, bb = (b, a) if swap else (a, b)
    text = PAIRWISE_RUBRIC.format(problem=problem, a=aa, b=bb)
    ids = judge.tokenizer.apply_chat_template([{"role": "user", "content": text}], return_tensors="pt",
                                              add_generation_prompt=True).to(next(judge.model.parameters()).device)
    with torch.no_grad():
        out = judge.model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=judge.tokenizer.eos_token_id,
                                   eos_token_id=judge.tokenizer.eos_token_id)
    raw = judge.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    decoded = raw.strip().upper()
    m = re.search(r"\b(A|B|TIE)\b", decoded)
    result = m.group(1) if m else "TIE"
    if swap:
        result = FLIP[result]
    return {"key": key, "swap": swap, "raw": raw, "parsed": bool(m), "result": result}


def audit_items(cfg, base):
    """Every judge comparison behind the official numbers, in the order the supplied code makes them."""
    from task5_feedback.rlaif import PairwiseAIJudge
    from task5_feedback.score_perturbations import PAIR_TYPES, VARIANT_ORDER, load_diagnostic_groups, problem_of, response_of
    items = []
    D = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    for pid, V in sorted(D.items()):
        problem, clean = problem_of(V["clean_correct"]), response_of(V["clean_correct"])
        for ptype, vt in PAIR_TYPES.items():
            pert = response_of(V[vt])
            items.append({"source": "diagnostic", "role": "headline", "pair": ptype, "problem_id": pid, "problem": problem, "a": clean, "b": pert})
            items.append({"source": "diagnostic", "role": "reversed", "pair": ptype, "problem_id": pid, "problem": problem, "a": pert, "b": clean})
        resp = [response_of(V[vt]) for vt in VARIANT_ORDER]
        for i, j in itertools.combinations(range(len(resp)), 2):          # the supplied group_rewards loop
            items.append({"source": "diagnostic", "role": "group", "pair": f"{VARIANT_ORDER[i]}|{VARIANT_ORDER[j]}", "problem_id": pid,
                          "i": i, "j": j, "problem": problem, "a": resp[i], "b": resp[j]})
    for ds in ["gsm", "transfer"]:
        f = base / ds / "pairs.jsonl"
        if not f.exists():
            continue
        G = {p: {r["id"]: r for r in read_jsonl(base / ds / f"generations_{p}.jsonl")} for p in ["sft", "rlvr", "rlaif"]}
        for r in read_jsonl(f):
            if r["identical"]:
                continue
            x, y = G[r["x"]][r["id"]], G[r["y"]][r["id"]]
            items.append({"source": ds, "role": "policy", "pair": f"{r['x']}_vs_{r['y']}", "id": r["id"],
                          "problem": x["problem"], "a": x["response"], "b": y["response"]})
    keyer = object.__new__(PairwiseAIJudge); keyer.cfg = cfg            # the judge's own cache key, without loading a model
    for it in items:
        it["key"] = keyer._key(it["problem"], it["a"], it["b"])
    return items


def _audit_rows(items, done, cache):
    rows = []
    for it in items:
        d = done.get(it["key"])
        if d is None:
            continue
        c = cache.get(it["key"])
        rows.append({**{k: v for k, v in it.items() if k not in ("problem", "a", "b")}, **d, "cls": raw_class(d["raw"]),
                     "cached": c, "matches_cache": (c == d["result"]) if c is not None else None})
    return rows


def judge_audit(cfg, base):
    items = audit_items(cfg, base)
    path = base / "probe" / "judge_raw.jsonl"
    done = {}
    if path.exists():                                                   # resume: reuse earlier raw outputs by cache key
        for r in read_jsonl(path):
            if "key" in r and "raw" in r:
                done[r["key"]] = {k: r[k] for k in ("swap", "raw", "parsed", "result")}
    todo = [k for k in dict.fromkeys(it["key"] for it in items) if k not in done]
    print(f"judge audit: {len(items)} comparisons ({len({it['key'] for it in items})} unique); "
          f"{len({it['key'] for it in items}) - len(todo)} already probed, {len(todo)} to run", flush=True)
    cache_file = base / "judge_cache.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    if todo:
        from task5_feedback.rlaif import PairwiseAIJudge
        judge = PairwiseAIJudge(cfg, cache_file)
        by_key = {it["key"]: it for it in items}
        for n, key in enumerate(todo, 1):
            it = by_key[key]
            res = raw_compare(judge, it["problem"], it["a"], it["b"])
            done[key] = {k: res[k] for k in ("swap", "raw", "parsed", "result")}
            if n % 25 == 0 or n == len(todo):
                write_jsonl(path, _audit_rows(items, done, cache))            # checkpoint
                print(f"  probed {n}/{len(todo)}", flush=True)
    rows = _audit_rows(items, done, cache)
    write_jsonl(path, rows)
    return rows


def judge_report(base, rows):
    """All official judge numbers recomputed from the raw outputs, plus their echo-as-tie versions."""
    from task5_feedback.evaluate_math import agreement_stats
    from task5_feedback.score_perturbations import PAIR_TYPES, VARIANT_ORDER
    U = {}
    for r in rows:
        U.setdefault(r["key"], r)                                       # unique comparisons (headline = group i=0)
    uniq = list(U.values())
    cls = collections.Counter(r["cls"] for r in uniq)
    fid = [r["matches_cache"] for r in uniq if r["matches_cache"] is not None]
    rep = {"n_comparisons": len(rows), "n_unique": len(uniq)}
    rep["outputs"] = {
        "classes": dict(cls), "echo_rate": cls["ECHO"] / len(uniq) if uniq else float("nan"),
        "fidelity_matches_cache": float(np.mean(fid)) if fid else None, "n_not_in_cache": sum(r["cached"] is None for r in uniq),
        "parse_failure_rate": float(np.mean([not r["parsed"] for r in uniq])) if uniq else float("nan"),
        "tie_rate_official": float(np.mean([r["result"] == "TIE" for r in uniq])) if uniq else float("nan"),
        "real_tie_rate": cls["TIE"] / len(uniq) if uniq else float("nan"),
        "by_source": {s: dict(collections.Counter(r["cls"] for r in uniq if r["source"] == s)) for s in sorted({r["source"] for r in uniq})},
        "echo_texts": dict(collections.Counter(r["raw"].strip() for r in uniq if r["cls"] == "ECHO")),
        "other_texts": dict(collections.Counter(r["raw"].strip() for r in uniq if r["cls"] == "OTHER")),
        "most_common_raw": collections.Counter(r["raw"].strip()[:30] for r in uniq).most_common(12)}

    # diagnostics: headline (clean first), reversed-order check, S_reason / S_outcome
    H = {(r["problem_id"], r["pair"]): r for r in rows if r["role"] == "headline"}
    Rv = {(r["problem_id"], r["pair"]): r for r in rows if r["role"] == "reversed"}
    diag = {}
    for ptype in PAIR_TYPES:
        hs = [r for (pid, p), r in sorted(H.items()) if p == ptype]
        if not hs:
            continue
        d = {"n": len(hs), "echoes": sum(r["cls"] == "ECHO" for r in hs),
             "official": _rates([r["result"] for r in hs]), "echo_as_tie": _rates([corrected(r) for r in hs])}
        both = [(H[k], Rv[k]) for k in sorted(H) if k[1] == ptype and k in Rv]
        if both:
            d["echoes_reversed"] = sum(v["cls"] == "ECHO" for _, v in both)
            d["order_consistency_official"] = float(np.mean([h["result"] == FLIP[v["result"]] for h, v in both]))
            d["order_consistency_echo_as_tie"] = float(np.mean([corrected(h) == FLIP[corrected(v)] for h, v in both]))
        diag[ptype] = d
    rep["diagnostics"] = diag
    for name, ptype in [("S_reason_judge", "reasoning"), ("S_outcome_judge", "outcome")]:
        if ptype in diag:
            rep[name] = {"official": diag[ptype]["official"]["better"], "echo_as_tie": diag[ptype]["echo_as_tie"]["better"]}

    # RLAIF training-time group reward over the 5 variants
    K = len(VARIANT_ORDER)
    Gr = collections.defaultdict(dict)
    for r in rows:
        if r["role"] == "group":
            Gr[r["problem_id"]][(r["i"], r["j"])] = r
    full = {pid: g for pid, g in Gr.items() if len(g) == K * (K - 1) // 2}
    if full:
        off = np.array([group_reward(g, K, lambda r: r["result"]) for g in full.values()]).mean(0)
        cor = np.array([group_reward(g, K, corrected) for g in full.values()]).mean(0)
        ech = collections.Counter()
        for g in full.values():
            for (i, j), r in g.items():
                if r["cls"] == "ECHO":
                    ech[VARIANT_ORDER[i]] += 1; ech[VARIANT_ORDER[j]] += 1
        rep["group_reward"] = {"n_problems": len(full), "official": dict(zip(VARIANT_ORDER, off.tolist())),
                               "echo_as_tie": dict(zip(VARIANT_ORDER, cor.tolist())),
                               "echo_comparisons": sum(r["cls"] == "ECHO" for g in full.values() for r in g.values()),
                               "echoes_involving_variant": {v: ech[v] for v in VARIANT_ORDER},
                               "top_official": VARIANT_ORDER[int(np.argmax(off))], "top_echo_as_tie": VARIANT_ORDER[int(np.argmax(cor))]}

    # policy pairs: win rate and verifier-judge agreement over ALL pairs (identical answers stay ties)
    pol = {}
    for ds in ["gsm", "transfer"]:
        f = base / ds / "pairs.jsonl"
        if not f.exists():
            continue
        A = {(r["pair"], str(r["id"])): r for r in rows if r["source"] == ds and r["role"] == "policy"}
        by = collections.defaultdict(list)
        for r in read_jsonl(f):
            by[f"{r['x']}_vs_{r['y']}"].append(r)
        for pair, d in by.items():
            off, cor, vd, ech, mism = [], [], [], 0, 0
            for r in d:
                a, s = A.get((pair, str(r["id"]))), r["x_score"]
                if a is not None:
                    mism += a["result"] != r["judge"]
                    if a["cls"] == "ECHO":
                        s = 0.5; ech += 1
                elif not r["identical"]:
                    mism += 1
                off.append(r["x_score"]); cor.append(s); vd.append(r["verifier_diff"])
            pol[f"{ds}/{pair}"] = {"n": len(d), "echoes": ech, "audit_mismatch": mism, "official": _winrate(off), "echo_as_tie": _winrate(cor),
                                   "verifier_judge_official": agreement_stats(vd, off), "verifier_judge_echo_as_tie": agreement_stats(vd, cor)}
    rep["policy_pairs"] = pol

    # position: did a decisive verdict go to the answer SHOWN first?
    def pos(rs):
        real = [r for r in rs if r["cls"] in ("A", "B")]
        k = sum(r["cls"] == "A" for r in real)
        dec = [r for r in rs if r["result"] != "TIE"]
        kp = sum(shown_first(r) for r in dec)
        return {"real_decisive": len(real), "real_first_shown": k, "real_first_shown_rate": k / len(real) if real else None,
                "real_first_shown_ci95": wilson(k, len(real)), "parser_decisive": len(dec), "parser_first_shown": kp,
                "parser_first_shown_rate": kp / len(dec) if dec else None}
    rep["position"] = {"all": pos(uniq), **{s: pos([r for r in uniq if r["source"] == s]) for s in sorted({r["source"] for r in uniq})}}
    return rep


def print_report(rep):
    o = rep["outputs"]
    print(f"\nraw judge outputs ({rep['n_unique']} unique comparisons): {o['classes']} | echo rate {o['echo_rate']:.3f} | "
          f"fidelity {o['fidelity_matches_cache']} | parse failures {o['parse_failure_rate']:.3f}")
    print("diagnostics (judge, clean shown as a)   official B/T/W   ->  echo-as-tie B/T/W   | order consistency official -> echo-as-tie")
    for p, v in rep["diagnostics"].items():
        s, c = v["official"], v["echo_as_tie"]
        oc = f"{v.get('order_consistency_official', float('nan')):.2f} -> {v.get('order_consistency_echo_as_tie', float('nan')):.2f}"
        print(f"  {p:10s} echoes {v['echoes']:2d}   {s['better']:.2f}/{s['tie']:.2f}/{s['wrong']:.2f}  ->  {c['better']:.2f}/{c['tie']:.2f}/{c['wrong']:.2f}   | {oc}")
    if "group_reward" in rep:
        g = rep["group_reward"]
        print(f"RLAIF group reward ({g['n_problems']} problems, {g['echo_comparisons']} echo comparisons): top {g['top_official']} -> {g['top_echo_as_tie']}")
        for v in g["official"]:
            print(f"  {v:34s} {g['official'][v]:.3f} -> {g['echo_as_tie'][v]:.3f}")
    print("win rates                      official (W/T/L)      ->  echo-as-tie (W/T/L)")
    for k, v in rep["policy_pairs"].items():
        s, c = v["official"], v["echo_as_tie"]
        print(f"  {k:24s} {s['win_rate']:.3f} ({s['wins']}/{s['ties']}/{s['losses']})  ->  {c['win_rate']:.3f} ({c['wins']}/{c['ties']}/{c['losses']})")
    p = rep["position"]["all"]
    print(f"position: real decisive {p['real_decisive']}, first-shown {p['real_first_shown_rate']} (95% CI {p['real_first_shown_ci95']}) | "
          f"parser decisive {p['parser_decisive']}, first-shown {p['parser_first_shown_rate']}")


# ------------------------------------------------------------------ (b) adapter magnitude
def adapter_delta_norms(adapter_dir, base_weights):
    """Relative LoRA update size per adapted matrix: ||(alpha/r) B A||_F / ||W||_F."""
    from safetensors.torch import load_file
    ad = repo_path(adapter_dir)
    conf = json.load(open(ad / "adapter_config.json"))
    scale = float(conf["lora_alpha"]) / float(conf["r"])
    sd = load_file(str(ad / "adapter_model.safetensors"))
    rel, absn = [], []
    for k, A in sd.items():
        if ".lora_A." not in k:
            continue
        B = sd[k.replace(".lora_A.", ".lora_B.")]
        dW = scale * (B.float() @ A.float())
        base_key = k.replace("base_model.model.", "").replace(".lora_A.weight", ".weight").replace(".lora_A.default.weight", ".weight")
        W = base_weights.get(base_key)
        absn.append(float(dW.norm()))
        if W is not None:
            rel.append(float(dW.norm() / W.float().norm()))
    return {"modules": len(absn), "scale_alpha_over_r": scale, "abs_mean": float(np.mean(absn)) if absn else float("nan"),
            "rel_mean": float(np.mean(rel)) if rel else float("nan"), "rel_max": float(np.max(rel)) if rel else float("nan"),
            "base_matched": len(rel)}


def load_base_weights(model_id):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from pathlib import Path
    path = Path(snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"]))
    W = {}
    for f in sorted(path.glob("*.safetensors")):
        W.update({k: v for k, v in load_file(str(f)).items() if k.endswith(("q_proj.weight", "v_proj.weight", "k_proj.weight", "o_proj.weight"))})
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--only", default="judge,lenient,adapters", help="comma-separated subset of judge, lenient, adapters")
    args = ap.parse_args()
    parts = set(args.only.split(","))
    cfg = load_yaml(args.config)
    base = repo_path(cfg["results_dir"]) / "task5_feedback"
    (base / "probe").mkdir(parents=True, exist_ok=True)
    import pandas as pd

    if "judge" in parts:
        rows = judge_audit(cfg, base)
        rep = judge_report(base, rows)
        save_json(base / "probe" / "judge_audit.json", rep)
        (base / "probe" / "judge_raw_summary.json").unlink(missing_ok=True)     # superseded by judge_audit.json
        print_report(rep)

    if "lenient" in parts:
        L = pd.DataFrame(lenient_table(base)); L.to_csv(base / "probe" / "lenient_accuracy.csv", index=False)
        print(L[["dataset", "policy", "exact_accuracy", "lenient_accuracy", "hash_present_but_unparsed"]].round(3).to_string(index=False))

    if "adapters" in parts:
        W = load_base_weights(cfg["base_model"])
        rows = []
        for name, path in [("dpo", "outputs/task1_dpo/standard"), ("ppo", "outputs/task2_ppo/standard"), ("grpo", "outputs/task3_grpo/standard"),
                           ("ppo_midpoint", cfg["paths"]["ppo_midpoint_policy"]), ("grpo_midpoint", cfg["paths"]["grpo_midpoint_policy"]),
                           ("rlvr", cfg["policies"]["rlvr"]), ("rlaif", cfg["policies"]["rlaif"])]:
            if (repo_path(path) / "adapter_model.safetensors").exists():
                rows.append({"adapter": name, "path": path, **adapter_delta_norms(path, W)})
        A = pd.DataFrame(rows); A.to_csv(base / "probe" / "adapter_norms.csv", index=False)
        print(A.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
