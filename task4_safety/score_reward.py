from __future__ import annotations

"""Task 4 supplementary instrument (for RQ1): score every XSTest response with the SAME fixed reward
model used in Tasks 2-3.

Question it answers: does the reward signal that drove PPO/GRPO prefer the safe behaviour?
e.g. on UNSAFE prompts, does it score refusals above harmful compliance? on SAFE prompts, does it
score answers above over-refusals? If not, maximising reward cannot be expected to improve safety
calibration -- a direct counterexample for RQ1.

Each unique (prompt, response) is scored once with batch size 1 (no padding, so no padding-dependent
score noise; Task 1 showed batched scores of identical text can differ by ~0.1).
Output: results/task4_safety/rm_scores.jsonl  (key, xstest_id, reward)
"""

import argparse
import json
import time

import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import score_reward_pairs
from common.models import clear_gpu, load_reward_model
from task4_safety.judge_responses import POLICY_ORDER, load_cache, pair_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = repo_path(cfg["results_dir"]) / "task4_safety"
    rows = [r for p in POLICY_ORDER for r in read_jsonl(out / f"generated_{p}.jsonl")]
    path = out / "rm_scores.jsonl"
    done = load_cache(path)
    todo = {}
    for r in rows:
        k = pair_key(r["prompt"], r["response"])
        if k not in done:
            todo[k] = r
    print(f"{len(rows)} pairs, {len(set(pair_key(r['prompt'], r['response']) for r in rows))} unique, {len(todo)} to score")
    if not todo:
        return
    rm, rm_tok = load_reward_model(cfg)
    t0 = time.perf_counter()
    with open(path, "a", encoding="utf-8") as fh, torch.no_grad():
        for i, (k, r) in enumerate(todo.items()):
            s = float(score_reward_pairs(rm, rm_tok, [[{"role": "user", "content": r["prompt"]}]], [r["response"]],
                                         max_length=int(cfg.get("reward_max_length", 1024)))[0])
            fh.write(json.dumps({"key": k, "xstest_id": r["xstest_id"], "reward": s}) + "\n"); fh.flush()
            if (i + 1) % 100 == 0 or i + 1 == len(todo):
                el = time.perf_counter() - t0
                print(f"  scored {i+1}/{len(todo)} | ~{el/(i+1)*(len(todo)-i-1)/60:.1f} min left", flush=True)
    del rm
    clear_gpu()


if __name__ == "__main__":
    main()
