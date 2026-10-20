from __future__ import annotations

"""Held-out evaluation for GRPO adapters (midpoint, standard, normalisation forks).

Same protocol as Task 2's evaluation (kept self-contained here so Task 3 runs without Task 2's
files): first `n_prompts` rows of the fixed held-out RL prompt pool, the config's sampled decoding,
seed reset before generation, and a 768-token cap (the cache's generation cap, and the same cap
Task 2 used for frozen evaluation, so the two tasks' held-out numbers use one protocol).
Metrics: reward (mean/std), KL from reference (course sampled estimator, token mean), entropy
(sampled helper and exact), length (mean/std/median/IQR), truncation rate, distinct-2.
"""

import argparse
import re

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode

EVAL_CAP_DEFAULT = 768


def length_stats(x):
    a = np.asarray(x, dtype=float)
    q25, q50, q75 = np.percentile(a, [25, 50, 75])
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "median": float(q50), "iqr": float(q75 - q25), "n": int(len(a))}


def distinct2(text):
    w = re.findall(r"\w+", text.lower())
    b = list(zip(w, w[1:]))
    return len(set(b)) / len(b) if b else 1.0


@torch.no_grad()
def rl_generation_eval(model, tok, rm, rm_tok, rows, cfg, max_new_tokens, batch_size=4, seed=None):
    gen = cfg["generation"]
    tok.truncation_side = "left"
    if seed is not None:
        set_seed(seed)
    recs = []
    kl_num = kl_den = ent_num = samp_num = 0.0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        msgs = [prompt_messages(r) for r in chunk]
        out = batch_generate(model, tok, msgs, max_prompt_length=int(cfg["max_prompt_length"]), max_new_tokens=int(max_new_tokens),
                             temperature=float(gen["temperature"]), top_p=float(gen["top_p"]), do_sample=bool(gen["do_sample"]))
        rewards = score_reward_pairs(rm, rm_tok, msgs, out["responses"], max_length=int(cfg.get("reward_max_length", 1280)))
        pw = out["prompt_width"]
        for i in range(len(chunk)):
            a = (out["sequences"][i:i + 1], out["attention_mask"][i:i + 1], pw, out["response_ids"][i:i + 1])
            m = out["response_mask"][i:i + 1].float()
            lp, logits = response_token_logprobs(model, *a)
            lpf = torch.log_softmax(logits.float(), dim=-1)
            ent_tok = -(lpf.exp() * lpf).sum(-1)
            del logits, lpf
            with reference_mode(model):
                rlp, rl = response_token_logprobs(model, *a)
                del rl
            n = float(m.sum()); kl_seq = float(((lp - rlp) * m).sum())
            kl_num += kl_seq; kl_den += n; ent_num += float((ent_tok * m).sum()); samp_num += float((-lp * m).sum())
            recs.append({"index": start + i, "prompt_id": chunk[i].get("prompt_id"), "prompt": msgs[i][-1]["content"],
                         "response": out["responses"][i], "length_tokens": out["response_lengths"][i], "truncated": out["truncated"][i],
                         "reward": float(rewards[i]), "kl_seq_sum": kl_seq, "distinct2": distinct2(out["responses"][i])})
        del out
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {"n_prompts": len(recs), "max_new_tokens": int(max_new_tokens),
            "reward_mean": float(np.mean([r["reward"] for r in recs])), "reward_std": float(np.std([r["reward"] for r in recs], ddof=1)),
            "kl_token_mean": kl_num / max(kl_den, 1.0), "kl_seq_mean": float(np.mean([r["kl_seq_sum"] for r in recs])),
            "entropy_sampled": samp_num / max(kl_den, 1.0), "entropy_exact": ent_num / max(kl_den, 1.0),
            "length_tokens": length_stats([r["length_tokens"] for r in recs]),
            "truncation_rate": float(np.mean([r["truncated"] for r in recs])),
            "distinct2_mean": float(np.mean([r["distinct2"] for r in recs]))}, recs


def evaluate_policy(config_path, adapter, name, n_prompts=100, batch_size=4):
    cfg = load_yaml(config_path)
    adapter = None if adapter in (None, "none", "None") else adapter
    out_dir = repo_path(cfg["results_dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm, rm_tok = load_reward_model(cfg)
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])[:n_prompts]
    cap = int(cfg.get("eval_max_completion_length", cfg.get("cache_generation_cap", EVAL_CAP_DEFAULT)))
    metrics, recs = rl_generation_eval(model, tok, rm, rm_tok, rows, cfg, cap, batch_size, seed=int(cfg["seed"]))
    metrics.update({"name": name, "adapter": adapter, "seed": int(cfg["seed"]), "decoding": cfg["generation"]})
    save_json(out_dir / "eval_metrics.json", metrics)
    write_jsonl(out_dir / "eval_generations.jsonl", recs)
    print(f"[{name}] reward={metrics['reward_mean']:+.3f} KL={metrics['kl_token_mean']:.4f} H={metrics['entropy_sampled']:.3f} "
          f"len={metrics['length_tokens']['mean']:.0f} cut={metrics['truncation_rate']:.0%} distinct2={metrics['distinct2_mean']:.3f}")
    del model, rm
    clear_gpu()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True, help='adapter folder, or "none" for base Qwen')
    ap.add_argument("--name", default="standard")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    evaluate_policy(args.config, args.adapter, args.name, args.n_prompts, args.batch_size)


if __name__ == "__main__":
    main()
