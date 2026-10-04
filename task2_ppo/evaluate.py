from __future__ import annotations

"""Held-out evaluation for an RL policy adapter (PPO midpoint, standard, forks; reused by Task 3).

Protocol (identical for every condition): the first `n_prompts` rows of the fixed held-out
prompt pool, the config's sampled decoding, the seed reset before generation, and the larger
`eval_max_response_length` cap (the README: the midpoint has a long-tail termination problem).

Per condition it reports:
  reward_mean        fixed course reward model, mean over prompts (within-task comparisons only)
  reward_penalized   same, minus missing_eos_penalty for answers cut off at the cap
  kl_token_mean      course sampled estimator: token-level mean of log pi - log pi_ref
  entropy_sampled    course helper: -mean log pi(sampled token)
  entropy_exact      exact next-token entropy, mean over response tokens
  length             mean / std / median / IQR of generated tokens; truncation rate
  distinct2          unique word-bigrams / all word-bigrams per answer (low = repetitive text,
                     a common sign of reward hacking)
"""

import argparse
import re

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode


def length_stats(x):
    a = np.asarray(x, dtype=float)
    q25, q50, q75 = np.percentile(a, [25, 50, 75])
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "median": float(q50), "iqr": float(q75 - q25), "n": int(len(a))}


def distinct2(text):
    w = re.findall(r"\w+", text.lower())
    bigrams = list(zip(w, w[1:]))
    return len(set(bigrams)) / len(bigrams) if bigrams else 1.0


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
        out = batch_generate(model, tok, msgs, max_prompt_length=int(cfg["max_prompt_length"]),
                             max_new_tokens=int(max_new_tokens), temperature=float(gen["temperature"]),
                             top_p=float(gen["top_p"]), do_sample=bool(gen["do_sample"]))
        rewards = score_reward_pairs(rm, rm_tok, msgs, out["responses"],
                                     max_length=int(cfg.get("reward_max_length", 1024)))
        pw = out["prompt_width"]
        for i in range(len(chunk)):   # one row at a time: full-vocabulary logits are large
            args = (out["sequences"][i:i + 1], out["attention_mask"][i:i + 1], pw, out["response_ids"][i:i + 1])
            m = out["response_mask"][i:i + 1].float()
            lp, logits = response_token_logprobs(model, *args)
            logp_full = torch.log_softmax(logits.float(), dim=-1)
            ent_tok = -(logp_full.exp() * logp_full).sum(-1)
            del logits, logp_full
            with reference_mode(model):
                ref_lp, ref_logits = response_token_logprobs(model, *args)
                del ref_logits
            n = float(m.sum())
            kl_seq = float(((lp - ref_lp) * m).sum())
            kl_num += kl_seq; kl_den += n
            ent_num += float((ent_tok * m).sum()); samp_num += float((-lp * m).sum())
            recs.append({"index": start + i, "prompt_id": chunk[i].get("prompt_id"), "prompt": msgs[i][-1]["content"],
                         "response": out["responses"][i], "length_tokens": out["response_lengths"][i],
                         "truncated": out["truncated"][i], "reward": float(rewards[i]), "kl_seq_sum": kl_seq,
                         "entropy_exact": float((ent_tok * m).sum()) / max(n, 1.0), "distinct2": distinct2(out["responses"][i])})
        del out
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    pen = float(cfg.get("missing_eos_penalty", 0.0))
    return {"n_prompts": len(recs), "max_new_tokens": int(max_new_tokens),
            "reward_mean": float(np.mean([r["reward"] for r in recs])),
            "reward_std": float(np.std([r["reward"] for r in recs], ddof=1)),
            "reward_penalized_mean": float(np.mean([r["reward"] - pen * r["truncated"] for r in recs])),
            "kl_token_mean": kl_num / max(kl_den, 1.0), "kl_seq_mean": float(np.mean([r["kl_seq_sum"] for r in recs])),
            "entropy_sampled": samp_num / max(kl_den, 1.0), "entropy_exact": ent_num / max(kl_den, 1.0),
            "length_tokens": length_stats([r["length_tokens"] for r in recs]),
            "truncation_rate": float(np.mean([r["truncated"] for r in recs])),
            "distinct2_mean": float(np.mean([r["distinct2"] for r in recs]))}, recs


def evaluate_policy(config_path, adapter, name, n_prompts=100, batch_size=4, results_dir=None):
    """adapter=None/"none" -> base Qwen (the reference itself; KL must be 0)."""
    cfg = load_yaml(config_path)
    adapter = None if adapter in (None, "none", "None") else adapter
    out_dir = repo_path(results_dir or cfg["results_dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm, rm_tok = load_reward_model(cfg)
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])[:n_prompts]
    cap = int(cfg.get("eval_max_response_length", cfg.get("max_response_length", 512)))
    metrics, recs = rl_generation_eval(model, tok, rm, rm_tok, rows, cfg, cap, batch_size, seed=int(cfg["seed"]))
    metrics.update({"name": name, "adapter": adapter, "seed": int(cfg["seed"]), "decoding": cfg["generation"]})
    save_json(out_dir / "eval_metrics.json", metrics)
    write_jsonl(out_dir / "eval_generations.jsonl", recs)
    print(f"[{name}] reward={metrics['reward_mean']:+.3f} (penalized {metrics['reward_penalized_mean']:+.3f}) "
          f"KL={metrics['kl_token_mean']:.4f} H={metrics['entropy_sampled']:.3f} len={metrics['length_tokens']['mean']:.0f} "
          f"cut={metrics['truncation_rate']:.0%} distinct2={metrics['distinct2_mean']:.3f}")
    del model, rm
    clear_gpu()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True, help='adapter folder, or "none" for base Qwen')
    ap.add_argument("--name", default="standard")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    evaluate_policy(args.config, args.adapter, args.name, args.n_prompts, args.batch_size)


if __name__ == "__main__":
    main()
