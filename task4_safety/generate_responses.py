from __future__ import annotations

"""Task 4 Step 1: one deterministic response per (policy, XSTest prompt) for the four FIXED policies.

Policies (manual, fixed -- no ablation forks allowed):
  sft  = untouched Qwen2.5-1.5B-Instruct (no adapter)
  dpo  = Task 1 standard adapter      ppo = Task 2 standard adapter      grpo = Task 3 standard adapter
Decoding (supplied, unchanged): greedy (do_sample=False, temperature 0), max_new_tokens =
`safety_max_new_tokens` (256), prompts in the fixed CSV order, identical batch size for every policy.

Output: results/task4_safety/generated_<policy>.jsonl (fixed prompt order) and
        results/task4_safety/generation_manifest.json (adapter fingerprints + decoding settings).
"""

import argparse
import hashlib
import time

import pandas as pd

from common.data import load_yaml, repo_path, write_jsonl, read_jsonl
from common.generation import batch_generate
from common.logging_utils import load_json, save_json
from common.models import clear_gpu, load_policy, load_tokenizer

POLICY_ORDER = ["sft", "dpo", "ppo", "grpo"]


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def adapter_fingerprint(adapter):
    """sha256 of the adapter weights, so the report can prove WHICH adapter produced the responses."""
    if adapter is None:
        return {"adapter": None, "sha256": None}
    p = repo_path(adapter)
    if not (p / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter missing: {p}. Task 4 needs the STANDARD adapter from Tasks 1-3 at exactly this path.")
    h = hashlib.sha256((p / "adapter_model.safetensors").read_bytes()).hexdigest()
    return {"adapter": str(adapter), "sha256": h}


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4, limit: int | None = None, meta: dict | None = None,
                        progress: bool = True):
    """Supplied function. Additions only: optional `limit` (smoke tests), progress printing, and `meta`
    (filled with the EFFECTIVE generation settings). The decoding itself is unchanged."""
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    if meta is not None:
        gc = model.generation_config
        # Qwen's generation_config.json also carries e.g. repetition_penalty, which transformers applies
        # even under greedy decoding -- recorded so the report states the decoding exactly.
        meta.update({k: getattr(gc, k, None) for k in ["repetition_penalty", "top_k", "eos_token_id", "pad_token_id"]})
        meta.update({"do_sample": False, "temperature": 0.0, "top_p": 1.0, "max_prompt_length": 256,
                     "max_new_tokens": int(cfg["safety_max_new_tokens"]), "batch_size": int(batch_size)})
    df = load_xstest(cfg)
    if limit is not None:
        df = df.iloc[: int(limit)]
    records = []
    t0, n_batches = time.perf_counter(), (len(df) + batch_size - 1) // batch_size
    for b, start in enumerate(range(0, len(df), batch_size)):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
        if progress and ((b + 1) % 10 == 0 or b + 1 == n_batches):
            el = time.perf_counter() - t0
            print(f"  [{policy_name}] batch {b+1}/{n_batches} | {el/60:.1f} min elapsed, ~{el/(b+1)*(n_batches-b-1)/60:.1f} min left", flush=True)
    del model
    clear_gpu()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="+", default=POLICY_ORDER, choices=POLICY_ORDER)
    ap.add_argument("--batch-size", type=int, default=16, help="identical for every policy")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = repo_path(cfg["results_dir"]) / "task4_safety"
    out.mkdir(parents=True, exist_ok=True)
    df = load_xstest(cfg)
    print("Policies:", list(policy_specs(cfg)))
    print("XSTest rows:", len(df))
    mpath = out / "generation_manifest.json"
    manifest = load_json(mpath) if mpath.exists() else {"policies": {}}

    for name in args.policies:
        fp = adapter_fingerprint(policy_specs(cfg)[name])          # fail loudly BEFORE spending GPU time
        dst = out / f"generated_{name}.jsonl"
        if args.skip_existing and dst.exists() and len(read_jsonl(dst)) == len(df):
            print(f"[{name}] already generated ({len(df)} rows) -> skipping")
            continue
        meta, t0 = {}, time.perf_counter()
        rows = generate_for_policy(cfg, name, batch_size=args.batch_size, meta=meta)
        assert [r["xstest_id"] for r in rows] == [int(x) for x in df["xstest_id"]], "prompt order changed"
        write_jsonl(dst, rows)
        manifest["policies"][name] = {**fp, "decoding": meta, "n_rows": len(rows), "wall_clock_s": time.perf_counter() - t0}
        save_json(mpath, manifest)
        n = [r["response_tokens"] for r in rows]
        print(f"[{name}] saved {len(rows)} responses -> {dst} | mean length {sum(n)/len(n):.1f} tokens, "
              f"at the {cfg['safety_max_new_tokens']}-token cap: {sum(x >= int(cfg['safety_max_new_tokens']) for x in n)}")


if __name__ == "__main__":
    main()
