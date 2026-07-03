"""Offline correctness oracle (NOT in the serving path).

Dumps a HuggingFace model's greedy tokens, final logits, and per-layer hidden
states for a fixed prompt set. The Go engine's correctness tests load these
dumps and diff against them; per-layer dumps let us binary-search exactly
which kernel/layer diverges.

Output layout (one directory per prompt under --out):
    <out>/manifest.json                  model id, dtype, prompt list
    <out>/prompt_<i>/tokens.json         input ids + greedy continuation ids
    <out>/prompt_<i>/logits.npy          final-position logits per step [steps, vocab]
    <out>/prompt_<i>/layer_<j>.npy       hidden state after layer j for the prompt pass

Usage:
    python tools/gen_reference.py --model <hf-id-or-path> --prompts tools/prompts.txt \
        --out refdumps/<model-name> --max-new-tokens 32
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--prompts", required=True, help="text file, one prompt per line")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device).eval()

    prompts = [
        line.strip()
        for line in Path(args.prompts).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    manifest = {
        "model": args.model,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "num_layers": model.config.num_hidden_layers,
        "prompts": prompts,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for i, prompt in enumerate(prompts):
        pdir = out_root / f"prompt_{i}"
        pdir.mkdir(exist_ok=True)
        input_ids = tok(prompt, return_tensors="pt").input_ids.to(args.device)

        # --- Per-layer hidden states for the prompt (prefill) pass ---
        with torch.no_grad():
            prefill = model(input_ids, output_hidden_states=True)
        # hidden_states[0] is the embedding output; [j+1] is after layer j.
        for j, hs in enumerate(prefill.hidden_states):
            np.save(pdir / f"layer_{j}.npy", hs[0].float().cpu().numpy())

        # --- Greedy decode, capturing final-position logits each step ---
        ids = input_ids
        step_logits = []
        gen_ids = []
        with torch.no_grad():
            past = None
            cur = ids
            for _ in range(args.max_new_tokens):
                out = model(cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1]
                step_logits.append(logits.float().cpu().numpy())
                nxt = int(torch.argmax(logits))
                gen_ids.append(nxt)
                if nxt == tok.eos_token_id:
                    break
                cur = torch.tensor([[nxt]], device=args.device)

        np.save(pdir / "logits.npy", np.stack(step_logits))
        (pdir / "tokens.json").write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "input_ids": input_ids[0].tolist(),
                    "generated_ids": gen_ids,
                    "generated_text": tok.decode(gen_ids),
                }
            )
        )
        print(f"[{i}] {prompt!r} -> {tok.decode(gen_ids)!r}")

    print(f"wrote {len(prompts)} reference dumps to {out_root}")


if __name__ == "__main__":
    main()
