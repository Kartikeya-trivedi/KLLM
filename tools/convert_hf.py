"""Convert a HuggingFace Llama-architecture checkpoint to kllm's fp32 format.

The engine speaks plain Llama (RMSNorm, rotate_half RoPE, SiLU-gated MLP,
GQA, no attention/MLP bias, no qk-norm), so this works for Llama 2/3,
TinyLlama, and other exact-arch matches. It casts every weight to fp32
(the engine's Phase-0/1 compute dtype), materializes lm_head from the
embedding if the model ties them, and copies config.json (the engine reads
the standard fields). Sharded checkpoints are merged.

Models that need kernel changes first (NOT handled): Gemma (embedding
scaling, gelu-tanh, logit softcap, qk-norm), Qwen2 (QKV bias), Qwen3 /
anything with per-head qk-norm.

Usage:
    python tools/convert_hf.py --hf <local_dir_or_hub_id> --out testmodels/real
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file

from make_test_model import save_safetensors

# Weights the engine's forward pass actually consumes (per layer, plus
# top-level embed / norm / lm_head). Anything else in the checkpoint is a
# buffer we don't need and is dropped.
LAYER_SUFFIXES = (
    "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    "input_layernorm.weight", "post_attention_layernorm.weight",
)


def gather_tensors(hf_dir: Path) -> dict:
    idx = hf_dir / "model.safetensors.index.json"
    tensors = {}
    if idx.exists():
        weight_map = json.loads(idx.read_text())["weight_map"]
        for shard in sorted(set(weight_map.values())):
            tensors.update(load_file(str(hf_dir / shard)))
    else:
        shards = sorted(hf_dir.glob("*.safetensors"))
        if not shards:
            raise SystemExit(f"no .safetensors files in {hf_dir}")
        for shard in shards:
            tensors.update(load_file(str(shard)))
    return tensors


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf", required=True, help="local checkpoint dir or HF hub id")
    p.add_argument("--out", required=True, help="output directory (engine format)")
    args = p.parse_args()

    hf_dir = Path(args.hf)
    if not hf_dir.exists():
        from huggingface_hub import snapshot_download
        hf_dir = Path(snapshot_download(
            args.hf, ignore_patterns=["*.bin", "*.pth", "*.gguf", "original/*"]))

    cfg = json.loads((hf_dir / "config.json").read_text())
    n_layers = cfg["num_hidden_layers"]
    tensors = gather_tensors(hf_dir)

    if "lm_head.weight" not in tensors and cfg.get("tie_word_embeddings"):
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]

    # Validate the architecture matches what the engine implements.
    required = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for i in range(n_layers):
        for s in LAYER_SUFFIXES:
            required.add(f"model.layers.{i}.{s}")
    missing = sorted(required - set(tensors))
    if missing:
        raise SystemExit(f"checkpoint is missing {len(missing)} expected tensors "
                         f"(arch mismatch?), e.g. {missing[:3]}")
    for name in tensors:
        if name.endswith(("q_proj.bias", "k_proj.bias", "v_proj.bias", "q_norm.weight", "k_norm.weight")):
            raise SystemExit(f"unsupported tensor {name!r} — this arch (bias/qk-norm) "
                             "needs kernel changes; see convert_hf.py header")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    converted = {}
    total = 0
    for name in required:
        t = tensors[name]
        converted[name] = t.float().cpu().numpy()  # bf16/fp16 -> fp32
        total += t.numel()
    save_safetensors(out_dir / "model.safetensors", converted)
    shutil.copy(hf_dir / "config.json", out_dir / "config.json")

    print(f"converted {len(converted)} tensors, {total:,} params -> {out_dir} (fp32)")


if __name__ == "__main__":
    main()
