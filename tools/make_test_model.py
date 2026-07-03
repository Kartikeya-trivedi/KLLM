"""Generate a tiny random-weight Llama-style checkpoint for Phase 0.

Writes config.json + model.safetensors (numpy only, no torch/safetensors
dependency) that HF `AutoModelForCausalLM.from_pretrained` can load, so the
same weights drive both gen_reference.py and the Go engine. There is no
tokenizer: drive it with raw token ids (gen_reference.py --raw-ids).

Usage:
    python tools/make_test_model.py --out testmodels/tiny-llama
"""

import argparse
import json
from pathlib import Path

import numpy as np

DTYPE_NAMES = {"float32": "F32", "float16": "F16"}


def save_safetensors(path: Path, tensors: dict):
    """Minimal safetensors writer: 8-byte LE header length, JSON header, data."""
    header = {}
    offset = 0
    parts = []
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        raw = arr.tobytes()
        header[name] = {
            "dtype": DTYPE_NAMES[arr.dtype.name],
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        parts.append(raw)
    header_json = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(len(header_json).to_bytes(8, "little"))
        f.write(header_json)
        for raw in parts:
            f.write(raw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="testmodels/tiny-llama")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--intermediate-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-kv-heads", type=int, default=2)
    p.add_argument("--vocab-size", type=int, default=512)
    args = p.parse_args()

    h = args.hidden_size
    head_dim = h // args.num_heads
    kv_dim = args.num_kv_heads * head_dim
    rng = np.random.default_rng(args.seed)

    def linear(rows, cols):
        return (rng.standard_normal((rows, cols)) * 0.02).astype(np.float32)

    def norm_weight():
        return (1.0 + 0.01 * rng.standard_normal(h)).astype(np.float32)

    tensors = {"model.embed_tokens.weight": linear(args.vocab_size, h)}
    for i in range(args.num_layers):
        pre = f"model.layers.{i}"
        tensors[f"{pre}.self_attn.q_proj.weight"] = linear(h, h)
        tensors[f"{pre}.self_attn.k_proj.weight"] = linear(kv_dim, h)
        tensors[f"{pre}.self_attn.v_proj.weight"] = linear(kv_dim, h)
        tensors[f"{pre}.self_attn.o_proj.weight"] = linear(h, h)
        tensors[f"{pre}.mlp.gate_proj.weight"] = linear(args.intermediate_size, h)
        tensors[f"{pre}.mlp.up_proj.weight"] = linear(args.intermediate_size, h)
        tensors[f"{pre}.mlp.down_proj.weight"] = linear(h, args.intermediate_size)
        tensors[f"{pre}.input_layernorm.weight"] = norm_weight()
        tensors[f"{pre}.post_attention_layernorm.weight"] = norm_weight()
    tensors["model.norm.weight"] = norm_weight()
    tensors["lm_head.weight"] = linear(args.vocab_size, h)

    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": h,
        "intermediate_size": args.intermediate_size,
        "num_hidden_layers": args.num_layers,
        "num_attention_heads": args.num_heads,
        "num_key_value_heads": args.num_kv_heads,
        "head_dim": head_dim,
        "vocab_size": args.vocab_size,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "bos_token_id": 1,
        "eos_token_id": 2,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    save_safetensors(out / "model.safetensors", tensors)

    total = sum(t.size for t in tensors.values())
    print(f"wrote {out} ({len(tensors)} tensors, {total:,} params, seed {args.seed})")


if __name__ == "__main__":
    main()
