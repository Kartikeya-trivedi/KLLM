"""Convert a HuggingFace checkpoint to kllm's fp32 format.

Supported architectures:
  - plain Llama (Llama 2/3, TinyLlama, Sarvam-1, Mistral without sliding):
    RMSNorm, rotate_half RoPE, SiLU MLP, GQA, no bias, no qk-norm
  - Gemma 3 text-only (model_type gemma3_text, e.g. gemma-3-1b): (1+w)
    norms, sandwich norms, qk-norm, GELU-tanh, sliding-window layers

Casts every weight to fp32 (the engine's compute dtype), materializes
lm_head from the embedding if tied, merges shards, copies config.json.
Multimodal Gemma 3 (4B+) and archs with QKV bias (Qwen2) are refused.

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
LLAMA_SUFFIXES = (
    "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    "input_layernorm.weight", "post_attention_layernorm.weight",
)
GEMMA3_SUFFIXES = LLAMA_SUFFIXES + (
    "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
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
    model_type = cfg.get("model_type", "llama")
    if "text_config" in cfg:
        raise SystemExit("multimodal checkpoint (nested text_config) — use the "
                         "text-only variant (e.g. gemma-3-1b) instead")
    is_gemma3 = model_type.startswith("gemma3")
    n_layers = cfg["num_hidden_layers"]
    tensors = gather_tensors(hf_dir)

    # Gemma ties embeddings by default (config may omit the flag entirely).
    if "lm_head.weight" not in tensors and (cfg.get("tie_word_embeddings") or is_gemma3):
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]

    # Validate the architecture matches what the engine implements.
    suffixes = GEMMA3_SUFFIXES if is_gemma3 else LLAMA_SUFFIXES
    required = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for i in range(n_layers):
        for s in suffixes:
            required.add(f"model.layers.{i}.{s}")
    missing = sorted(required - set(tensors))
    if missing:
        raise SystemExit(f"checkpoint is missing {len(missing)} expected tensors "
                         f"(arch mismatch?), e.g. {missing[:3]}")
    bad = ("q_proj.bias", "k_proj.bias", "v_proj.bias")
    if not is_gemma3:
        bad += ("q_norm.weight", "k_norm.weight")
    for name in tensors:
        if name.endswith(bad):
            raise SystemExit(f"unsupported tensor {name!r} — this arch needs kernel "
                             "changes; see convert_hf.py header")

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

    # Write the RUNTIME layer_types into the output config: the formula
    # deriving them from sliding_window_pattern differs across transformers
    # versions, so ask transformers itself what assignment it uses. The
    # engine treats explicit layer_types as authoritative.
    if is_gemma3:
        try:
            from transformers import AutoConfig
            rt = AutoConfig.from_pretrained(hf_dir)
            layer_types = getattr(rt, "layer_types", None)
            if layer_types:
                out_cfg = json.loads((out_dir / "config.json").read_text())
                out_cfg["layer_types"] = list(layer_types)
                (out_dir / "config.json").write_text(json.dumps(out_cfg, indent=2))
                print(f"runtime layer_types: {list(layer_types)}")
        except Exception as e:  # best-effort: old transformers can't load gemma3
            print(f"note: could not resolve runtime layer_types ({e})")

    print(f"converted {len(converted)} tensors, {total:,} params -> {out_dir} (fp32)")


if __name__ == "__main__":
    main()
