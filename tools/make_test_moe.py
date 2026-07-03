"""Generate tiny random-weight MoE checkpoints for Phase 5.

Two variants sharing identical weights:
  <out>          Mixtral-style (softmax top-k renorm) — HF-loadable, so
                 gen_reference.py dumps the oracle.
  <out>-sigmoid  sigmoid + expert-bias selection routing (Sarvam/DSv3
                 family) — adds a gate bias vector; oracle comes from
                 gen_reference_numpy_moe.py (HF has no matching class with
                 plain Llama attention).

Usage:
    python tools/make_test_moe.py --out testmodels/tiny-mixtral
"""

import argparse
import json
from pathlib import Path

import numpy as np

from make_test_model import save_safetensors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="testmodels/tiny-mixtral")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--expert-intermediate", type=int, default=96)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-kv-heads", type=int, default=2)
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=2)
    args = p.parse_args()

    h = args.hidden_size
    rng = np.random.default_rng(args.seed)

    def linear(rows, cols):
        return (rng.standard_normal((rows, cols)) * 0.02).astype(np.float32)

    def norm_weight():
        return (1.0 + 0.01 * rng.standard_normal(h)).astype(np.float32)

    kv_dim = args.num_kv_heads * (h // args.num_heads)
    tensors = {"model.embed_tokens.weight": linear(args.vocab_size, h)}
    for i in range(args.num_layers):
        pre = f"model.layers.{i}"
        tensors[f"{pre}.self_attn.q_proj.weight"] = linear(h, h)
        tensors[f"{pre}.self_attn.k_proj.weight"] = linear(kv_dim, h)
        tensors[f"{pre}.self_attn.v_proj.weight"] = linear(kv_dim, h)
        tensors[f"{pre}.self_attn.o_proj.weight"] = linear(h, h)
        tensors[f"{pre}.input_layernorm.weight"] = norm_weight()
        tensors[f"{pre}.post_attention_layernorm.weight"] = norm_weight()
        # Router gate scaled up so expert choice isn't numerically marginal.
        tensors[f"{pre}.block_sparse_moe.gate.weight"] = (
            rng.standard_normal((args.num_experts, h)) * 0.5).astype(np.float32)
        for e in range(args.num_experts):
            epre = f"{pre}.block_sparse_moe.experts.{e}"
            tensors[f"{epre}.w1.weight"] = linear(args.expert_intermediate, h)
            tensors[f"{epre}.w2.weight"] = linear(h, args.expert_intermediate)
            tensors[f"{epre}.w3.weight"] = linear(args.expert_intermediate, h)
    tensors["model.norm.weight"] = norm_weight()
    tensors["lm_head.weight"] = linear(args.vocab_size, h)

    config = {
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "hidden_size": h,
        "intermediate_size": args.expert_intermediate,
        "num_hidden_layers": args.num_layers,
        "num_attention_heads": args.num_heads,
        "num_key_value_heads": args.num_kv_heads,
        "head_dim": h // args.num_heads,
        "vocab_size": args.vocab_size,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "num_local_experts": args.num_experts,
        "num_experts_per_tok": args.top_k,
        "sliding_window": None,
        "output_router_logits": False,
        "router_aux_loss_coef": 0.0,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "bos_token_id": 1,
        "eos_token_id": 2,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    save_safetensors(out / "model.safetensors", tensors)

    # Sigmoid variant: same weights + selection bias for the gate.
    sig = Path(str(args.out) + "-sigmoid")
    sig.mkdir(parents=True, exist_ok=True)
    sig_tensors = dict(tensors)
    for i in range(args.num_layers):
        sig_tensors[f"model.layers.{i}.block_sparse_moe.gate.e_score_correction_bias"] = (
            rng.standard_normal(args.num_experts) * 0.5).astype(np.float32)
    sig_config = dict(config)
    sig_config["kllm_router"] = "sigmoid_bias"
    (sig / "config.json").write_text(json.dumps(sig_config, indent=2))
    save_safetensors(sig / "model.safetensors", sig_tensors)

    total = sum(t.size for t in tensors.values())
    print(f"wrote {out} and {sig} ({total:,} params, {args.num_experts} experts, top-{args.top_k})")


if __name__ == "__main__":
    main()
