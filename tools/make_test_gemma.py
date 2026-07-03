"""Generate a tiny random-weight Gemma 3 checkpoint (HF-loadable).

Exercises every Gemma-3-specific code path at toy scale: (1+w) norms,
embedding scaling, qk-norm, sandwich norms, GELU-tanh, decoupled head_dim
(32 != hidden/heads), a non-default query_pre_attn_scalar, and a tiny
sliding window (8) with a 2-layer local/global pattern so the reference
prompts (up to 28 tokens) actually hit the window edge.

Needs transformers >= 4.50 (Gemma3 classes) — run on Modal, not the lab box.
The output is an HF directory; convert with tools/convert_hf.py for the
engine and dump the oracle with tools/gen_reference.py.

Usage:
    python tools/make_test_gemma.py --out-hf /tmp/hf-tiny-gemma
"""

import argparse

import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-hf", required=True, help="output HF checkpoint dir")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    n_layers = 4
    layer_types = ["sliding_attention" if (i + 1) % 2 else "full_attention"
                   for i in range(n_layers)]
    config = Gemma3TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,  # deliberately != hidden/heads (16)
        vocab_size=512,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        rope_local_base_freq=10_000.0,
        sliding_window=8,
        sliding_window_pattern=2,
        layer_types=layer_types,
        query_pre_attn_scalar=24,  # non-default: proves the scale plumbing
        attention_bias=False,
        hidden_activation="gelu_pytorch_tanh",
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    torch.manual_seed(args.seed)
    model = Gemma3ForCausalLM(config).float().eval()
    model.save_pretrained(args.out_hf, safe_serialization=True)
    n = sum(t.numel() for t in model.parameters())
    print(f"wrote {args.out_hf} ({n:,} params, layer_types={layer_types})")


if __name__ == "__main__":
    main()
