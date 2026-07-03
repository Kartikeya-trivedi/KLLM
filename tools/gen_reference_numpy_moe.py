"""Numpy oracle for the sigmoid-routed MoE test model (offline only).

HF has no model class combining plain Llama attention with sigmoid +
expert-bias routing, so this reference implements the tiny model directly in
numpy (fp32) and dumps the same format as gen_reference.py: per-layer hidden
states following HF's convention ([0]=embedding, [j]=input to layer j,
last=final-norm output), per-step greedy logits, and token ids.

Routing (Sarvam/DeepSeek-V3 family, scaling factor 1):
    s = sigmoid(gate(x))
    selection: top-k by (s + e_score_correction_bias)
    weights:   s[selected] / sum(s[selected])

Usage:
    python tools/gen_reference_numpy_moe.py --model testmodels/tiny-mixtral-sigmoid \
        --prompts tools/prompts_tiny.txt --out refdumps/tiny-mixtral-sigmoid --max-new-tokens 16
"""

import argparse
import json
from pathlib import Path

import numpy as np

from quantize_w4 import load_safetensors


def rmsnorm(x, w, eps):
    scale = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return x * scale * w


def rope(x, positions, n_heads, head_dim, theta):
    # x: [T, n_heads*head_dim]; HF rotate_half convention.
    T = x.shape[0]
    half = head_dim // 2
    j = np.arange(half, dtype=np.float64)
    freq = theta ** (-2.0 * j / head_dim)          # [half]
    ang = positions[:, None].astype(np.float64) * freq[None, :]  # [T, half]
    cos = np.cos(ang).astype(np.float32)[:, None, :]  # [T, 1, half]
    sin = np.sin(ang).astype(np.float32)[:, None, :]
    xh = x.reshape(T, n_heads, head_dim)
    a, b = xh[:, :, :half], xh[:, :, half:]
    out = np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)
    return out.reshape(T, n_heads * head_dim).astype(np.float32)


def silu(x):
    return x / (1.0 + np.exp(-x))


class TinyMoE:
    def __init__(self, model_dir: Path):
        self.w = load_safetensors(model_dir / "model.safetensors")
        self.cfg = json.loads((model_dir / "config.json").read_text())
        c = self.cfg
        self.L = c["num_hidden_layers"]
        self.H = c["num_attention_heads"]
        self.KV = c["num_key_value_heads"]
        self.hd = c["head_dim"]
        self.n_exp = c["num_local_experts"]
        self.top_k = c["num_experts_per_tok"]
        self.eps = c["rms_norm_eps"]
        self.theta = c["rope_theta"]
        self.kcache = [None] * self.L
        self.vcache = [None] * self.L

    def moe(self, l, x):
        pre = f"model.layers.{l}.block_sparse_moe"
        logits = x @ self.w[f"{pre}.gate.weight"].T           # [T, n_exp]
        s = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))  # sigmoid scores
        biased = s + self.w[f"{pre}.gate.e_score_correction_bias"][None, :]
        out = np.zeros_like(x)
        for t in range(x.shape[0]):
            sel = np.argsort(-biased[t], kind="stable")[: self.top_k]
            wsel = s[t, sel]
            wsel = wsel / wsel.sum()
            for e, wt in zip(sel, wsel):
                epre = f"{pre}.experts.{e}"
                g = silu(x[t] @ self.w[f"{epre}.w1.weight"].T)
                u = x[t] @ self.w[f"{epre}.w3.weight"].T
                out[t] += wt * ((g * u) @ self.w[f"{epre}.w2.weight"].T)
        return out.astype(np.float32)

    def forward(self, ids, pos0, collect_hidden=False):
        T = len(ids)
        positions = np.arange(pos0, pos0 + T)
        x = self.w["model.embed_tokens.weight"][ids].astype(np.float32)
        hidden = [x.copy()] if collect_hidden else None

        for l in range(self.L):
            pre = f"model.layers.{l}"
            xn = rmsnorm(x, self.w[f"{pre}.input_layernorm.weight"], self.eps)
            q = xn @ self.w[f"{pre}.self_attn.q_proj.weight"].T
            k = xn @ self.w[f"{pre}.self_attn.k_proj.weight"].T
            v = xn @ self.w[f"{pre}.self_attn.v_proj.weight"].T
            q = rope(q.astype(np.float32), positions, self.H, self.hd, self.theta)
            k = rope(k.astype(np.float32), positions, self.KV, self.hd, self.theta)

            self.kcache[l] = k if self.kcache[l] is None or pos0 == 0 else np.concatenate([self.kcache[l], k])
            self.vcache[l] = v if self.vcache[l] is None or pos0 == 0 else np.concatenate([self.vcache[l], v])
            K, V = self.kcache[l], self.vcache[l]  # [S, KV*hd]
            S = K.shape[0]

            attn = np.zeros((T, self.H * self.hd), dtype=np.float32)
            gsize = self.H // self.KV
            for hh in range(self.H):
                kvh = hh // gsize
                qh = q[:, hh * self.hd : (hh + 1) * self.hd]        # [T, hd]
                kh = K[:, kvh * self.hd : (kvh + 1) * self.hd]      # [S, hd]
                vh = V[:, kvh * self.hd : (kvh + 1) * self.hd]
                sc = (qh @ kh.T) / np.sqrt(self.hd)                 # [T, S]
                mask = positions[:, None] >= np.arange(S)[None, :]
                sc = np.where(mask, sc, -np.inf).astype(np.float32)
                sc = sc - sc.max(axis=1, keepdims=True)
                p = np.exp(sc)
                p /= p.sum(axis=1, keepdims=True)
                attn[:, hh * self.hd : (hh + 1) * self.hd] = p @ vh
            x = x + attn @ self.w[f"{pre}.self_attn.o_proj.weight"].T

            xn = rmsnorm(x, self.w[f"{pre}.post_attention_layernorm.weight"], self.eps)
            x = x + self.moe(l, xn)
            if collect_hidden:
                hidden.append(x.copy())

        xn = rmsnorm(x, self.w["model.norm.weight"], self.eps)
        if collect_hidden:
            hidden[-1] = xn.copy() if self.L == 0 else hidden[-1]
            hidden = hidden[:-1] + [xn.copy()]  # HF: last entry is post-norm
        logits = xn[-1] @ self.w["lm_head.weight"].T
        return logits.astype(np.float32), hidden


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-new-tokens", type=int, default=16)
    args = p.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    prompts = [ln.strip() for ln in Path(args.prompts).read_text().splitlines() if ln.strip()]

    model_dir = Path(args.model)
    cfg = json.loads((model_dir / "config.json").read_text())
    (out_root / "manifest.json").write_text(json.dumps({
        "model": args.model, "dtype": "float32", "oracle": "numpy",
        "max_new_tokens": args.max_new_tokens,
        "num_layers": cfg["num_hidden_layers"], "prompts": prompts,
    }, indent=2))

    for i, prompt in enumerate(prompts):
        ids = [int(x) for x in prompt.split()]
        pdir = out_root / f"prompt_{i}"
        pdir.mkdir(exist_ok=True)

        m = TinyMoE(model_dir)  # fresh KV per prompt
        logits, hidden = m.forward(ids, 0, collect_hidden=True)
        for j, hs in enumerate(hidden):
            np.save(pdir / f"layer_{j}.npy", hs)

        step_logits, gen = [], []
        cur = logits
        pos = len(ids)
        for _ in range(args.max_new_tokens):
            step_logits.append(cur.copy())
            nxt = int(np.argmax(cur))
            gen.append(nxt)
            if nxt == cfg.get("eos_token_id", -1):
                break
            cur, _ = m.forward([nxt], pos)
            pos += 1

        np.save(pdir / "logits.npy", np.stack(step_logits))
        (pdir / "tokens.json").write_text(json.dumps({
            "prompt": prompt, "input_ids": ids, "generated_ids": gen, "generated_text": "",
        }))
        print(f"[{i}] {prompt!r} -> {gen}")
    print(f"wrote {len(prompts)} numpy reference dumps to {out_root}")


if __name__ == "__main__":
    main()
