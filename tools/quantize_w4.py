"""Group-wise symmetric int4 weight quantizer (W4, numpy only).

Quantizes every attention/MLP projection matrix of a Llama-style checkpoint;
embeddings, lm_head, and norms stay fp32. For weight W[out, in], per-group
(along the input dim) scale s = max|w|/7, q = clamp(round(w/s), -8, 7),
stored as (q+8) nibbles packed two per byte (even col = low nibble):

    <base>.qweight  U8  [out, in/2]
    <base>.scales   F32 [out, in/group]

Writes TWO checkpoints:
    <out>       the int4 model the Go engine loads
    <out>-dq    fp32 with dequantized weights — HF loads this, and
                gen_reference dumps on it are the EXACT oracle the W4 engine
                must match (proves the kernel computes dequant*matmul, not
                merely something close).

Usage:
    python tools/quantize_w4.py --model testmodels/tiny-llama \
        --out testmodels/tiny-llama-w4 --group-size 32
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from make_test_model import save_safetensors

QUANT_SUFFIXES = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                  "gate_proj.weight", "up_proj.weight", "down_proj.weight")


def load_safetensors(path: Path) -> dict:
    raw = path.read_bytes()
    hlen = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + hlen])
    data = raw[8 + hlen :]
    out = {}
    dtypes = {"F32": np.float32, "U8": np.uint8}
    for name, ent in header.items():
        if name == "__metadata__":
            continue
        b, e = ent["data_offsets"]
        out[name] = np.frombuffer(data[b:e], dtype=dtypes[ent["dtype"]]).reshape(ent["shape"])
    return out


def quantize(w: np.ndarray, group: int):
    out_dim, in_dim = w.shape
    assert in_dim % group == 0 and in_dim % 2 == 0
    g = w.reshape(out_dim, in_dim // group, group)
    scales = (np.abs(g).max(axis=2) / 7.0).clip(min=1e-8).astype(np.float32)
    q = np.clip(np.round(g / scales[:, :, None]), -8, 7).astype(np.int8)
    dq = (q.astype(np.float32) * scales[:, :, None]).reshape(out_dim, in_dim)
    qu = (q.reshape(out_dim, in_dim) + 8).astype(np.uint8)
    packed = (qu[:, 0::2] | (qu[:, 1::2] << 4)).astype(np.uint8)
    return packed, scales, dq


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--group-size", type=int, default=32)
    args = p.parse_args()

    src = Path(args.model)
    tensors = load_safetensors(src / "model.safetensors")

    qt, dqt = {}, {}
    n_quant = 0
    for name, w in tensors.items():
        if name.endswith(QUANT_SUFFIXES) and w.ndim == 2:
            base = name[: -len(".weight")]
            packed, scales, dq = quantize(np.asarray(w, dtype=np.float32), args.group_size)
            qt[base + ".qweight"] = packed
            qt[base + ".scales"] = scales
            dqt[name] = dq
            n_quant += 1
        else:
            qt[name] = w
            dqt[name] = w

    for out_dir, ts in ((Path(args.out), qt), (Path(str(args.out) + "-dq"), dqt)):
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / "config.json", out_dir / "config.json")
        save_safetensors(out_dir / "model.safetensors", ts)
    cfg = json.loads((Path(args.out) / "config.json").read_text())
    cfg["kllm_quant"] = {"method": "w4", "group_size": args.group_size}
    (Path(args.out) / "config.json").write_text(json.dumps(cfg, indent=2))

    print(f"quantized {n_quant} matrices (group {args.group_size}) -> {args.out} and {args.out}-dq")


if __name__ == "__main__":
    main()
