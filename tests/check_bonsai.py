"""Bonsai GGUF checks.

  python tests/check_bonsai.py dequant GGUF UNPACKED_DIR      # GGUF dequant vs -unpacked FP16 safetensors
  python tests/check_bonsai.py same GGUF_A GGUF_B [EVERY]      # identical weights? (every EVERY-th tensor)
"""
import sys
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gguf_read import GGUF

MAP = {"token_embd.weight": "model.embed_tokens.weight", "output.weight": "lm_head.weight", "output_norm.weight": "model.norm.weight"}
SUB = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
       "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm", "attn_norm": "input_layernorm",
       "ffn_norm": "post_attention_layernorm", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
def hf_name(n):
    if n in MAP: return MAP[n]
    _, l, sub, w = n.split(".")
    return f"model.layers.{l}.{SUB[sub]}.{w}"

if sys.argv[1] == "dequant":
    from safetensors import safe_open
    g = GGUF(sys.argv[2])
    import glob
    f = safe_open(glob.glob(sys.argv[3] + "/*.safetensors")[0], "pt")
    keys = set(f.keys()); worst = 0; exact = n = 0
    for name, (dims, ty, off) in sorted(g.t.items()):
        hn = hf_name(name)
        if hn not in keys: print("missing in HF:", name, hn); continue
        a = g.float(name).astype(np.float32)
        b = f.get_tensor(hn).float().numpy()
        assert a.shape == b.shape, (name, a.shape, b.shape)
        d = float(np.abs(a - b).max()); worst = max(worst, d if ty not in (0, 1, 30) else 0)
        n += 1; exact += d == 0
        if d != 0 and ty not in (0, 1, 30): print(f"  {name} type {ty}: max |diff| {d:.3e}")
    print(f"{n} tensors compared, {exact} bit-exact; worst low-bit tensor max |diff| {worst:.3e}")
    print("HF has", len(keys - {hf_name(x) for x in g.t}), "tensors not in the GGUF:", sorted(keys - {hf_name(x) for x in g.t})[:5])
else:
    A, B = GGUF(sys.argv[2]), GGUF(sys.argv[3])
    every = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    names = sorted(set(A.t) & set(B.t))[::every]; same = 0; worst = 0
    only = sorted(set(A.t) ^ set(B.t))
    for i, name in enumerate(names):
        a, b = A.float(name), B.float(name)
        d = float(np.abs(a.astype(np.float32) - b.astype(np.float32)).max()); same += d == 0; worst = max(worst, d)
        if d: print(f"  {name}: types {A.t[name][1]}/{B.t[name][1]} max |diff| {d:.3e} (max |w| {float(np.abs(a).max()):.3e})")
        if i % 50 == 0: print(f"  ... {i}/{len(names)}", flush=True)
    print(f"{len(names)} shared tensors, {same} identical, worst max |diff| {worst:.3e}; not shared: {only[:6]}")
