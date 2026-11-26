"""Convert a model into one flat .BIN file (format v2) for the bare-metal engine.

  python export.py SRC OUT NAME [--bits 8|4] [--head-bits 8|4] [--gguf FILE] ...

SRC is a Hugging Face folder (Qwen3 dense or Qwen3.5 hybrid: config.json, *.safetensors,
tokenizer.json). With --gguf FILE the weights come from a GGUF file (PrismML Bonsai Q1_0 / PQ2_0 /
PTQ1_0 / Q2_0, or F16/BF16/F32/Q8_0 tensors) and SRC only supplies tokenizer + chat template.

Output > 0xF0000000 bytes is split into OUT (part 0), <stem>.B01, <stem>.B02 ... (raw byte split;
the concatenation is the whole file). NAME is the display name (bytes 192..255 of the header).

Layout (little-endian, sections 64-byte aligned):
  header (256 bytes, see ModelHeader in src/engine.h)
  rope    f32 [rope_len][rope_dim/2][2] (cos, sin), YaRN attention factor folded in
  params  f32, per layer: attn_norm[dim], ffn_norm[dim], then full: q_norm[hd] k_norm[hd],
          linear: conv_w[conv_dim][conv_k], dt_bias[nv], A[nv] (= -exp(A_log)), gnorm[dv];
          then final_norm[dim]. Zero-centred norms (Qwen3.5) are stored as 1+w.
  matrices: data, then a MatDesc table (off_mats, n_mats) in this order:
          embed, head (== embed desc when tied), per layer: full: wq wk wv wo | linear: in_qkv
          in_z in_a in_b out, then w1 w3 w2. Weight types:
            Q8  int8 codes, fp16 scale per 32
            Q4  per 32: 16 bytes (byte i = q[i] | q[i+16] << 4), fp16 scale (+ fp16 min if asym)
                per gs; sym: w = s*(q-8), asym: w = s*q + m
            Q1  per 128: 16 bytes of sign bits (LSB first, 1 = +s), fp16 scale per gs
            T2  per 128: 32 bytes, code of weight j at byte j%32 bits 2*(j/32); w = s*(code-1)
  hadamard (optional): u32 block, n; u32 widths[n]; f32 signs (concatenated)
  tokenizer: see write_tokenizer()
"""
import argparse, glob, json, math, os, struct, sys, hashlib
import numpy as np
import torch

PART = 0xF0000000
ALIGN = 64
WT_Q8, WT_Q4, WT_Q1, WT_T2 = 0, 1, 2, 3
MF_HAD = 1
HF_ATTN_GATE, HF_V_TILED, HF_HADAMARD, HF_GDN_V_GROUPED = 1, 2, 4, 8
CS_QWEN3, CS_QWEN35, CS_NOTHINK, CS_QWEN38 = 0, 1, 2, 3

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("name")
ap.add_argument("--gguf")
ap.add_argument("--bits", type=int, default=8, choices=[4, 8])
ap.add_argument("--head-bits", type=int, choices=[4, 8], help="lm_head bits (default: --bits)")
ap.add_argument("--embed-bits", type=int, choices=[4, 8], help="separate embedding-lookup table bits (default: shared with head when tied)")
ap.add_argument("--attn-bits", type=int, choices=[4, 8], help="bits for attention / linear-attention matrices (default: --bits; MLP always uses --bits)")
ap.add_argument("--q4", default="asym", choices=["sym", "asym"])
ap.add_argument("--q4-gs", type=int, default=32, choices=[32, 64, 128])
ap.add_argument("--ctx", type=int, help="rope table length (default 32768 Qwen3, 65536 otherwise)")
ap.add_argument("--tok-src", help="HF folder for tokenizer/template if not SRC")
args = ap.parse_args()

def f32(t): return t.to(torch.float32).numpy() if isinstance(t, torch.Tensor) else np.asarray(t, np.float32)

# ------------------------------------------------------------------ output with part splitting
class Out:
    def __init__(self, path):
        self.path = path; self.f = open(path, "wb"); self.pos = 0
        self.f.write(b"\0" * 256); self.pos = 256
    def align(self):
        pad = (-self.pos) % ALIGN
        if pad: self.f.write(b"\0" * pad); self.pos += pad
        return self.pos
    def section(self, data):
        off = self.align()
        if isinstance(data, np.ndarray): data = data.tobytes()
        self.f.write(data); self.pos += len(data)
        return off
    def finish(self, header):
        self.align()
        self.f.seek(0); self.f.write(header); self.f.close()
        stem = os.path.splitext(self.path)[0]
        for old in glob.glob(stem + ".B[0-9][0-9]"): os.remove(old)
        size = os.path.getsize(self.path)
        parts = [self.path]
        if size > PART:
            with open(self.path, "r+b") as f:
                n = 1
                while PART * n < size:
                    f.seek(PART * n)
                    pp = f"{stem}.B{n:02d}"
                    with open(pp, "wb") as g:
                        left = min(PART, size - PART * n)
                        while left:
                            b = f.read(min(left, 64 << 20)); g.write(b); left -= len(b)
                    parts.append(pp); n += 1
                f.truncate(PART)
        return size, parts

# ------------------------------------------------------------------ quantizers -> (type, gs, w bytes, s, m)
def fp16(a): return np.asarray(a, np.float32).astype(np.float16)

def quant_q8(w):
    R, C = w.shape; assert C % 32 == 0
    g = w.reshape(R, C // 32, 32)
    s = fp16(np.abs(g).max(axis=2) / 127.0)
    sf = s.astype(np.float32); inv = np.where(sf > 0, 1.0 / np.where(sf > 0, sf, 1), 0)
    q = np.clip(np.rint(g * inv[:, :, None]), -127, 127).astype(np.int8)
    return WT_Q8, 32, q.reshape(R, C).tobytes(), s.tobytes(), None

def pack_q4(q):  # q uint8 [R, C] values 0..15 -> [R, C/2]
    R, C = q.shape
    b = q.reshape(R, C // 32, 2, 16)
    return (b[:, :, 0, :] | (b[:, :, 1, :] << 4)).astype(np.uint8).reshape(R, C // 2)

def quant_q4(w, mode, gs):
    R, C = w.shape; assert C % gs == 0
    g = w.reshape(R, C // gs, gs)
    if mode == "sym":   # w = s * (q - 8): ggml Q4_0 scale (largest magnitude -> -8) plus a search
        idx = np.abs(g).argmax(axis=2)      # over smaller scales, keeping the least squared error
        vmax = np.take_along_axis(g, idx[:, :, None], 2)[:, :, 0]
        best_err = None
        for f in (1.0, 0.96, 0.92, 0.88, 0.84, -8 / 7):
            s = fp16(vmax / -8.0 * f); sf = s.astype(np.float32)
            inv = np.where(sf != 0, 1.0 / np.where(sf != 0, sf, 1), 0)
            q = np.clip(np.rint(g * inv[:, :, None]) + 8, 0, 15)
            err = (((q - 8) * sf[:, :, None] - g) ** 2).sum(axis=2)
            if best_err is None: best_err, bq, bs = err, q, s
            else:
                better = err < best_err
                best_err = np.where(better, err, best_err)
                bq = np.where(better[:, :, None], q, bq); bs = np.where(better, s, bs)
        return WT_Q4, gs, pack_q4(bq.astype(np.uint8).reshape(R, C)).tobytes(), fp16(bs).tobytes(), None
    # asym: min/max with a small search over the clipping range (minimise squared error)
    lo = g.min(axis=2); hi = g.max(axis=2)
    best_err = None
    for shrink in (1.0, 0.97, 0.94, 0.9):
        mid = (lo + hi) / 2; l2 = mid + (lo - mid) * shrink; h2 = mid + (hi - mid) * shrink
        s = fp16((h2 - l2) / 15.0); m = fp16(l2)
        sf = s.astype(np.float32); mf = m.astype(np.float32)
        inv = np.where(sf > 0, 1.0 / np.where(sf > 0, sf, 1), 0)
        q = np.clip(np.rint((g - mf[:, :, None]) * inv[:, :, None]), 0, 15)
        err = ((q * sf[:, :, None] + mf[:, :, None] - g) ** 2).sum(axis=2)
        if best_err is None:
            best_err, bq, bs, bm = err, q, s, m
        else:
            better = err < best_err
            best_err = np.where(better, err, best_err)
            bq = np.where(better[:, :, None], q, bq); bs = np.where(better, s, bs); bm = np.where(better, m, bm)
    return WT_Q4, gs, pack_q4(bq.astype(np.uint8).reshape(R, C)).tobytes(), fp16(bs).tobytes(), fp16(bm).tobytes()

def quant_float(w, bits):
    w = np.ascontiguousarray(w, np.float32)
    out_w, out_s, out_m = [], [], []
    step = max(1, (1 << 24) // w.shape[1])
    for r in range(0, w.shape[0], step):
        t, gs, a, b, c = quant_q8(w[r:r + step]) if bits == 8 else quant_q4(w[r:r + step], args.q4, args.q4_gs)
        out_w.append(a); out_s.append(b)
        if c is not None: out_m.append(c)
    return t, gs, b"".join(out_w), b"".join(out_s), (b"".join(out_m) if out_m else None)

def t2_pack(codes):  # codes uint8 [R, C] in 0..3 -> our T2 layout [R, C/4]
    R, C = codes.shape; assert C % 128 == 0
    c = codes.reshape(R, C // 128, 4, 32).astype(np.uint8)
    return (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).reshape(R, C // 4)

from gguf_read import *   # GGUF reader, GG_* type ids

def gguf_matrix(g, name, row_perm=None):
    """Native low-bit tensor -> our type without dequantising; float tensors -> quantised."""
    ty, rows, cols, qk, b = g.raw(name)
    if ty in (GG_Q1_0, GG_PQ1_0, GG_Q2_0, GG_PQ2_0, GG_PTQ1_0):
        codes, d, kind = g.ternary_or_bits(ty, b, qk)
        if row_perm is not None: codes, d = codes[row_perm], d[row_perm]
        s = d.astype(np.float16).tobytes()
        if kind == "q1":
            assert cols % 128 == 0
            return WT_Q1, qk, np.packbits(codes.reshape(rows, cols), axis=1, bitorder="little").tobytes(), s, None
        return WT_T2, qk, t2_pack(codes.reshape(rows, cols)).tobytes(), s, None
    w = g.float(name)
    if row_perm is not None: w = w[row_perm]
    return quant_float(w, args.bits)

# ------------------------------------------------------------------ model description
cfg = {}
gg = GGUF(args.gguf) if args.gguf else None
tok_dir = args.tok_src or args.src
if os.path.exists(f"{args.src}/config.json"): cfg = json.load(open(f"{args.src}/config.json"))
tc = cfg.get("text_config", cfg)

if gg:
    A = gg.kv["general.architecture"]; P = A + "."
    kv = lambda k, d=None: gg.kv.get(P + k, d)
    arch = 1 if A == "qwen35" else 0
    assert A in ("qwen3", "qwen35"), A
    dim, hidden, L = kv("embedding_length"), kv("feed_forward_length"), kv("block_count")
    nh, nkv = kv("attention.head_count"), kv("attention.head_count_kv")
    hd = kv("attention.key_length"); V = gg.t["token_embd.weight"][0][1]
    theta = float(kv("rope.freq_base")); eps = float(kv("attention.layer_norm_rms_epsilon"))
    rope_dim = kv("rope.dimension_count", hd)
    if arch == 1:
        nk, dk = kv("ssm.group_count"), kv("ssm.state_size"); nv = kv("ssm.time_step_rank"); dv = kv("ssm.inner_size") // nv
        conv_k = kv("ssm.conv_kernel"); interval = kv("full_attention_interval")
        full = [(l + 1) % interval == 0 for l in range(L)]
    else:
        nk = nv = dk = dv = conv_k = 0; full = [True] * L
    tied = "output.weight" not in gg.t
    scaling = None
    if kv("rope.scaling.type") == "yarn":
        scaling = {"rope_type": "yarn", "factor": float(kv("rope.scaling.factor")),
                   "original_max_position_embeddings": int(kv("rope.scaling.original_context_length"))}
else:
    arch = 1 if tc.get("model_type", "").startswith("qwen3_5") else 0
    dim, hidden, L = tc["hidden_size"], tc["intermediate_size"], tc["num_hidden_layers"]
    nh, nkv = tc["num_attention_heads"], tc["num_key_value_heads"]
    hd = tc.get("head_dim") or dim // nh; V = tc["vocab_size"]
    rp = tc.get("rope_parameters") or {}
    theta = float(rp.get("rope_theta", tc.get("rope_theta", 10000.0))); eps = float(tc["rms_norm_eps"])
    rope_dim = int(hd * rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 1.0)))
    if arch == 1:
        nk, nv, dk, dv = tc["linear_num_key_heads"], tc["linear_num_value_heads"], tc["linear_key_head_dim"], tc["linear_value_head_dim"]
        conv_k = tc["linear_conv_kernel_dim"]; full = [t == "full_attention" for t in tc["layer_types"]]
    else:
        nk = nv = dk = dv = conv_k = 0; full = [True] * L
    tied = bool(cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings", False)))
    scaling = tc.get("rope_scaling") or (rp if rp.get("rope_type", "default") != "default" else None)

rope_len = args.ctx or (32768 if arch == 0 and not scaling else 65536)
conv_dim = 2 * nk * dk + nv * dv
print(f"arch={'qwen35' if arch else 'qwen3'} dim={dim} hidden={hidden} L={L} heads={nh}/{nkv}x{hd} rope_dim={rope_dim} "
      f"V={V} lin={nk}/{nv}x{dk}/{dv} full={sum(full)} tied={tied} rope_len={rope_len} scaling={scaling}")

# ------------------------------------------------------------------ weights access
if gg:
    def P_(l, n): return f"blk.{l}.{n}"
else:
    import safetensors
    from safetensors import safe_open
    files = sorted(glob.glob(f"{args.src}/*.safetensors"))
    handles = [safe_open(f, "pt") for f in files]
    keymap = {}
    for h in handles:
        for k in h.keys(): keymap[k] = h
    pre = "model.language_model." if arch == 1 else "model."
    def T(n):
        return keymap[n].get_tensor(n)
    def P_(l, n): return f"{pre}layers.{l}.{n}"

# ------------------------------------------------------------------ rope table
def rope_table():
    inv = 1.0 / (theta ** (np.arange(0, rope_dim, 2, dtype=np.float64) / rope_dim))
    att = 1.0
    if scaling and scaling.get("rope_type", scaling.get("type")) == "yarn":
        # transformers _compute_yarn_parameters (static YaRN)
        factor = scaling["factor"]; orig = scaling.get("original_max_position_embeddings")
        bf, bs_ = scaling.get("beta_fast", 32), scaling.get("beta_slow", 1)
        mscale = scaling.get("mscale"); mscale_all = scaling.get("mscale_all_dim")
        def get_mscale(s, m=1.0): return 1.0 if s <= 1 else 0.1 * m * math.log(s) + 1.0
        att = scaling.get("attention_factor")
        if att is None:
            att = get_mscale(factor, mscale) / get_mscale(factor, mscale_all) if (mscale and mscale_all) else get_mscale(factor)
        def corr_dim(nrot): return (rope_dim * math.log(orig / (nrot * 2 * math.pi))) / (2 * math.log(theta))
        low, high = corr_dim(bf), corr_dim(bs_)
        if scaling.get("truncate", True): low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, rope_dim - 1)
        if low == high: high += 0.001
        ramp = np.clip((np.arange(rope_dim // 2, dtype=np.float32) - low) / (high - low), 0, 1)
        extra_mask = 1 - ramp
        pos_freqs = theta ** (np.arange(0, rope_dim, 2).astype(np.float32) / rope_dim)
        inv_extra = 1.0 / pos_freqs; inv_inter = 1.0 / (factor * pos_freqs)
        inv = (inv_inter * (1 - extra_mask) + inv_extra * extra_mask).astype(np.float32).astype(np.float64)
    inv32 = inv.astype(np.float32)
    ang = np.arange(rope_len, dtype=np.float32)[:, None] * inv32[None, :]   # fp32 like torch
    return (np.stack([np.cos(ang.astype(np.float64)) * att, np.sin(ang.astype(np.float64)) * att], -1)).astype(np.float32)

out = Out(args.out)
off_rope = out.section(rope_table())

# ------------------------------------------------------------------ hadamard metadata (GGUF Bonsai 2)
had_names, had_inv, had_bs, had_signs, v_grouped = set(), set(), 0, {}, False
if gg and "prism.hadamard.version" in gg.kv:
    assert gg.kv["prism.hadamard.transform"] == "normalized-sylvester-walsh-hadamard"
    assert gg.kv["prism.hadamard.axis"] == "input-last-dimension"
    had_bs = gg.kv["prism.hadamard.block_size"]
    had_names = set(gg.kv["prism.hadamard.weight_names"]); had_inv = set(gg.kv.get("prism.hadamard.inverse_weight_names", []))
    if gg.kv["prism.hadamard.sign_mode"] == "explicit":
        vals = np.array(gg.kv["prism.hadamard.sign_values"], np.float32); o = 0
        for wd in gg.kv["prism.hadamard.sign_widths"]: had_signs[wd] = vals[o:o + wd]; o += wd
    v_grouped = bool(gg.kv.get("prism.hadamard.gdn_v_grouped", False))
    assert gg.kv["prism.hadamard.version"] == 1 or not tied

# ------------------------------------------------------------------ params
params = []
def pnorm(x, zc):  # zero-centred RMSNorm weight -> 1 + w
    x = f32(x); return x + 1.0 if zc else x
for l in range(L):
    if gg:
        g = lambda n: gg.float(P_(l, n))
        params += [g("attn_norm.weight"), g("post_attention_norm.weight" if arch == 1 else "ffn_norm.weight")]
        if full[l]: params += [g("attn_q_norm.weight"), g("attn_k_norm.weight")]
        else:
            cw = gg.float(P_(l, "ssm_conv1d.weight"))     # [conv_dim, conv_k]
            params += [cw.reshape(conv_dim, conv_k), g("ssm_dt.bias"), g("ssm_a"), g("ssm_norm.weight")]
    else:
        zc = arch == 1
        params += [pnorm(T(P_(l, "input_layernorm.weight")), zc), pnorm(T(P_(l, "post_attention_layernorm.weight")), zc)]
        if full[l]: params += [pnorm(T(P_(l, "self_attn.q_norm.weight")), zc), pnorm(T(P_(l, "self_attn.k_norm.weight")), zc)]
        else:
            la = lambda n: T(P_(l, "linear_attn." + n))
            params += [f32(la("conv1d.weight")).reshape(conv_dim, conv_k), f32(la("dt_bias")),
                       -np.exp(f32(la("A_log"))), f32(la("norm.weight"))]
params.append(gg.float("output_norm.weight") if gg else pnorm(T(pre + "norm.weight"), arch == 1))
off_params = out.section(np.concatenate([np.asarray(p, np.float32).reshape(-1) for p in params]))

# ------------------------------------------------------------------ matrices
descs = []
def add(m, rows, cols, flags=0):
    t, gs, w, s, mm = m
    ow = out.section(w); os_ = out.section(s); om = out.section(mm) if mm is not None else 0
    descs.append(struct.pack("<6I3Q", t, rows, cols, gs, flags, 0, ow, os_, om))
    return descs[-1]

head_bits = args.head_bits or args.bits
print("matrices...")
if gg:
    ty, rows, cols, _, _ = gg.raw("token_embd.weight")
    flags = MF_HAD if "token_embd.weight" in had_inv else 0   # embedding rows need the inverse transform
    emb = add(gguf_matrix(gg, "token_embd.weight"), V, dim, flags)
    if tied: descs.append(emb)
    else: add(gguf_matrix(gg, "output.weight"), V, dim, MF_HAD if "output.weight" in had_names else 0)
else:
    E = f32(T(pre + "embed_tokens.weight"))
    emb_bits = args.embed_bits or (head_bits if tied else args.bits)
    add(quant_float(E, emb_bits), V, dim)
    if tied and emb_bits == head_bits: descs.append(descs[-1])
    else: add(quant_float(E if tied else f32(T("lm_head.weight")), head_bits), V, dim)
    del E

qd, kvd = nh * hd, nkv * hd
gated = arch == 1
def gate_perm():  # HF q_proj rows per head [q_h(hd), gate_h(hd)] -> [all q][all gates]
    idx = np.arange(2 * qd).reshape(nh, 2, hd)
    return np.concatenate([idx[:, 0].reshape(-1), idx[:, 1].reshape(-1)])

for l in range(L):
    if gg:
        def gm(n, rows, cols, perm=None):
            name = P_(l, n + ".weight")
            add(gguf_matrix(gg, name, perm), rows, cols, MF_HAD if name in had_names else 0)
        if full[l]:
            gm("attn_q", qd * (2 if gated else 1), dim, gate_perm() if gated else None)
            gm("attn_k", kvd, dim); gm("attn_v", kvd, dim); gm("attn_output", dim, qd)
        else:
            gm("attn_qkv", conv_dim, dim); gm("attn_gate", nv * dv, dim)
            gm("ssm_alpha", nv, dim); gm("ssm_beta", nv, dim); gm("ssm_out", dim, nv * dv)
        gm("ffn_gate", hidden, dim); gm("ffn_up", hidden, dim); gm("ffn_down", dim, hidden)
    else:
        def hm(n, rows, cols, perm=None):
            w = f32(T(P_(l, n + ".weight")))
            if perm is not None: w = w[perm]
            add(quant_float(w, args.bits if n.startswith("mlp.") else (args.attn_bits or args.bits)), rows, cols)
        if full[l]:
            hm("self_attn.q_proj", qd * (2 if gated else 1), dim, gate_perm() if gated else None)
            hm("self_attn.k_proj", kvd, dim); hm("self_attn.v_proj", kvd, dim); hm("self_attn.o_proj", dim, qd)
        else:
            hm("linear_attn.in_proj_qkv", conv_dim, dim); hm("linear_attn.in_proj_z", nv * dv, dim)
            hm("linear_attn.in_proj_a", nv, dim); hm("linear_attn.in_proj_b", nv, dim); hm("linear_attn.out_proj", dim, nv * dv)
        hm("mlp.gate_proj", hidden, dim); hm("mlp.up_proj", hidden, dim); hm("mlp.down_proj", dim, hidden)
    print(f"\r  layer {l + 1}/{L}", end="", flush=True)
print()
off_mats = out.section(b"".join(descs))

off_had = 0
if had_bs:
    widths = sorted(had_signs)
    blob = struct.pack("<II", had_bs, len(widths)) + np.array(widths, np.uint32).tobytes()
    blob += b"".join(had_signs[w].astype(np.float32).tobytes() for w in widths)
    off_had = out.section(blob)

# ------------------------------------------------------------------ tokenizer
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

def unicode_classes(tok, cache_key):
    """Class of every code point for the pre-tokenizer regex, probed on the real HF pre-tokenizer.
    0 = other, 1 = letter (\\p{L}, plus \\p{M} for Qwen3.5), 2 = number, 3 = whitespace."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", f"uclass_{cache_key}.npy")
    if os.path.exists(path): return np.load(path)
    pt = tok.backend_tokenizer.pre_tokenizer
    cls = np.zeros(0x110000, np.uint8)
    print("probing unicode classes (cached afterwards)...")
    for c in range(0x110000):
        if 0xD800 <= c < 0xE000: continue
        ch = chr(c)
        if len(pt.pre_tokenize_str("a" + ch)) == 1: cls[c] = 1
        elif len(pt.pre_tokenize_str("!" + ch)) == 1 and ch not in "\r\n": cls[c] = 0
        elif len(pt.pre_tokenize_str(ch + "a")) == 1 or ch in "\r\n": cls[c] = 3
        else: cls[c] = 2
    os.makedirs(os.path.dirname(path), exist_ok=True); np.save(path, cls)
    return cls

def nfc_tables(norm):
    # Python's unicodedata is newer than the tables inside HF tokenizers: marks the HF
    # normalizer does not reorder (ccc 0 there) are probed and kept at ccc 0.
    import unicodedata as ud
    ccc = []; decomp = []; data = []; comp = []
    for c in range(0x110000):
        if 0xD800 <= c < 0xE000: continue
        ch = chr(c)
        k = ud.combining(ch)
        if k:
            probe = "á" + ch if k == 1 else "a" + ch + "̴"   # misordered vs ccc 230 / ccc 1
            if norm is None or norm.normalize_str(probe) == ud.normalize("NFC", probe): ccc.append((c, k))
        if 0xAC00 <= c <= 0xD7A3: continue   # Hangul: algorithmic
        d = ud.normalize("NFD", ch)
        if d != ch:
            decomp.append((c, len(data), len(d))); data += [ord(x) for x in d]
        dm = ud.decomposition(ch)
        if dm and not dm.startswith("<"):
            parts = [int(x, 16) for x in dm.split()]
            if len(parts) == 2 and ud.normalize("NFC", chr(parts[0]) + chr(parts[1])) == ch:
                comp.append(((parts[0] << 21) | parts[1], c))
    comp.sort()
    ccc_a = np.array(ccc, np.uint32).reshape(-1, 2)
    dec_a = np.array(decomp, np.uint32).reshape(-1, 3)
    comp_a = np.array([(k & 0xFFFFFFFF, k >> 32, v) for k, v in comp], np.uint32).reshape(-1, 3)
    return ccc_a, dec_a, np.array(data, np.uint32), comp_a

def chat_style(tmpl):
    if "reasoning_effort" in tmpl: return CS_QWEN38
    if "enable_thinking" not in tmpl: return CS_NOTHINK
    if "render_content" in tmpl: return CS_QWEN35
    return CS_QWEN3

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(tok_dir)
tj = json.loads(tok.backend_tokenizer.to_str())
regex = tj["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
letter_m = "[\\p{L}\\p{M}]+" in regex
assert regex in ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                 "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"), regex
assert tj["normalizer"] is None or tj["normalizer"]["type"] == "NFC", tj["normalizer"]
nfc = tj["normalizer"] is not None

u2b = bytes_to_unicode()
vocab = [b""] * V
for t_, i in tj["model"]["vocab"].items():
    if i < V: vocab[i] = bytes(u2b[c] for c in t_)
added = []
special = {}
for at in tj["added_tokens"]:
    if at["id"] < V:
        vocab[at["id"]] = at["content"].encode("utf-8")
        special[at["content"]] = at["id"]
        added.append(at["id"])
tok2id = tj["model"]["vocab"]
merges = []
for m in tj["model"]["merges"]:
    a, b = m.split(" ") if isinstance(m, str) else m
    merges.append((tok2id[a], tok2id[b], tok2id[a + b]))
byte_tok = [tok2id[[c for c, b in u2b.items() if b == x][0]] for x in range(256)]
if gg:   # the GGUF vocabulary must be the HF tokenizer's
    gt = gg.kv["tokenizer.ggml.tokens"]; gtype = gg.kv.get("tokenizer.ggml.token_type") or [1] * len(gt)
    for i in range(min(len(gt), V)):
        if gtype[i] == 5 and not vocab[i]: continue   # GGUF fills ids unused by HF with [PADn] (type UNUSED)
        exp = vocab[i] if i not in added else None
        if exp is not None and bytes(u2b[c] for c in gt[i]) != exp and gt[i].encode() != exp:
            raise SystemExit(f"GGUF token {i} {gt[i]!r} differs from HF tokenizer")
    tmpl = gg.kv.get("tokenizer.chat_template") or tok.chat_template
else:
    tmpl = tok.chat_template
style = chat_style(tmpl)
think_sys = b""
if style == CS_QWEN38:   # thinking mode prepends the default reasoning-effort instruction to the system prompt
    r = tok.apply_chat_template([{"role": "user", "content": "U"}], tokenize=False, add_generation_prompt=True, chat_template=tmpl)
    think_sys = r.split("<|im_start|>system\n", 1)[1].split("<|im_end|>", 1)[0].encode() if "<|im_start|>system\n" in r else b""
print(f"chat style {style}, letter_m={letter_m}, nfc={nfc}, {len(added)} added tokens")

key = hashlib.sha1(regex.encode()).hexdigest()[:8]
cls = unicode_classes(tok, key)
ranges = []
prev = -1
for c in range(0x110000):
    if cls[c] != prev: ranges.append((c, int(cls[c]))); prev = cls[c]
rng_a = np.array([(c << 8) | k for c, k in ranges], np.uint32)
ccc_a, dec_a, dec_data, comp_a = nfc_tables(tok.backend_tokenizer.normalizer) if nfc else (np.zeros((0, 2), np.uint32),) * 2 + (np.zeros(0, np.uint32), np.zeros((0, 3), np.uint32))

tb = bytearray(); offs = []
for v in vocab: offs.append(len(tb)); tb.extend(v)
offs.append(len(tb))
blobs = [np.array(offs, np.uint32), bytes(tb), np.array(merges, np.uint32), np.array(byte_tok, np.uint32),
         rng_a, ccc_a, dec_a, dec_data, comp_a, np.array(sorted(added), np.uint32), think_sys]
# tokenizer header: 16 u32 counts/flags then 16 u64 offsets (relative to off_tok)
th_counts = [V, len(merges), len(ranges), len(ccc_a), len(dec_a), len(dec_data), len(comp_a), len(added),
             len(think_sys), (1 if letter_m else 0) | (2 if nfc else 0), style] + [0] * 5
tok_hdr_size = 16 * 4 + 16 * 8
body = bytearray(); rel = []
for bl in blobs:
    while (tok_hdr_size + len(body)) % ALIGN: body.append(0)
    rel.append(tok_hdr_size + len(body)); body.extend(bl.tobytes() if isinstance(bl, np.ndarray) else bl)
off_tok = out.section(struct.pack("<16I", *th_counts) + struct.pack("<16Q", *(rel + [0] * (16 - len(rel)))) + bytes(body))

sp = lambda s: special[s]
flags = (HF_ATTN_GATE if gated else 0) | (HF_V_TILED if (arch == 1 and gg) else 0) | (HF_HADAMARD if had_bs else 0) | \
        (HF_GDN_V_GROUPED if v_grouped else 0)
full_mask = sum(1 << l for l in range(L) if full[l])
hdr = struct.pack("<4sII" + "I" * 10 + "I" * 6 + "ff" + "Q" + "Q" * 6 + "I" * 6,
    b"QWN3", 2, arch,
    dim, hidden, L, nh, nkv, hd, V, rope_len, 32, rope_dim,
    nk, nv, dk, dv, conv_k, flags,
    theta, eps, full_mask,
    off_rope, off_params, off_mats, off_tok, off_had, len(descs),
    len(merges), sp("<|im_start|>"), sp("<|im_end|>"), sp("<|endoftext|>"), sp("<think>"), sp("</think>"))
assert len(hdr) <= 192, len(hdr)
hdr = bytearray(hdr.ljust(192, b"\0"))
name = args.name.encode()[:63]
hdr += name.ljust(64, b"\0")
size, parts = out.finish(bytes(hdr))
print(f"wrote {', '.join(parts)}: {size / 2**20:.1f} MiB total, {len(merges)} merges, {len(descs)} matrices")
