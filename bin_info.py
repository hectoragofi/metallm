"""Print .BIN header facts and the weight bytes streamed per decoded token.

  python bin_info.py build/*.BIN
"""
import os, struct, sys
import numpy as np

HDR = "<4sII" + "I" * 10 + "I" * 6 + "ff" + "Q" + "Q" * 6 + "I" * 6
TN = {0: "int8", 1: "int4", 2: "1-bit", 3: "ternary"}
for path in sys.argv[1:]:
    stem = os.path.splitext(path)[0]
    parts = [path] + [p for p in (f"{stem}.B{k:02d}" for k in range(1, 20)) if os.path.exists(p)]
    size = sum(os.path.getsize(p) for p in parts)
    with open(path, "rb") as f: head = f.read(256)
    h = struct.unpack(HDR, head[:struct.calcsize(HDR)])
    (magic, ver, arch, dim, hidden, L, nh, nkv, hd, V, rope_len, gs, rope_dim, nk, nv, dk, dv, ck, flags,
     theta, eps, full_mask, off_rope, off_params, off_mats, off_tok, off_had, n_mats) = h[:28]
    name = head[192:256].split(b"\0")[0].decode()
    def read_at(off, n):   # across <stem>.BIN/.B01/... parts
        out = b""
        for p in parts:
            sz = os.path.getsize(p)
            if off < sz and n > 0:
                with open(p, "rb") as f: f.seek(off); c = f.read(min(n, sz - off))
                out += c; n -= len(c); off = 0
            else: off -= sz
        return out
    if True:
        d = np.frombuffer(read_at(off_mats, 48 * n_mats), dtype=[("type", "<u4"), ("rows", "<u4"), ("cols", "<u4"), ("gs", "<u4"),
                                                                       ("flags", "<u4"), ("pad", "<u4"), ("w", "<u8"), ("s", "<u8"), ("m", "<u8")])
    def mbytes(r):
        wb = int(r["rows"]) * int(r["cols"]) * {0: 8, 1: 4, 2: 1, 3: 2}[int(r["type"])] // 8
        ng = int(r["rows"]) * (int(r["cols"]) // int(r["gs"]))
        return wb + 2 * ng * (2 if r["m"] else 1)
    seen = {}; types = {}
    for i, r in enumerate(d):
        seen[int(r["w"])] = mbytes(r)
        t = TN[int(r["type"])]; types[t] = types.get(t, 0) + int(r["rows"]) * int(r["cols"])
    tied = d[0]["w"] == d[1]["w"]
    per_tok = sum(seen.values()) - (0 if tied else mbytes(d[0]))   # embedding: one row per token unless shared with the head
    tot = sum(types.values())
    mix = ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in sorted(types.items(), key=lambda x: -x[1]))
    print(f"{os.path.basename(path):14s} {name!r:28s} {'+'.join(os.path.basename(p) for p in parts)}: {size / 2**20:8.1f} MiB, "
          f"{'qwen3.5' if arch else 'qwen3'} L={L} dim={dim}, weights {mix}; {per_tok / 1e9:.2f} GB/token")
