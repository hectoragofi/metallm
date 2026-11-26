"""Minimal GGUF reader (memory-mapped) with dequantisation of the PrismML Bonsai formats.

Shared by export.py and the Bonsai checks."""
import struct
import numpy as np

# ------------------------------------------------------------------ GGUF reader
GG_F32, GG_F16, GG_Q8_0, GG_BF16, GG_Q1_0, GG_Q2_0, GG_PQ2_0, GG_PTQ1_0, GG_PQ1_0 = 0, 1, 8, 30, 41, 42, 142, 143, 144
class GGUF:
    FMT = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    def __init__(self, path):
        self.mm = np.memmap(path, np.uint8, "r")
        self.p = 0
        assert bytes(self.mm[:4]) == b"GGUF"; self.p = 4
        ver = self.rd('I'); nt = self.rd('Q'); nkv = self.rd('Q')
        self.kv = {}
        for _ in range(nkv):
            k = self.s(); t = self.rd('I'); self.kv[k] = self.val(t)
        self.t = {}
        for _ in range(nt):
            name = self.s(); nd = self.rd('I'); dims = [self.rd('Q') for _ in range(nd)]
            ty = self.rd('I'); off = self.rd('Q'); self.t[name] = (dims, ty, off)
        al = self.kv.get("general.alignment", 32)
        self.data = (self.p + al - 1) // al * al
    def rd(self, f):
        n = struct.calcsize('<' + f); v = struct.unpack('<' + f, bytes(self.mm[self.p:self.p + n]))[0]; self.p += n; return v
    def s(self):
        n = self.rd('Q'); v = bytes(self.mm[self.p:self.p + n]).decode('utf-8', 'replace'); self.p += n; return v
    def val(self, t):
        if t == 8: return self.s()
        if t == 9:
            et = self.rd('I'); n = self.rd('Q')
            if et == 8: return [self.s() for _ in range(n)]
            if et == 9: return [self.val(et) for _ in range(n)]
            f = self.FMT[et]; sz = struct.calcsize(f)
            a = np.frombuffer(bytes(self.mm[self.p:self.p + n * sz]), '<' + f); self.p += n * sz; return a.tolist()
        return self.rd(self.FMT[t])
    BLK = {GG_F32: (1, 4), GG_F16: (1, 2), GG_BF16: (1, 2), GG_Q8_0: (32, 34), GG_Q1_0: (128, 18), GG_PQ1_0: (64, 10),
           GG_Q2_0: (64, 18), GG_PQ2_0: (128, 34), GG_PTQ1_0: (128, 28)}
    def raw(self, name):
        dims, ty, off = self.t[name]
        cols = dims[0]; rows = int(np.prod(dims[1:])) if len(dims) > 1 else 1
        qk, bs = self.BLK[ty]
        if ty == GG_Q2_0:   # old Bonsai repos store a 128-weight layout under type 42
            n = rows * cols
            if self.next_off(name) - off == n // 128 * 34: qk, bs = 128, 34
        nbytes = rows * (cols // qk) * bs
        a = self.mm[self.data + off: self.data + off + nbytes]
        return ty, rows, cols, qk, np.asarray(a).reshape(rows, cols // qk, bs)
    def next_off(self, name):
        offs = sorted(o for _, _, o in self.t.values()); o = self.t[name][2]
        i = offs.index(o); return offs[i + 1] if i + 1 < len(offs) else len(self.mm) - self.data
    def float(self, name):
        ty, rows, cols, qk, b = self.raw(name)
        if ty == GG_F32: w = b.reshape(-1).view(np.float32)
        elif ty == GG_F16: w = b.reshape(-1).view(np.float16).astype(np.float32)
        elif ty == GG_BF16: w = (b.reshape(-1).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            codes, d, kind = self.ternary_or_bits(ty, b, qk)
            if kind == "q1": w = (codes.astype(np.float32) * 2 - 1)
            elif kind == "t2": w = codes.astype(np.float32) - 1
            else: w = codes.astype(np.float32)
            w = (w.reshape(rows, cols // qk, qk) * d[:, :, None]).reshape(-1)
        dims = self.t[name][0]
        return w.reshape(list(reversed(dims))) if len(dims) > 1 else w
    def ternary_or_bits(self, ty, b, qk):
        R, G, _ = b.shape
        if ty == GG_Q8_0:
            d = b[:, :, :2].copy().view(np.float16)[:, :, 0].astype(np.float32)
            return b[:, :, 2:].copy().view(np.int8).reshape(R, G * 32), d, "q8"
        if ty in (GG_Q1_0, GG_PQ1_0):
            d = b[:, :, :2].copy().view(np.float16)[:, :, 0].astype(np.float32)
            bits = np.unpackbits(b[:, :, 2:], axis=2, bitorder="little")  # [R, G, qk]
            return bits.reshape(R, G * qk), d, "q1"
        if ty in (GG_Q2_0, GG_PQ2_0):
            d = b[:, :, :2].copy().view(np.float16)[:, :, 0].astype(np.float32)
            q = b[:, :, 2:]                                           # [R, G, qk/4]
            codes = np.stack([(q >> (2 * k)) & 3 for k in range(4)], axis=3)   # j = byte*4 + k
            return codes.reshape(R, G * qk), d, "t2"
        if ty == GG_PTQ1_0:
            qs = b[:, :, :24]; qh = b[:, :, 24:26]
            d = b[:, :, 26:28].copy().view(np.float16)[:, :, 0].astype(np.float32)
            pow3 = np.array([1, 3, 9, 27, 81, 243], np.uint16)
            def trits(x, n):  # x [R,G,c] -> [R,G,n,c]
                return np.stack([(((x.astype(np.uint16) * pow3[k]) & 0xFF) * 3) >> 8 for k in range(n)], axis=2)
            parts = [trits(qs[:, :, 0:16], 5).reshape(R, G, 80), trits(qs[:, :, 16:24], 5).reshape(R, G, 40),
                     trits(qh, 4).reshape(R, G, 8)]
            return np.concatenate(parts, axis=2).reshape(R, G * 128).astype(np.uint8), d, "t2"
        raise ValueError(f"unsupported GGUF type {ty}")
