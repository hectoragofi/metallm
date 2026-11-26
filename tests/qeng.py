"""ctypes wrapper around build/qwen.dll for the tests and benchmarks."""
import ctypes as C, os, numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
dll = C.CDLL(os.path.join(HERE, "build", "qwen.dll"))
dll.eng_mem_required.restype = C.c_size_t
dll.eng_mem_required.argtypes = [C.c_void_p, C.c_int]
dll.eng_init.restype = C.c_char_p
dll.eng_init.argtypes = [C.c_void_p, C.c_void_p, C.c_uint64, C.c_int, C.c_void_p, C.c_size_t]
dll.eng_forward.restype = C.POINTER(C.c_float)
dll.eng_forward.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int]
dll.eng_forward_batch.restype = C.POINTER(C.c_float)
dll.eng_forward_batch.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.c_int, C.c_int, C.c_void_p]
dll.eng_encode.restype = C.c_int
dll.eng_encode.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.POINTER(C.c_int), C.c_int]
dll.eng_encode_special.restype = C.c_int
dll.eng_encode_special.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.POINTER(C.c_int), C.c_int]
dll.eng_token_bytes.restype = C.POINTER(C.c_ubyte)
dll.eng_token_bytes.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_int)]
dll.eng_chat_turn.restype = C.c_int
dll.eng_chat_turn.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_char_p, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_int), C.c_int]
dll.eng_set_isa.restype = C.c_int
dll.eng_set_isa.argtypes = [C.c_int]
dll.eng_sample.restype = C.c_int
dll.eng_sample.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_float, C.c_float, C.c_int, C.POINTER(C.c_uint64)]
dll.eng_spec_step.restype = C.c_int
dll.eng_spec_step.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_int), C.c_int, C.c_int, C.c_float, C.c_float, C.c_int,
                              C.POINTER(C.c_uint64), C.POINTER(C.c_int)]

dll.eng_detect_isa.restype = C.c_int
dll.eng_history.restype = C.c_int
dll.eng_history.argtypes = [C.c_void_p, C.POINTER(C.POINTER(C.c_int))]

ISA_AVX2, ISA_AVX512, ISA_VNNI512, ISA_AVXVNNI = 1, 2, 4, 8
# default kernels for tests: AVX2 (QWEN_ISA=<flags> overrides, e.g. 7 = AVX-512 VNNI)
dll.eng_set_isa(int(os.environ.get("QWEN_ISA", "1")) & dll.eng_detect_isa())

def load_model(path):
    """Reads <stem>.BIN plus .B01, .B02 ... parts into one buffer."""
    stem = os.path.splitext(path)[0]
    parts = [path]; n = 1
    while os.path.exists(f"{stem}.B{n:02d}"): parts.append(f"{stem}.B{n:02d}"); n += 1
    size = sum(os.path.getsize(p) for p in parts)
    buf = np.empty(size, np.uint8); o = 0
    for p in parts:
        a = np.fromfile(p, np.uint8); buf[o:o + a.size] = a; o += a.size
    return buf

class Engine:
    def __init__(self, path, max_seq=1024, model=None):
        self.model = load_model(path) if model is None else model
        need = dll.eng_mem_required(self.model.ctypes.data, max_seq)
        self.arena = np.empty(need + 64, np.uint8)
        self.e = (C.c_ubyte * 16384)()
        err = dll.eng_init(self.e, self.model.ctypes.data, self.model.size, max_seq, self.arena.ctypes.data, self.arena.size)
        assert err is None, err
        self.V = int(np.frombuffer(self.model[:64].tobytes(), np.uint32)[9])
        self.arena_mb = need / 2**20
    def forward(self, tok, pos, want=1):
        p = dll.eng_forward(self.e, tok, pos, want)
        return np.ctypeslib.as_array(p, shape=(self.V,)).copy() if want else None
    def forward_batch(self, toks, pos, all_logits=False):
        a = (C.c_int * len(toks))(*toks)
        if all_logits:
            out = np.empty((len(toks), self.V), np.float32)
            dll.eng_forward_batch(self.e, a, len(toks), pos, out.ctypes.data)
            return out
        p = dll.eng_forward_batch(self.e, a, len(toks), pos, None)
        return np.ctypeslib.as_array(p, shape=(self.V,)).copy()
    def encode(self, s, special=False):
        b = s.encode("utf-8") if isinstance(s, str) else s
        out = (C.c_int * (len(b) + 16))()
        n = (dll.eng_encode_special if special else dll.eng_encode)(self.e, b, len(b), out, len(b) + 16)
        return list(out[:n])
    def token_bytes(self, i):
        n = C.c_int(); p = dll.eng_token_bytes(self.e, i, C.byref(n)); return bytes(p[:n.value])
    def decode(self, ids):
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", "replace")
    def chat_turn(self, msg, first, think, sys=None):
        b = msg.encode(); s = sys.encode() if sys else b""
        out = (C.c_int * 65536)()
        n = dll.eng_chat_turn(self.e, s if sys is not None else None, len(s), b, len(b), first, think, out, 65536)
        return list(out[:n])
    def spec_step(self, draft, pos, cur, k, temp=0.0, top_p=1.0, top_k=1, rng=None):
        p = C.c_int(pos); out = (C.c_int * 32)(); r = rng if rng is not None else C.c_uint64(1)
        n = dll.eng_spec_step(self.e, draft.e, C.byref(p), cur, k, temp, top_p, top_k, C.byref(r), out)
        return list(out[:n]), p.value

def set_isa(f): return dll.eng_set_isa(f)

_pw = None
def set_threads(n):
    """Multi-threaded g_parfor via build/parfor_win.dll (test-only shim); n=1 restores serial."""
    global _pw, _serial
    if _pw is None:
        _pw = C.CDLL(os.path.join(HERE, "build", "parfor_win.dll"))
        _serial = C.c_void_p.in_dll(dll, "g_parfor").value
    _pw.parfor_win_threads(n)
    C.c_void_p.in_dll(dll, "g_parfor").value = C.cast(_pw.parfor_win, C.c_void_p).value if n > 1 else _serial
