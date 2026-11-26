"""Kernel micro-benchmark: matmat throughput per weight type and ISA, checked against the C reference.

  python tests/bench_kernels.py [--rows 4096] [--cols 4096] [--n 1,4,16] [--threads 1]
"""
import argparse, ctypes as C, time
import numpy as np, qeng
dll = qeng.dll
dll.bench_mem.restype = C.c_size_t
dll.bench_mem.argtypes = [C.c_int] * 5
dll.bench_matmat.argtypes = [C.c_int] * 6 + [C.c_void_p, C.c_void_p, C.c_int]
ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=4096)
ap.add_argument("--cols", type=int, default=4096)
ap.add_argument("--n", default="1,4,16")
ap.add_argument("--types", default="q8,q4sym,q4asym,q1,t2")
ap.add_argument("--threads", type=int, default=1)
args = ap.parse_args()
# thread CPU time: robust when other work shares the machine (single-thread runs only)
clock = time.thread_time if args.threads == 1 else time.perf_counter
if args.threads > 1: qeng.set_threads(args.threads)
avail = dll.eng_detect_isa()
isas = [i for i in (1, 3, 7, 9) if i & avail == i]
names = {1: "avx2", 3: "avx512", 7: "avx512vnni", 9: "avxvnni"}
types = [(0, 32, 0, "q8"), (1, 32, 0, "q4sym"), (1, 32, 1, "q4asym"), (2, 128, 0, "q1"), (3, 128, 0, "t2")]
R, K = args.rows, args.cols
for n in map(int, args.n.split(",")):
    for ty, gs, asym, tn in types:
        if tn not in args.types.split(","): continue
        mem = np.empty(dll.bench_mem(ty, R, K, gs, n), np.uint8)
        out = np.empty(n * R, np.float32); ref = np.empty(n * R, np.float32)
        dll.eng_set_isa(0); dll.bench_matmat(ty, R, K, gs, n, asym, mem.ctypes.data, ref.ctypes.data, 1)
        wbytes = R * K * {0: 1, 1: .5, 2: .125, 3: .25}[ty]
        line = f"n={n:2d} {tn:7s}"
        for isa in isas:
            dll.eng_set_isa(isa)
            dll.bench_matmat(ty, R, K, gs, n, asym, mem.ctypes.data, out.ctypes.data, 0)
            err = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-30))
            it = 0; t0 = clock()
            while True:
                dll.bench_matmat(ty, R, K, gs, n, asym, mem.ctypes.data, out.ctypes.data, 0); it += 1
                dt = clock() - t0
                if dt > 0.6: break
            dt /= it
            line += f" | {names[isa]} {wbytes/dt/1e9:5.1f} GB/s {R*K*n/dt/1e9:5.1f} GMAC/s err {err:.0e}"
        print(line, flush=True)
dll.eng_set_isa(1)
