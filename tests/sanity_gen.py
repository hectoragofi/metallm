"""Engine-only sanity check for models without an HF reference in RAM: perplexity on the eval
text, greedy chat replies, single-thread decode speed (CPU time) and bytes read per token.

  python tests/sanity_gen.py build/BONSAI8.BIN [--isa 7] [--gen 48]
"""
import argparse, inspect, json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, qeng

ap = argparse.ArgumentParser()
ap.add_argument("bin")
ap.add_argument("--isa", type=int, default=1)
ap.add_argument("--gen", type=int, default=48)
ap.add_argument("--ntok", type=int, default=256)
ap.add_argument("--think", type=int, default=0)
args = ap.parse_args()
qeng.set_isa(args.isa)
t0 = time.time(); e = qeng.Engine(args.bin, 1024); print(f"loaded in {time.time()-t0:.1f}s, arena {e.arena_mb:.0f} MiB")
text = (inspect.getsource(json.decoder)[:3000] + "\n\nThe history of computing is a story of abstraction layers. Early machines "
        "were programmed by rewiring panels; later, stored programs let the same hardware run many tasks.\n")
ids = e.encode(text)[:args.ntok]
L = e.forward_batch(ids, 0, all_logits=True).astype(np.float64)
L -= L.max(1, keepdims=True); lp = L - np.log(np.exp(L).sum(1, keepdims=True))
print(f"ppl on {len(ids)} tokens of eval text: {float(np.exp(-lp[np.arange(len(ids)-1), ids[1:]].mean())):.3f}")
im_end = e.encode("<|im_end|>", special=True)[0]
dt = ng = 0
for p in ["What is the capital of France? Answer in one sentence.", "List the first ten prime numbers, separated by commas.",
          "Write a Python function that returns the factorial of n."]:
    toks = e.chat_turn(p, 1, args.think)
    lg = e.forward_batch(toks, 0); pos = len(toks); out = []
    t0 = time.thread_time()
    for _ in range(args.gen):
        t = int(lg.argmax())
        if t == im_end: break
        out.append(t); lg = e.forward(t, pos, 1); pos += 1
    dt += time.thread_time() - t0; ng += len(out)
    print(f"  {p!r} -> {e.decode(out)!r}")
size = os.path.getsize(args.bin)
print(f"decode {ng/dt:.2f} tok/s (1 thread CPU time, isa {args.isa})")
