"""Speculative decoding check + measurement, driven exactly like the UEFI chat loop.

  python tests/test_spec.py --target build/QW35-2.BIN --draft build/QW35-08.BIN [--hf model-q35-2b] [--k 4]

1. temp 0: spec output must equal target-only greedy token for token (several prompts, 2 turns).
2. With --hf: after a 2-turn spec conversation, both engines' token histories equal HF's template.
3. temp > 0: the spec sampler's first-token distribution vs the target's (chi-square-ish check).
Prints acceptance rate (drafted tokens accepted / drafted) and speedup (CPU time, 1 thread).
"""
import argparse, ctypes as C, sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, qeng

ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True)
ap.add_argument("--draft", required=True)
ap.add_argument("--hf")
ap.add_argument("--k", default="2,4,6")
ap.add_argument("--gen", type=int, default=96)
ap.add_argument("--isa", type=int, default=1)
ap.add_argument("--threads", type=int, default=1, help=">1: Windows thread shim, wall time")
ap.add_argument("--samples", type=int, default=0, help="temp>0 distribution check samples")
args = ap.parse_args()
qeng.set_isa(args.isa)
if args.threads > 1: qeng.set_threads(args.threads)
clock = time.thread_time if args.threads == 1 else time.perf_counter
T = qeng.Engine(args.target, 2048); D = qeng.Engine(args.draft, 2048)
im_end = T.encode("<|im_end|>", special=True)[0]
eot = T.encode("<|endoftext|>", special=True)[0]

def history(e):
    p = C.POINTER(C.c_int)(); n = qeng.dll.eng_history(e.e, C.byref(p)); return list(p[:n])

def turn(msgs, pos, first, think, spec_k, temp=0.0, rng=None):
    """One chat turn like efi_main: chat_turn on both engines, prefill both, then generate."""
    toks = T.chat_turn(msgs, first, think, None)
    if spec_k: assert D.chat_turn(msgs, first, think, None) == toks
    lg = T.forward_batch(toks, pos)
    if spec_k: D.forward_batch(toks, pos)
    pos += len(toks)
    r = rng if rng is not None else C.c_uint64(7)
    cur = int(qeng.dll.eng_sample(T.e, lg.ctypes.data_as(C.POINTER(C.c_float)), temp, 0.95, 20, C.byref(r)))
    out, drafted, accepted, steps = [], 0, 0, 0
    t0 = clock()
    while len(out) < args.gen and cur not in (im_end, eot):
        out.append(cur)
        if spec_k:
            got, pos = T.spec_step(D, pos, cur, spec_k, temp, 0.95, 20 if temp > 0 else 1, r)
            drafted += spec_k; accepted += len(got) - 1; steps += 1
            for j, t in enumerate(got[:-1]):
                if t in (im_end, eot):     # reply ends inside the accepted run: stop there
                    pos -= len(got) - 1 - j; got = got[:j + 1]; break
                out.append(t)
            cur = got[-1]
        else:
            lg = T.forward(cur, pos, 1); pos += 1
            cur = int(qeng.dll.eng_sample(T.e, lg.ctypes.data_as(C.POINTER(C.c_float)), temp, 0.95, 20, C.byref(r)))
    dt = clock() - t0
    return out[:args.gen], pos, dt, drafted, accepted, steps

prompts = ["Explain how a hash table works.", "Write a short poem about the sea.", "What are the planets of the solar system?"]
base = {}
for p in prompts:
    out, _, dt, *_ = turn(p, 0, 1, 0, 0)
    base[p] = (out, dt)
for k in map(int, args.k.split(",")):
    tot_dt = tot_base = 0; dr = ac = st = ntok = 0; exact = True
    for p in prompts:
        out, _, dt, d, a, s = turn(p, 0, 1, 0, k)
        exact &= out == base[p][0]
        if out != base[p][0]:
            j = next((i for i, (x, y) in enumerate(zip(out, base[p][0])) if x != y), None)
            print(f"  k={k} MISMATCH on {p!r} at {j}: {T.decode(out[:j+3])!r} vs {T.decode(base[p][0][:j+3])!r}")
        tot_dt += dt; tot_base += base[p][1]; dr += d; ac += a; st += s; ntok += len(out)
    print(f"k={k}: greedy identical to target-only: {exact}; acceptance {ac/dr:.3f} ({ac/st:.2f} drafts/step); "
          f"{ntok/tot_base:.2f} -> {ntok/tot_dt:.2f} tok/s (x{tot_base/tot_dt:.2f}, {args.threads} thread(s))")

# two-turn conversation with spec, then compare histories with HF
if args.hf:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf)
    kw = {"enable_thinking": False} if "enable_thinking" in (tok.chat_template or "") else {}
    pos = 0; msgs = []
    for i, u in enumerate(["Name three colours.", "And three animals?"]):
        out, pos, *_ = turn(u, pos, 1 if i == 0 else 0, 0, 4)
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": T.decode(out)}]
    t3 = T.chat_turn("Thanks!", 0, 0); D.chat_turn("Thanks!", 0, 0)
    T.forward_batch(t3, pos); D.forward_batch(t3, pos)
    msgs.append({"role": "user", "content": "Thanks!"})
    ref = tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw), add_special_tokens=False)
    print("spec chat history == HF template: target", history(T) == ref, " draft", history(D) == ref)

if args.samples:
    # distribution of the first two generated tokens, spec (k=4) vs target-only, temp 0.8
    from collections import Counter
    p = "Pick a random fruit and name it."
    cs, ct = Counter(), Counter()
    for i in range(args.samples):
        out, *_ = turn(p, 0, 1, 0, 4, temp=0.8, rng=C.c_uint64(1000 + i)); cs[tuple(out[:3])] += 1
        out, *_ = turn(p, 0, 1, 0, 0, temp=0.8, rng=C.c_uint64(50000 + i)); ct[tuple(out[:3])] += 1
    keys = set(cs) | set(ct)
    tv = 0.5 * sum(abs(cs[k] - ct[k]) for k in keys) / args.samples
    print(f"temp 0.8, first 3 tokens over {args.samples} samples: total variation spec vs target {tv:.3f} "
          f"({len(keys)} distinct prefixes)")
