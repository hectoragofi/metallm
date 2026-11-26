"""Quality of one or more .BIN exports vs HF FP32 (teacher-forced on a fixed text + greedy chat).

  python tests/eval_quant.py --hf model-q35-0.8b build/A.BIN build/B.BIN ... [--ntok 384] [--gen 32]

Per file: mean logit cosine, top-1 agreement, KL(HF||C), perplexity (HF FP32 ppl printed
first), greedy tokens matching HF out of --gen, single-thread decode tok/s, file size.
"""
import argparse, inspect, json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, qeng

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True)
ap.add_argument("bins", nargs="+")
ap.add_argument("--ntok", type=int, default=384)
ap.add_argument("--gen", type=int, default=32)
ap.add_argument("--dtype", default="float32", help="HF reference dtype (bfloat16 for models too big for FP32 in RAM)")
ap.add_argument("--isa", type=int, default=1)
args = ap.parse_args()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(args.hf)
text = (inspect.getsource(json.decoder)[:3000] + "\n\n" +
        "The history of computing is a story of abstraction layers. Early machines were programmed by "
        "rewiring panels; later, stored programs let the same hardware run many tasks. 计算机的历史也是"
        "抽象层次不断提高的历史。Today a laptop runs billions of instructions per second.\n")
ids = tok.encode(text, add_special_tokens=False)[:args.ntok]
PROMPTS = ["What is the capital of France? Answer in one sentence.", "List the first ten prime numbers, separated by commas.",
           "Write a Python function that returns the factorial of n."]
kw = {"enable_thinking": False} if "enable_thinking" in (tok.chat_template or "") else {}
pids = [tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, **kw),
                   add_special_tokens=False) for p in PROMPTS]

hf = AutoModelForCausalLM.from_pretrained(args.hf, dtype=getattr(torch, args.dtype)).eval()
grefs = []
with torch.no_grad():
    ref = hf(torch.tensor([ids])).logits[0].float().numpy()
    for p in pids:
        o = hf(torch.tensor([p]), use_cache=True); past = o.past_key_values
        nxt = int(o.logits[0, -1].argmax()); gref = []
        for _ in range(args.gen):
            gref.append(nxt)
            o = hf(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
            past, nxt = o.past_key_values, int(o.logits[0, -1].argmax())
        grefs.append(gref)
del hf
V = ref.shape[1]
def lsm(x): x = x - x.max(-1, keepdims=True); return x - np.log(np.exp(x).sum(-1, keepdims=True))
lr = lsm(ref.astype(np.float64)); pr = np.exp(lr)
tgt = np.array(ids[1:])
ppl_hf = float(np.exp(-lr[np.arange(len(tgt)), tgt].mean()))
print(f"HF {args.dtype}: ppl {ppl_hf:.3f} on {len(ids)} tokens")
qeng.set_isa(args.isa)
for b in args.bins:
    e = qeng.Engine(b, 1024)
    lc = e.forward_batch(ids, 0, all_logits=True)[:, :V].astype(np.float64)
    cos = float(np.mean(np.sum(lc * ref, 1) / np.linalg.norm(lc, axis=1) / np.linalg.norm(ref, axis=1)))
    top1 = float(np.mean(lc.argmax(1) == ref.argmax(1)))
    lq = lsm(lc)
    kl = float(np.mean(np.sum(pr * (lr - lq), 1)))
    ppl = float(np.exp(-lq[np.arange(len(tgt)), tgt].mean()))
    matches, dt, ng = [], 0.0, 0
    for p, gref in zip(pids, grefs):
        lg = e.forward_batch(p, 0); pos = len(p); g = []
        t0 = time.thread_time()
        for _ in range(args.gen):
            n = int(lg[:V].argmax()); g.append(n); lg = e.forward(n, pos, 1); pos += 1
        dt += time.thread_time() - t0; ng += args.gen
        matches.append(next((i for i, (x, y) in enumerate(zip(g, gref)) if x != y), args.gen))
    size = sum(os.path.getsize(p) for p in [b] + [f"{os.path.splitext(b)[0]}.B{k:02d}" for k in range(1, 9)] if os.path.exists(p))
    print(f"{os.path.basename(b):22s} {size/2**20:8.1f} MiB  cos {cos:.5f}  top1 {top1:.3f}  KL {kl:.4f}  ppl {ppl:.3f} "
          f"(+{100*(ppl/ppl_hf-1):.2f}%)  greedy {'+'.join(map(str, matches))}/{len(pids)}x{args.gen}  decode {ng/dt:.1f} tok/s (1 thread, CPU time)")
    del e
