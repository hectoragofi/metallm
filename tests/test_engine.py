"""Check the C engine (build/qwen.dll) against Hugging Face transformers in FP32.

  python tests/test_engine.py [--bin build/QWEN3-06.BIN] [--hf model] [--isa 1] [--n 32]
"""
import argparse, sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import qeng

ap = argparse.ArgumentParser()
ap.add_argument("--bin", default="build/QWEN3-06.BIN")
ap.add_argument("--hf", default="model")
ap.add_argument("--isa", type=int, default=1)
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--prompt", default="Write a haiku about old computers.")
args = ap.parse_args()

eng = qeng.Engine(args.bin, 1024)
print(f"engine ok, arena {eng.arena_mb:.0f} MiB, isa {qeng.set_isa(args.isa)}")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tok = AutoTokenizer.from_pretrained(args.hf)

# ---- tokenizer (quick set; test_tokenizer.py has the broad one) ----
tests = ["Hello, world!", "what is the capital of france?", "  leading spaces and trailing  ",
         "I'm here, you're there, they'll go. DON'T", "numbers 12345 and 3.14159!!", "line1\nline2\n\n  indented\n",
         "tabs\tand\t\tmore", "emoji? (none) ... --- ===> <|fake|>", "def f(x):\n    return x**2 + 1\n",
         "MixedCASE wordsWith Caps", "a  b   c    d", "end with newline\n", "!!! ??? ...", "x's y'S Z'Re",
         "The quick brown fox jumps over the lazy dog. " * 3, "café naïve ünïcödé — “quotes”"]
bad = 0
for s in tests:
    a, b = eng.encode(s), tok.encode(s, add_special_tokens=False)
    if a != b: bad += 1; print("TOKENIZER MISMATCH", repr(s), a, b)
    assert eng.decode(a) == s
print(f"tokenizer: {len(tests)-bad}/{len(tests)} match")

# ---- forward vs HF FP32 ----
msgs = [{"role": "user", "content": args.prompt}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok.encode(prompt, add_special_tokens=False)
print("prompt tokens:", len(ids))
hf = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.float32).eval()

N_GEN = args.n
with torch.no_grad():
    o = hf(torch.tensor([ids]), use_cache=True)
    ref_logits = o.logits[0, -1].float().numpy()
    ref, past, nxt = [], o.past_key_values, int(o.logits[0, -1].argmax())
    for _ in range(N_GEN):   # greedy without stopping at EOS
        ref.append(nxt)
        o = hf(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
        past, nxt = o.past_key_values, int(o.logits[0, -1].argmax())

t0 = time.time()
for p, t in enumerate(ids[:-1]): eng.forward(t, p, 0)
c_logits = eng.forward(ids[-1], len(ids) - 1, 1)
t_pre = time.time() - t0
V = min(len(ref_logits), len(c_logits))
c_logits, ref_logits = c_logits[:V], ref_logits[:V]
cos = float(np.dot(c_logits, ref_logits) / np.linalg.norm(c_logits) / np.linalg.norm(ref_logits))
print(f"last-token logits cosine vs FP32: {cos:.5f}  top1 C={c_logits.argmax()} HF={ref_logits.argmax()}")

t0 = time.time()
lb = eng.forward_batch(ids, 0)
t_batch = time.time() - t0
print(f"batched prefill: max |diff| vs sequential {np.abs(lb[:V] - c_logits).max():.2e}")

gen = []; pos = len(ids); nxt = int(c_logits.argmax())
t0 = time.time()
for _ in range(N_GEN):
    gen.append(nxt)
    lg = eng.forward(nxt, pos, 1); pos += 1
    nxt = int(lg.argmax())
dt = time.time() - t0
match = sum(1 for a, b in zip(gen, ref) if a == b)
first_div = next((i for i, (a, b) in enumerate(zip(gen, ref)) if a != b), None)
print(f"greedy tokens matching HF: {match}/{N_GEN} (first divergence: {first_div})")
print("C :", repr(eng.decode(gen)))
print("HF:", repr(tok.decode(ref)))
print(f"speed (1 thread, isa {args.isa}): prefill {len(ids)/t_pre:.1f} tok/s, batched prefill {len(ids)/t_batch:.1f} tok/s, decode {N_GEN/dt:.1f} tok/s")
