"""Chat template check: the engine's token history after a scripted conversation (fed the way
efi_main feeds it: eng_chat_turn tokens, then the reply tokens, the sampled <|im_end|> not fed)
must equal HF apply_chat_template of the same messages, tokenized.

  python tests/test_chat.py --bin build/QW35-08.BIN --hf model-q35-0.8b [--logits]
"""
import argparse, ctypes as C, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, qeng
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--bin", required=True)
ap.add_argument("--hf", required=True)
ap.add_argument("--logits", action="store_true", help="also compare final logits with a fresh engine fed HF's tokens")
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(args.hf)
eng = qeng.Engine(args.bin, 2048)
qeng.dll.eng_history.restype = C.c_int
qeng.dll.eng_history.argtypes = [C.c_void_p, C.POINTER(C.POINTER(C.c_int))]
def history(e):
    p = C.POINTER(C.c_int)(); n = qeng.dll.eng_history(e.e, C.byref(p)); return list(p[:n])
tmpl = tok.chat_template if isinstance(tok.chat_template, str) else ""
has_think = "enable_thinking" in tmpl
separate = "preserve_thinking" in tmpl

def hf_ids(msgs, think):
    kw = {"enable_thinking": bool(think)} if has_think else {}
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    return tok.encode(s, add_special_tokens=False), s

def gen_prompt_tail(think):
    """What the generation prompt already contains after 'assistant\\n' (the reply continues it)."""
    _, s = hf_ids([{"role": "user", "content": "x"}], think)
    return s.split("<|im_start|>assistant\n")[-1]

def reply_tokens(reasoning, answer, think):
    """Tokens the model would generate after the generation prompt, and the assistant message
    content an HF user would append (the decoded generated text, as from model.generate)."""
    tail = gen_prompt_tail(think)
    full = f"<think>\n{reasoning}\n</think>\n\n{answer}" if think and has_think else tail + answer
    assert full.startswith(tail), (tail, full)
    cont = full[len(tail):]
    return tok.encode(cont, add_special_tokens=False), cont

def run(convo, think, sys_prompt=None, retry=None):
    """convo: list of (user, reasoning, answer); the last answer is not fed (we stop at the prompt)."""
    pos, first, msgs = 0, 1, []
    if sys_prompt is not None: msgs.append({"role": "system", "content": sys_prompt})
    turn_pos = []
    for i, (u, r, a) in enumerate(convo):
        t = eng.chat_turn(u, first, think, sys_prompt if first else None)
        turn_pos.append((pos, first))
        eng.forward_batch(t, pos); pos += len(t); first = 0
        msgs.append({"role": "user", "content": u})
        if i < len(convo) - 1:
            rt, full = reply_tokens(r, a, think)
            for x in rt: eng.forward(x, pos, 0); pos += 1
            if separate:   # templates that only read reasoning_content (Bonsai-2-27B): what a reasoning parser yields
                r_, _, a_ = full.partition("</think>") if "</think>" in full else ("", "", full)
                msgs.append({"role": "assistant", "content": a_, "reasoning_content": r_.replace("<think>", "", 1)})
            else:
                msgs.append({"role": "assistant", "content": full})
    if retry is not None:   # /retry of the last turn with a different message
        p0, f0 = turn_pos[-1]
        t = eng.chat_turn(retry, f0, think, sys_prompt if f0 else None)
        eng.forward_batch(t, p0); pos = p0 + len(t)
        msgs[-1] = {"role": "user", "content": retry}
    ref, s = hf_ids(msgs, think)
    got = history(eng)
    return got == ref, got, ref, msgs

convs = [
    [("hi", "", "")],
    [("  spaced message \n", "", "")],
    [("what's 2+2?", "Simple sum.", "4"), ("and 3+3?", "", "")],
    [("tell me a joke", "User wants humour.\nPick one.", "  Why did the chicken cross the road?  \n"), ("another", "", "")],
    [("中文问题？", "思考一下。", "这是答案。"), ("😀 emoji turn", "hmm", "ok 👍"), ("third\n\nturn", "", "")],
    [("a", "r1", "b"), ("c", "r2", "\n\nd with leading newlines"), ("e", "r3", "f</think>tricky"), ("g", "", "")],
]
ok = total = 0
for think in ((0, 1) if has_think else (0,)):
    for sp in (None, "You are a helpful assistant.", "  padded system  "):
        for c in convs:
            good, got, ref, msgs = run(c, think, sp)
            total += 1; ok += good
            if not good:
                j = next((j for j, (x, y) in enumerate(zip(got, ref)) if x != y), min(len(got), len(ref)))
                print(f"MISMATCH think={think} sys={sp!r} turns={len(c)} at {j}:\n  C : {tok.decode(got[max(0,j-8):j+12])!r}\n  HF: {tok.decode(ref[max(0,j-8):j+12])!r}")
    # retry of a later turn and of the first turn
    for c, rmsg in ((convs[2], "and 4+4?"), (convs[0], "hello again")):
        good, got, ref, _ = run(c, think, "sys", retry=rmsg)
        total += 1; ok += good
        if not good: print(f"RETRY MISMATCH think={think}:\n  C : {tok.decode(got)!r}\n  HF: {tok.decode(ref)!r}")
print(f"chat template: {ok}/{total} conversations match HF exactly")

if args.logits:
    good, got, ref, _ = run(convs[4], 1 if has_think else 0, "sys")
    e2 = qeng.Engine(args.bin, 2048, model=eng.model)
    a = e2.forward_batch(ref, 0)
    # replay our conversation on the first engine and take its last logits
    pos, first = 0, 1
    for i, (u, r, an) in enumerate(convs[4]):
        t = eng.chat_turn(u, first, 1 if has_think else 0, "sys" if first else None)
        b = eng.forward_batch(t, pos); pos += len(t); first = 0
        if i < len(convs[4]) - 1:
            rt, _ = reply_tokens(r, an, 1 if has_think else 0)
            for x in rt: eng.forward(x, pos, 0); pos += 1
    print(f"final logits vs fresh HF-token prefill: max |diff| {np.abs(a - b).max():.3e}")
