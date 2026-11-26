"""Broad tokenizer check: C eng_encode / eng_encode_special vs the HF tokenizer.

  python tests/test_tokenizer.py --bin build/QW35-08.BIN --hf model-q35-0.8b [--random 3000]
"""
import argparse, random, sys
sys.stdout.reconfigure(encoding="utf-8")
import qeng
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--bin", required=True)
ap.add_argument("--hf", required=True)
ap.add_argument("--random", type=int, default=3000)
args = ap.parse_args()
eng = qeng.Engine(args.bin, 256)
tok = AutoTokenizer.from_pretrained(args.hf)

fixed = [
    "Hello, world!", "  leading spaces and trailing  ", "I'm here, you're there, they'll go. DON'T", "x's y'S Z'Re 'll 'LL",
    "numbers 12345 and 3.14159!! 1,000,000 ¹²³ ½ ⅓ Ⅻ ٣٤٥ ১২৩", "line1\nline2\n\n  indented\n", "\r\n\r\nwin\r\nlines \r\n",
    "tabs\tand\t\tmore \t \t", "a  b   c    d     e", "     ", "\n\n\n", " \n \n ", "end with spaces   ", "  \t\n  x",
    "中文分词测试，你好世界！这是一个句子。", "日本語のテキスト、カタカナとひらがな。", "한국어 텍스트 테스트입니다", "混合 English 和中文 123",
    "emoji 😀😃😄 👨‍👩‍👧‍👦 🇳🇱🇩🇪 👍🏽 ❤️ ✨🎉", "math ∑∫∂√∞ ≤≥≠ α+β=γ", "arrows → ← ↑ ↓ ⇒", "currency €100 £50 ¥1000 ₿",
    "café naïve ünïcödé — “quotes” ‘single’ «guillemets»", "decomposed: café naïve Å 각",
    "combining à́̂ ȩ́ क़ क़", "Ǆ ǅ ǆ ﬁ ﬂ ℌ Ⅸ ①", "compat 豈更 Ω Å ̈́",
    "zero​width‌join‍ nbsp here　ideographic", "Ελληνικά κείμενο, Русский текст, עברית, العربية",
    "हिन्दी पाठ, தமிழ், ภาษาไทย", "code: for (int i=0;i<n;++i){x[i]+=y[i]*2;}", "def f(x):\n    return x**2 + 1\n",
    "<|im_start|>user\nhi<|im_end|>\n<think>\n\n</think>", "<tool_call>{\"a\": 1}</tool_call>", "<|fake|> <|im_start", "URL https://example.com/a?b=c&d=e#f",
    "!!! ??? ... --- === *** ### @@@ $$$ %%%", "'s'S'T't're'RE've'm'll'd", "ſ 'ſ 'ſt", "\u0000\u0001 ctrl \x7f", "\x1c\x1d\x1e\x1f seps",
    "The quick brown fox jumps over the lazy dog. " * 10, "a" * 3000, " " * 2000 + "x", "😀" * 300, "1234567890" * 50,
]
added = [t.content for t in tok.added_tokens_decoder.values()]
def check(s):
    if any(t in s for t in added):   # added tokens are only split out by the special-aware path
        a = eng.encode(s, special=True); b = tok.encode(s, add_special_tokens=False)
        return a == b, a, b
    a = eng.encode(s); b = tok.encode(s, add_special_tokens=False)
    return a == b, a, b
ok = bad = 0
for s in fixed:
    good, a, b = check(s)
    if good: ok += 1
    else:
        bad += 1; print("MISMATCH", repr(s[:80]), a[:20], b[:20])
# special-token aware path
for s in fixed[-14:-8]:
    a = eng.encode(s, special=True); b = tok.encode(s, add_special_tokens=False)
    if a == b: ok += 1
    else: bad += 1; print("SPECIAL MISMATCH", repr(s[:80]), a[:20], b[:20])

# random strings drawn from a mix of scripts, whitespace, digits, punctuation, combining marks, emoji
pools = [range(0x20, 0x7f), range(0xa0, 0x250), range(0x300, 0x370), range(0x370, 0x530), range(0x590, 0x700), range(0x900, 0xe00),
         range(0x1100, 0x1200), range(0x2000, 0x2070), range(0x2100, 0x2c00), range(0x3000, 0x3100), range(0x4e00, 0x4f00),
         range(0xac00, 0xac80), range(0xf900, 0xfb50), range(0xff00, 0xfff0), range(0x1f300, 0x1fa00), range(0x1d400, 0x1d800)]
ws = [" ", "  ", "\n", "\t", "\r\n", " \n", "　", " "]
rnd = random.Random(1)
rbad = 0
for i in range(args.random):
    parts = []
    for _ in range(rnd.randint(1, 30)):
        r = rnd.random()
        if r < 0.15: parts.append(rnd.choice(ws))
        elif r < 0.25: parts.append(str(rnd.randint(0, 99999)))
        elif r < 0.4: parts.append(rnd.choice(["the", "The", " and", "'s", "don't", "I'll", "x", "ab"]))
        else:
            pool = rnd.choice(pools)
            parts.append("".join(chr(rnd.choice(pool)) for _ in range(rnd.randint(1, 6))))
    s = "".join(parts)
    s = "".join(c for c in s if not (0xd800 <= ord(c) < 0xe000))
    good, a, b = check(s)
    if good: ok += 1
    else:
        rbad += 1; bad += 1
        if rbad <= 10:
            j = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print("RANDOM MISMATCH", ascii(s), len(a), len(b), j, a[j-2:j+4], b[j-2:j+4])
total = ok + bad
print(f"tokenizer: {ok}/{total} match ({len(fixed)} fixed + 6 special + {args.random} random)")
