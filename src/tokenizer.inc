// Included by engine.c: byte-level BPE tokenizer matching HF tokenizers (NFC normalizer,
// Qwen2/Qwen3.5 pre-tokenizer regex, added tokens), and the chat templates.

const u8* eng_token_bytes(Engine* e, int id, int* len) {
    if (id < 0 || id >= e->V) { *len = 0; return e->vocab_bytes; }
    u32 a = e->vocab_offs[id], b = e->vocab_offs[id + 1];
    *len = (int)(b - a);
    return e->vocab_bytes + a;
}

static void build_merges(Engine* e) {
    memset(e->mkeys, 0, 8ull * (e->mmask + 1));
    for (u32 r = 0; r < e->h->n_merges; r++) {
        u64 key = ((u64)e->merges[r * 3] << 32 | e->merges[r * 3 + 1]) | (1ull << 63);
        u32 i = (u32)(key * 0x9E3779B97F4A7C15ull >> 40) & e->mmask;
        while (e->mkeys[i] && e->mkeys[i] != key) i = (i + 1) & e->mmask;
        if (!e->mkeys[i]) { e->mkeys[i] = key; e->mvals[i] = r; }
    }
}
static int merge_rank(Engine* e, u32 a, u32 b) {
    u64 key = ((u64)a << 32 | b) | (1ull << 63);
    u32 i = (u32)(key * 0x9E3779B97F4A7C15ull >> 40) & e->mmask;
    while (e->mkeys[i]) { if (e->mkeys[i] == key) return (int)e->mvals[i]; i = (i + 1) & e->mmask; }
    return -1;
}

// BPE of one pre-tokenizer piece: repeatedly merge the lowest-rank adjacent pair (leftmost on ties).
static int bpe_chunk(Engine* e, const u8* s, int n, int* out, int max_out) {
    int* t = e->bpe_scratch; int* nx = t + MAX_CHUNK; int* rk = e->bpe_rank;
    for (int i = 0; i < n; i++) { t[i] = (int)e->bytetok[s[i]]; nx[i] = i + 1; }
    for (int i = 0; i + 1 < n; i++) rk[i] = merge_rank(e, t[i], t[i + 1]);
    if (n) rk[n - 1] = -1;
    for (;;) {
        int best = -1, bi = -1, prev_of_best = -1, pv = -1;
        for (int i = 0; i < n; i = nx[i]) {
            if (rk[i] >= 0 && (best < 0 || rk[i] < best)) { best = rk[i]; bi = i; prev_of_best = pv; }
            pv = i;
        }
        if (best < 0) break;
        int j = nx[bi];
        t[bi] = (int)e->merges[best * 3 + 2];
        nx[bi] = nx[j];
        rk[bi] = nx[bi] < n ? merge_rank(e, t[bi], t[nx[bi]]) : -1;
        if (prev_of_best >= 0) rk[prev_of_best] = merge_rank(e, t[prev_of_best], t[bi]);
    }
    int w = 0; for (int i = 0; i < n && w < max_out; i = nx[i]) out[w++] = t[i];
    return w;
}

// ---- UTF-8 / Unicode ----
static int utf8_dec(const u8* s, int n, int* adv) {
    u8 c = s[0];
    if (c < 0x80) { *adv = 1; return c; }
    int len = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 0;
    if (!len || len > n) { *adv = 1; return 0xFFFD; }
    u32 cp = c & (0x7F >> len);
    for (int i = 1; i < len; i++) { if ((s[i] & 0xC0) != 0x80) { *adv = 1; return 0xFFFD; } cp = cp << 6 | (s[i] & 0x3F); }
    *adv = len;
    return (int)cp;
}
static int utf8_enc(u32 c, u8* o) {
    if (c < 0x80) { o[0] = c; return 1; }
    if (c < 0x800) { o[0] = 0xC0 | c >> 6; o[1] = 0x80 | (c & 63); return 2; }
    if (c < 0x10000) { o[0] = 0xE0 | c >> 12; o[1] = 0x80 | (c >> 6 & 63); o[2] = 0x80 | (c & 63); return 3; }
    o[0] = 0xF0 | c >> 18; o[1] = 0x80 | (c >> 12 & 63); o[2] = 0x80 | (c >> 6 & 63); o[3] = 0x80 | (c & 63); return 4;
}
enum { C_OTHER = 0, C_LET = 1, C_NUM = 2, C_WS = 3 };
static int uclass(Engine* e, u32 c) {
    int lo = 0, hi = e->n_uranges - 1;   // ranges sorted by start: entry = start << 8 | class
    while (lo < hi) { int mid = (lo + hi + 1) >> 1; if ((e->uranges[mid] >> 8) <= c) lo = mid; else hi = mid - 1; }
    return e->uranges[lo] & 0xFF;
}

// ---- NFC ----
static int bsearch_u32(const u32* tab, int n, int stride, u32 key) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) { int mid = (lo + hi) >> 1; u32 v = tab[mid * stride]; if (v == key) return mid; if (v < key) lo = mid + 1; else hi = mid - 1; }
    return -1;
}
static int ccc_of(Engine* e, u32 c) { if (c < 0x300) return 0; int i = bsearch_u32(e->ccc, e->n_ccc, 2, c); return i < 0 ? 0 : (int)e->ccc[i * 2 + 1]; }
static u32 compose(Engine* e, u32 a, u32 b) {
    if (a >= 0x1100 && a < 0x1113 && b >= 0x1161 && b < 0x1176) return 0xAC00 + ((a - 0x1100) * 21 + (b - 0x1161)) * 28;
    if (a >= 0xAC00 && a < 0xD7A4 && (a - 0xAC00) % 28 == 0 && b > 0x11A7 && b < 0x11C3) return a + (b - 0x11A7);
    u64 key = ((u64)a << 21) | b;
    int lo = 0, hi = e->n_comp - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1; const u32* r = e->comp + mid * 3; u64 k = (u64)r[1] << 32 | r[0];
        if (k == key) return r[2];
        if (k < key) lo = mid + 1; else hi = mid - 1;
    }
    return 0;
}
// NFC of s[0..n) into out (capacity cap bytes); returns output length.
static int nfc(Engine* e, const u8* s, int n, u8* out, int cap) {
    u32* cp = e->cps; int m = 0, maxcp = 4 * NFC_BLOCK - 8, adv;
    for (int i = 0; i < n && m < maxcp; i += adv) {
        u32 c = utf8_dec(s + i, n - i, &adv);
        if (c >= 0xAC00 && c < 0xD7A4) {
            u32 si = c - 0xAC00; cp[m++] = 0x1100 + si / 588; cp[m++] = 0x1161 + (si % 588) / 28;
            if (si % 28) cp[m++] = 0x11A7 + si % 28;
            continue;
        }
        int k = c >= 0xC0 ? bsearch_u32(e->decomp, e->n_decomp, 3, c) : -1;
        if (k >= 0) { const u32* d = e->decomp + k * 3; for (u32 j = 0; j < d[2] && m < maxcp; j++) cp[m++] = e->decomp_data[d[1] + j]; }
        else cp[m++] = c;
    }
    for (int i = 1; i < m; i++) {   // canonical ordering (stable insertion sort of non-starter runs)
        int cc = ccc_of(e, cp[i]);
        if (!cc) continue;
        int j = i; u32 c = cp[i];
        while (j > 0) { int pc = ccc_of(e, cp[j - 1]); if (pc <= cc) break; cp[j] = cp[j - 1]; j--; }
        cp[j] = c;
    }
    if (m) {   // canonical composition
        int st = 0, last = ccc_of(e, cp[0]) ? 256 : 0, w = 1;
        for (int i = 1; i < m; i++) {
            u32 c = cp[i]; int cc = ccc_of(e, c);
            u32 comp = (last < cc || last == 0) ? compose(e, cp[st], c) : 0;
            if (comp && (last < cc || (last == 0 && w - 1 == st))) { cp[st] = comp; continue; }
            if (cc == 0) st = w;
            last = cc; cp[w++] = c;
        }
        m = w;
    }
    int o = 0;
    for (int i = 0; i < m && o + 4 <= cap; i++) o += utf8_enc(cp[i], out + o);
    return o;
}

// ---- pre-tokenizer ----
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// (Qwen3.5: \p{L} -> [\p{L}\p{M}] in the letter run and the punctuation class; the class table
// from export.py already puts \p{M} in C_LET for it.) Works on code points with classes.
typedef struct { const u32* cp; const u8* cl; int n; } Pieces;
static int is_nl(u32 c) { return c == '\n' || c == '\r'; }
static u32 lowc(u32 c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c == 0x17F ? 's' : c; }
static int pretok_len(const Pieces* P, int i) {
    const u32* cp = P->cp; const u8* cl = P->cl; int n = P->n;
    if (cp[i] == '\'' && i + 1 < n) {
        u32 a = lowc(cp[i + 1]), b = i + 2 < n ? lowc(cp[i + 2]) : 0;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    }
    {
        int j = i;
        if (cl[j] != C_LET && cl[j] != C_NUM && !is_nl(cp[j]) && j + 1 < n && cl[j + 1] == C_LET) j++;
        if (cl[j] == C_LET) { while (j < n && cl[j] == C_LET) j++; return j - i; }
    }
    if (cl[i] == C_NUM) return 1;
    {
        int j = i;
        if (cp[j] == ' ' && j + 1 < n && cl[j + 1] == C_OTHER) j++;
        if (cl[j] == C_OTHER) {
            while (j < n && cl[j] == C_OTHER) j++;
            while (j < n && is_nl(cp[j])) j++;
            return j - i;
        }
    }
    if (cl[i] == C_WS) {
        int j = i; while (j < n && cl[j] == C_WS) j++;
        int last_nl = -1; for (int k = i; k < j; k++) if (is_nl(cp[k])) last_nl = k;
        if (last_nl >= 0) return last_nl + 1 - i;
        if (j == n) return j - i;
        if (j - i > 1) return j - i - 1;
        return 1;
    }
    return 1;
}

static int encode_block(Engine* e, const u8* text, int len, int* out, int max_out) {
    // decode to code points + classes (cps buffer: [cp][byte offset]); class bytes in nfcbuf tail
    u32* cp = e->cps; u32* bo = e->cps + 2 * NFC_BLOCK; u8* cl = e->nfcbuf + 3 * NFC_BLOCK;
    int m = 0, adv;
    for (int i = 0; i < len && m < 2 * NFC_BLOCK; i += adv) {
        u32 c = utf8_dec(text + i, len - i, &adv);
        cp[m] = c; bo[m] = i;
        cl[m] = c < 128 ? e->asc[c] : (u8)uclass(e, c);
        m++;
    }
    Pieces P = {cp, cl, m};
    int w = 0;
    for (int i = 0; i < m && w < max_out;) {
        int l = pretok_len(&P, i);
        int b0 = bo[i], b1 = i + l < m ? (int)bo[i + l] : len;
        for (int s = b0; s < b1 && w < max_out; s += MAX_CHUNK) {
            int sl = b1 - s < MAX_CHUNK ? b1 - s : MAX_CHUNK;
            w += bpe_chunk(e, text + s, sl, out + w, max_out - w);
        }
        i += l;
    }
    return w;
}

// split point <= lim where a block boundary cannot change pre-tokenization or NFC: after "\n"
// and before an ASCII letter/digit.
static int safe_split(const u8* s, int n, int lim) {
    for (int i = lim; i > lim / 2; i--) {
        u8 c = s[i];
        if (s[i - 1] == '\n' && ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))) return i;
    }
    int i = lim; while (i > 0 && (s[i] & 0xC0) == 0x80) i--;   // fallback: code point boundary
    return i > 0 ? i : lim;
    (void)n;
}

int eng_encode(Engine* e, const u8* text, int len, int* out, int max_out) {
    int w = 0;
    while (len > 0 && w < max_out) {
        int blk = len;
        if (blk > NFC_BLOCK / 2) blk = safe_split(text, len, NFC_BLOCK / 2);
        const u8* t = text; int tl = blk;
        int ascii = 1; for (int i = 0; i < blk; i++) if (text[i] >= 0x80) { ascii = 0; break; }
        if (!ascii && e->n_ccc >= 0) { tl = nfc(e, text, blk, e->nfcbuf, 3 * NFC_BLOCK); t = e->nfcbuf; }
        w += encode_block(e, t, tl, out + w, max_out - w);
        text += blk; len -= blk;
    }
    return w;
}

// added tokens (special or not) written as text become their ids, the rest is BPE'd
int eng_encode_special(Engine* e, const u8* text, int len, int* out, int max_out) {
    int w = 0, start = 0;
    for (int i = 0; i < len && w < max_out;) {
        int best = -1, bl = 0;
        for (int a = 0; a < e->n_added; a++) {
            int id = (int)e->added[a], l; const u8* b = eng_token_bytes(e, id, &l);
            if (l > bl && l <= len - i && b[0] == text[i] && !memcmp(b, text + i, l)) { best = id; bl = l; }
        }
        if (best < 0) { i++; continue; }
        if (i > start) w += eng_encode(e, text + start, i - start, out + w, max_out - w);
        if (w < max_out) out[w++] = best;
        i += bl; start = i;
    }
    if (start < len && w < max_out) w += eng_encode(e, text + start, len - start, out + w, max_out - w);
    return w;
}

// ---- chat template ----
// Renders the HF chat template for one new user turn (plus the reply-closing tokens when !first)
// and tokenizes it the way HF does (whole string, added tokens split out).
int eng_chat_turn(Engine* e, const u8* sys, int sys_len, const u8* msg, int len, int first, int think, int* out, int max) {
    int st = e->chat_style, trim = st == CS_QWEN35 || st == CS_QWEN38;
    SB sb = {e->textbuf, 0, e->textcap};
    if (!first) sb_str(&sb, "<|im_end|>\n");
    if (first) {
        const u8* s = sys; int sl = sys ? sys_len : 0;
        if (trim) strip(&s, &sl);
        int has = sys && sys_len > 0;
        const u8* ts = e->think_sys; int tsl = (st == CS_QWEN38 && think) ? e->think_sys_len : 0;
        if (st == CS_QWEN38) {
            if (has && sl) { sb_str(&sb, "<|im_start|>system\n"); if (tsl) { sb_put(&sb, ts, tsl); sb_str(&sb, "\n\n"); } sb_put(&sb, s, sl); sb_str(&sb, "<|im_end|>\n"); }
            else if (tsl) { sb_str(&sb, "<|im_start|>system\n"); sb_put(&sb, ts, tsl); sb_str(&sb, "<|im_end|>\n"); }
        } else if (has) { sb_str(&sb, "<|im_start|>system\n"); sb_put(&sb, s, sl); sb_str(&sb, "<|im_end|>\n"); }
    }
    const u8* m = msg; int ml = len;
    if (trim) strip(&m, &ml);
    sb_str(&sb, "<|im_start|>user\n"); sb_put(&sb, m, ml); sb_str(&sb, "<|im_end|>\n");
    int gen_at = sb.n;
    sb_str(&sb, "<|im_start|>assistant\n");
    if (st == CS_NOTHINK || !think) sb_str(&sb, "<think>\n\n</think>\n\n");
    else if (st != CS_QWEN3) sb_str(&sb, "<think>\n");
    int n = eng_encode_special(e, sb.p, sb.n, out, max);
    // token index just after the generation prompt's <|im_start|>
    int asst = n; (void)gen_at;
    for (int i = n - 1; i >= 0; i--) if (out[i] == (int)e->h->tok_im_start) { asst = i + 1; break; }
    e->pend = 1; e->pend_first = first; e->pend_asst = asst; e->pend_content = n;
    return n;
}
