// Included by engine.c: logical/physical positions, rewinds, chat-history rewrite,
// speculative decoding, sampler.

static void rewind_to(Engine* e, int phys) {
    if (phys >= e->n_past) return;
    for (int s = 0; s < 2; s++) if (e->ckpt_pos[s] > phys) e->ckpt_pos[s] = -1;
    if (e->asst_pos > phys) e->asst_pos = -1;
    if (!e->state_floats) { e->n_past = phys; return; }
    int best = -1;
    for (int s = 0; s < 2; s++) if (e->ckpt_pos[s] >= 0 && (best < 0 || e->ckpt_pos[s] > e->ckpt_pos[best])) best = s;
    int from = 0;
    if (best >= 0) { restore(e, best); from = e->ckpt_pos[best]; } else reset_state(e);
    if (phys > from) forward_phys(e, e->hist + from, phys - from, from, 0, 0);
    e->n_past = phys;
}

static int decode_text(Engine* e, const int* toks, int n, u8* buf, int cap) {
    int w = 0;
    for (int i = 0; i < n; i++) {
        int len; const u8* b = eng_token_bytes(e, toks[i], &len);
        if (w + len > cap) break;
        memcpy(buf + w, b, len); w += len;
    }
    return w;
}
static int find(const u8* s, int n, const char* pat, int last) {
    int pl = 0; while (pat[pl]) pl++;
    int r = -1;
    for (int i = 0; i + pl <= n; i++) if (!memcmp(s + i, pat, pl)) { r = i; if (!last) return r; }
    return r;
}
static int is_pyspace(u32 c) {   // Python str.isspace (Jinja's trim filter is str.strip)
    return (c >= 9 && c <= 13) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}
static void strip(const u8** s, int* n) {
    const u8* p = *s; int len = *n, adv;
    while (len > 0) { u32 c = utf8_dec(p, len, &adv); if (!is_pyspace(c)) break; p += adv; len -= adv; }
    while (len > 0) {
        int j = len - 1; while (j > 0 && (p[j] & 0xC0) == 0x80) j--;
        u32 c = utf8_dec(p + j, len - j, &adv);
        if (j + adv != len || !is_pyspace(c)) break;
        len = j;
    }
    *s = p; *n = len;
}
static void lstrip_nl(const u8** s, int* n) { while (*n > 0 && **s == '\n') { (*s)++; (*n)--; } }

typedef struct { u8* p; int n, cap; } SB;
static void sb_put(SB* b, const void* s, int n) { if (n < 0) n = 0; if (b->n + n > b->cap) n = b->cap - b->n; memcpy(b->p + b->n, s, n); b->n += n; }
static void sb_str(SB* b, const char* s) { int n = 0; while (s[n]) n++; sb_put(b, s, n); }

// Rebuild the previous reply the way the HF template renders an earlier assistant turn.
static void chat_rewrite(Engine* e) {
    int A = e->asst_pos, C = e->content_pos, end = e->n_past;
    if (A < 0 || C < A || end < C) return;
    while (end > C && e->hist[end - 1] == (int)e->h->tok_im_end) end--;
    int half = e->textcap / 2;
    u8* raw = e->textbuf; int rn = decode_text(e, e->hist + C, end - C, raw, half);
    SB sb = {e->textbuf + half, 0, e->textcap - half};
    sb_str(&sb, "assistant\n");
    const u8* c = raw; int cn = rn;
    if (e->chat_style == CS_QWEN38) {     // preserve_thinking: reasoning kept, both parts trimmed
        const u8* r = raw; int rl = 0;
        int k = find(raw, rn, "</think>", 0);
        if (k >= 0) { rl = k; c = raw + k + 8; cn = rn - k - 8; }
        strip(&r, &rl); strip(&c, &cn);
        sb_str(&sb, "<think>\n"); sb_put(&sb, r, rl); sb_str(&sb, "\n</think>\n\n"); sb_put(&sb, c, cn);
    } else {                              // reasoning dropped: content after the last </think>
        if (e->chat_style == CS_QWEN35) strip(&c, &cn);
        int k = find(c, cn, "</think>", 1);
        if (k >= 0) { c += k + 8; cn -= k + 8; lstrip_nl(&c, &cn); }
        sb_put(&sb, c, cn);
    }
    int cap = e->max_seq;
    int* tk = e->tokbuf;
    int nt = eng_encode_special(e, sb.p, sb.n, tk, cap);
    if (nt >= cap || sb.n >= sb.cap) return;   // too long to rewrite safely: keep history as fed
    int newend = A + nt;
    if (newend > e->last_logical) return;      // would need a negative position offset
    int cp = 0; while (cp < nt && A + cp < e->n_past && e->hist[A + cp] == tk[cp]) cp++;
    if (!(cp == nt && newend == e->n_past)) {
        rewind_to(e, A + cp);
        if (nt > cp) forward_phys(e, tk + cp, nt - cp, A + cp, 0, 0);
    }
    e->pos_off = e->last_logical - e->n_past;
}

static void start_turn(Engine* e, int phys) {
    e->pend = 0;
    e->asst_pos = phys + e->pend_asst; e->content_pos = phys + e->pend_content;
    e->ckpt_pos[1] = -1;
    snapshot(e, 0, phys);
}

static int map_pos(Engine* e, int pos) {
    if (pos <= 0) {
        if (e->n_past == 0 && e->pos_off == 0 && !e->pend) return 0;   // already at a fresh start
        e->pos_off = 0; reset_state(e);
        e->ckpt_pos[0] = e->ckpt_pos[1] = e->ckpt_pos[2] = -1; e->asst_pos = -1;
        if (e->pend) start_turn(e, 0);
        return 0;
    }
    int phys = pos - e->pos_off;
    if (phys < 0) phys = 0;
    if (phys < e->n_past) rewind_to(e, phys);   // also drops tokens fed past the reply's end (spec decoding)
    if (e->pend && !e->pend_first && e->asst_pos >= 0) {   // next turn: rewrite the reply that ends here
        e->last_logical = pos;
        chat_rewrite(e);
        phys = pos - e->pos_off;
    }
    if (e->pend) start_turn(e, phys);
    return phys;
}

float* eng_forward_batch(Engine* e, const int* tokens, int n, int pos, float* logits) {
    if (n <= 0) return 0;
    int phys = map_pos(e, pos);
    if (phys + n > e->max_seq) { n = e->max_seq - phys; if (n <= 0) return logits ? logits : e->logits; }
    forward_phys(e, tokens, n, phys, logits ? 2 : 1, logits ? logits : e->logits);
    e->last_logical = pos + n;
    return logits ? logits : e->logits;
}

float* eng_forward(Engine* e, int token, int pos, int want_logits) {
    int phys = map_pos(e, pos);
    if (phys >= e->max_seq) return e->logits;
    forward_phys(e, &token, 1, phys, want_logits ? 1 : 0, e->logits);
    e->last_logical = pos + 1;
    return want_logits ? e->logits : NULL;
}

// ---------------- speculative decoding ----------------
void eng_record(Engine* e, int on) {
    if (on) { snapshot(e, 2, e->n_past); e->rec_on = 1; e->rec_n = 0; }
    else e->rec_on = 0;
}
void eng_rollback(Engine* e, int pos) {
    int phys = pos - e->pos_off, start = e->ckpt_pos[2], keep = phys - start;
    int logged = e->rec_n;
    e->rec_on = 0;
    if (phys >= e->n_past) return;
    if (start < 0 || keep < 0 || keep > logged) { rewind_to(e, phys); e->last_logical = pos; return; }
    if (e->state_floats) {
        restore(e, 2);
        int cd = e->conv_dim, K = e->conv_k, nv = e->nv, dk = e->dk, dv = e->dv;
        for (int l = 0; l < e->n_lin; l++) {
            float* cst = e->conv_state + (usize)l * (K - 1) * cd;
            float* S0 = e->rec_state + (usize)l * nv * dk * dv;
            for (int t = 0; t < keep; t++) {
                memmove(cst, cst + cd, 4ull * (K - 2) * cd);
                memcpy(cst + (usize)(K - 2) * cd, e->conv_log + ((usize)t * e->n_lin + l) * cd, 4ull * cd);
                const float* lg = e->rec_log + ((usize)t * e->n_lin + l) * nv * (1 + dk + dv);
                for (int h = 0; h < nv; h++) {
                    const float* rl = lg + (usize)h * (1 + dk + dv); float decay = rl[0];
                    const float* k = rl + 1; const float* dl = rl + 1 + dk;
                    float* S = S0 + (usize)h * dk * dv;
                    for (int i = 0; i < dk; i++) { float* Si = S + (usize)i * dv; for (int j = 0; j < dv; j++) Si[j] = decay * Si[j] + k[i] * dl[j]; }
                }
            }
        }
    }
    if (e->asst_pos > phys) e->asst_pos = -1;
    for (int s = 0; s < 2; s++) if (e->ckpt_pos[s] > phys) e->ckpt_pos[s] = -1;
    e->n_past = phys;
    e->last_logical = pos;
}

enum { KMAX = 64 };
static int dist(const float* lg, int V, float temp, float top_p, int top_k, int* idx, float* p) {
    if (top_k > KMAX || top_k <= 0) top_k = KMAX;
    float val[KMAX]; int n = 0;
    for (int i = 0; i < V; i++) {
        float v = lg[i];
        if (n == top_k && v <= val[n - 1]) continue;
        int j = n < top_k ? n++ : n - 1;
        while (j > 0 && val[j - 1] < v) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--; }
        val[j] = v; idx[j] = i;
    }
    float sum = 0;
    for (int i = 0; i < n; i++) { p[i] = m_expf((val[i] - val[0]) / temp); sum += p[i]; }
    float cum = 0; int m = n;
    for (int i = 0; i < n; i++) { cum += p[i] / sum; if (cum >= top_p) { m = i + 1; break; } }
    float tot = 0; for (int i = 0; i < m; i++) tot += p[i];
    for (int i = 0; i < m; i++) p[i] /= tot;
    return m;
}
static u32 rng_u32(u64* s) { u64 x = *s; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; *s = x; return (u32)((x * 0x2545F4914F6CDD1Dull) >> 32); }
static float rng_f(u64* s) { return (float)(rng_u32(s) >> 8) / 16777216.0f; }
static int argmax(const float* lg, int V) { int b = 0; for (int i = 1; i < V; i++) if (lg[i] > lg[b]) b = i; return b; }
static int pick(const int* idx, const float* p, int m, u64* rng) {
    float r = rng_f(rng);
    for (int i = 0; i < m; i++) { r -= p[i]; if (r < 0) return idx[i]; }
    return idx[m - 1];
}

int eng_sample(Engine* e, const float* lg, float temp, float top_p, int top_k, u64* rng) {
    if (temp <= 0 || top_k == 1) return argmax(lg, e->V);
    int idx[KMAX]; float p[KMAX];
    int m = dist(lg, e->V, temp, top_p, top_k, idx, p);
    return pick(idx, p, m, rng);
}

static float prob_of(const int* idx, const float* p, int m, int t) { for (int i = 0; i < m; i++) if (idx[i] == t) return p[i]; return 0; }

int eng_spec_step(Engine* T, Engine* D, int* pos, int cur, int k, float temp, float top_p, int top_k, u64* rng, int* out) {
    int p0 = *pos, V = T->V, greedy = temp <= 0 || top_k == 1;
    if (k > REC_MAX - 1) k = REC_MAX - 1;
    if (p0 + k + 1 > T->max_seq) k = T->max_seq - p0 - 1;
    if (p0 + k + 1 > D->max_seq) k = D->max_seq - p0 - 1;
    if (k < 1) {
        float* lg = eng_forward(T, cur, p0, 1);
        out[0] = eng_sample(T, lg, temp, top_p, top_k, rng); *pos = p0 + 1; return 1;
    }
    int d[REC_MAX], qi[REC_MAX][KMAX], qm[REC_MAX]; float qp[REC_MAX][KMAX];
    // draft: feed cur, d0 .. d_{k-2}; k drafted tokens
    map_pos(D, p0); eng_record(D, 1);
    float* lg = eng_forward(D, cur, p0, 1);
    for (int i = 0; i < k; i++) {
        if (greedy) d[i] = argmax(lg, D->V);
        else { qm[i] = dist(lg, D->V, temp, top_p, top_k, qi[i], qp[i]); d[i] = pick(qi[i], qp[i], qm[i], rng); }
        if (i < k - 1) lg = eng_forward(D, d[i], p0 + 1 + i, 1);
    }
    // verify: one batched target pass over [cur, d0 .. d_{k-1}]
    int tk[REC_MAX + 1]; tk[0] = cur; for (int i = 0; i < k; i++) tk[i + 1] = d[i];
    map_pos(T, p0); eng_record(T, 1);
    float* L = eng_forward_batch(T, tk, k + 1, p0, T->logits);
    int n = 0, acc = 0, pi[KMAX]; float pp[KMAX];
    for (; acc < k; acc++) {
        const float* P = L + (usize)acc * V;
        if (greedy) {
            int t = argmax(P, V);
            out[n++] = t;
            if (t != d[acc]) break;
        } else {
            int pm = dist(P, V, temp, top_p, top_k, pi, pp);
            float pd = prob_of(pi, pp, pm, d[acc]), qd = prob_of(qi[acc], qp[acc], qm[acc], d[acc]);
            if (qd > 0 && rng_f(rng) * qd < pd) { out[n++] = d[acc]; continue; }
            float r[KMAX], tot = 0;   // residual max(0, p - q), renormalised
            for (int j = 0; j < pm; j++) { float v = pp[j] - prob_of(qi[acc], qp[acc], qm[acc], pi[j]); r[j] = v > 0 ? v : 0; tot += r[j]; }
            if (tot <= 0) out[n++] = pick(pi, pp, pm, rng);
            else { for (int j = 0; j < pm; j++) r[j] /= tot; out[n++] = pick(pi, r, pm, rng); }
            break;
        }
    }
    if (acc == k) {   // all drafts accepted: bonus token from the target's last position
        const float* P = L + (usize)k * V;
        out[n++] = greedy ? argmax(P, V) : eng_sample(T, P, temp, top_p, top_k, rng);
    }
    // fed for good: cur + acc accepted drafts
    eng_rollback(T, p0 + 1 + acc);
    if (acc < k) eng_rollback(D, p0 + 1 + acc);
    else { eng_record(D, 0); eng_forward(D, d[k - 1], p0 + k, 0); }
    *pos = p0 + 1 + acc;
    return n;
}

int eng_history(Engine* e, const int** toks) { *toks = e->hist; return e->n_past; }
