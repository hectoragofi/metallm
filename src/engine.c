// Qwen3 / Qwen3.5 forward pass, quantized matmuls, tokenizer, chat template, sampler,
// speculative decoding. No libc, no OS.
#include "engine.h"

static void parfor_serial(task_fn fn, void* ctx, int n) { fn(ctx, 0, n); }
parfor_fn g_parfor = parfor_serial;
void (*g_poll)(void);

#define BMAX 64        // max tokens per batched forward step (longer batches are chunked)
#define REC_MAX 16     // max tokens in a speculative record log
#define MAX_CHUNK 8192 // longest pre-tokenizer piece handled by BPE in one go
#define NFC_BLOCK 16384

// ---------------- fp16 ----------------
float h2f(u16 h) {
    u32 s = (u32)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023, b;
    if (e == 0) {
        if (!m) b = s;
        else { e = 113; while (!(m & 1024)) { m <<= 1; e--; } m &= 1023; b = s | (e << 23) | (m << 13); }
    } else if (e == 31) b = s | 0x7f800000 | (m << 13);
    else b = s | ((e + 112) << 23) | (m << 13);
    float f; __builtin_memcpy(&f, &b, 4); return f;
}
u16 f2h(float f) {
    u32 x; __builtin_memcpy(&x, &f, 4);
    u32 s = (x >> 16) & 0x8000; int e = (int)((x >> 23) & 255) - 112; u32 m = x & 0x7fffff;
    if (((x >> 23) & 255) == 255) return (u16)(s | 0x7c00 | (m ? 0x200 : 0));
    if (e >= 31) return (u16)(s | 0x7c00);
    if (e <= 0) {
        if (e < -10) return (u16)s;
        m |= 0x800000; int sh = 14 - e; u32 r = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) r++;
        return (u16)(s | r);
    }
    u32 r = (u32)(e << 10) | (m >> 13), rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) r++;
    return (u16)(s | r);
}

// ---------------- portable kernels ----------------
static int row_bytes(int type, int cols) { return type == WT_Q8 ? cols : type == WT_Q4 ? cols / 2 : type == WT_Q1 ? cols / 8 : cols / 4; }

static void blk_codes(const QMat* m, int r, int b, int* c) {
    const u8* w = m->w + (usize)r * row_bytes(m->type, m->cols);
    switch (m->type) {
    case WT_Q8: for (int i = 0; i < 32; i++) c[i] = (i8)w[b * 32 + i]; break;
    case WT_Q4: for (int i = 0; i < 16; i++) { u8 v = w[b * 16 + i]; c[i] = v & 15; c[i + 16] = v >> 4; } break;
    case WT_Q1: for (int i = 0; i < 32; i++) c[i] = ((w[b * 4 + i / 8] >> (i & 7)) & 1) * 2; break;
    default: { const u8* g = w + (b / 4) * 32; int sh = 2 * (b & 3); for (int i = 0; i < 32; i++) c[i] = (g[i] >> sh) & 3; }
    }
}
// per-group scale ws and offset coefficient mm: w = ws * code + mm
static inline void grp(const QMat* m, int r, int b, float* ws, float* mm) {
    int gi = r * (m->cols / m->gs) + b / (m->gs / 32);
    float s = h2f(m->s[gi]);
    *ws = s;
    *mm = m->type == WT_Q8 ? 0 : m->type == WT_Q4 ? (m->m ? h2f(m->m[gi]) : -8 * s) : -s;
}
static void matmat_ref(float* out, int ldo, const QMat* m, const QAct* a, int n, int lda, int r0, int r1) {
    int nb = m->cols / 32, c[32];
    for (int r = r0; r < r1; r++)
        for (int t = 0; t < n; t++) {
            const i8* x = a->q + (usize)t * lda; const float* xs = a->s + (usize)t * (lda / 32); const float* xm = a->sum + (usize)t * (lda / 32);
            float acc = 0;
            for (int b = 0; b < nb; b++) {
                blk_codes(m, r, b, c);
                int d = 0; for (int i = 0; i < 32; i++) d += c[i] * x[b * 32 + i];
                float ws, mm; grp(m, r, b, &ws, &mm);
                acc += ws * xs[b] * (float)d + mm * xm[b];
            }
            out[(usize)t * ldo + r] = acc;
        }
}
static float dot_ref(const float* a, const float* b, int n) { float s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }
static void axpy_ref(float* y, float a, const float* x, int n) { for (int i = 0; i < n; i++) y[i] += a * x[i]; }
static float dot_h_ref(const float* a, const u16* b, int n) { float s = 0; for (int i = 0; i < n; i++) s += a[i] * h2f(b[i]); return s; }
static void axpy_h_ref(float* y, float a, const u16* x, int n) { for (int i = 0; i < n; i++) y[i] += a * h2f(x[i]); }
static void to_h_ref(u16* d, const float* s, int n) { for (int i = 0; i < n; i++) d[i] = f2h(s[i]); }

Kernels g_k = {matmat_ref, dot_ref, axpy_ref, dot_h_ref, axpy_h_ref, to_h_ref};

int eng_set_isa(int f) {
    Kernels k = {matmat_ref, dot_ref, axpy_ref, dot_h_ref, axpy_h_ref, to_h_ref};
    if (!(f & ISA_AVX2)) f = 0;
    if (!(f & ISA_AVX512)) f &= ~ISA_VNNI512;
    if (f & ISA_AVX2) kernels_avx2(&k);
    if (f & ISA_AVXVNNI) kernels_avxvnni(&k);
    if (f & ISA_AVX512) kernels_avx512(&k, (f & ISA_VNNI512) != 0);
    g_k = k;
    return f;
}
void eng_use_avx2(int on) { eng_set_isa(on ? ISA_AVX2 : 0); }
static int eng_detect_isa_(void);
void eng_use_avx512(int on) {
    int f = eng_detect_isa_();
    eng_set_isa(on ? f : f & ~(ISA_AVX512 | ISA_VNNI512));
}

static void cpuid(u32 leaf, u32 sub, u32* r) { __asm__ volatile("cpuid" : "=a"(r[0]), "=b"(r[1]), "=c"(r[2]), "=d"(r[3]) : "a"(leaf), "c"(sub)); }
static int eng_detect_isa_(void) {
    u32 r[4], f = 0;
    cpuid(0, 0, r); u32 maxl = r[0];
    cpuid(1, 0, r); u32 c1 = r[2];
    if (!(c1 & (1u << 27)) || maxl < 7) return 0;            // OSXSAVE off: XGETBV unusable
    u32 lo, hi; __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0)); (void)hi;
    cpuid(7, 0, r); u32 b7 = r[1], c7 = r[2];
    int fma = (c1 >> 12) & 1, f16c = (c1 >> 29) & 1, avx = (c1 >> 28) & 1;
    if ((lo & 7) == 7 && avx && fma && f16c && (b7 & (1u << 5))) f |= ISA_AVX2;
    if ((f & ISA_AVX2) && (lo & 0xE7) == 0xE7 && (b7 & (1u << 16)) && (b7 & (1u << 17)) && (b7 & (1u << 30)) && (b7 & (1u << 31))) {
        f |= ISA_AVX512;
        if (c7 & (1u << 11)) f |= ISA_VNNI512;
    }
    cpuid(7, 1, r);
    if ((f & ISA_AVX2) && (r[0] & (1u << 4))) f |= ISA_AVXVNNI;
    return (int)f;
}
int eng_detect_isa(void) { return eng_detect_isa_(); }

// ---------------- arena / layout ----------------
typedef struct { u8* p; usize used, cap; } Arena;
static void* aalloc(Arena* a, usize n) {
    usize off = (a->used + 63) & ~(usize)63;
    if (off + n > a->cap) return NULL;
    a->used = off + n;
    return a->p ? a->p + off : (void*)1;
}
static int imax(int a, int b) { return a > b ? a : b; }
static int imin(int a, int b) { return a < b ? a : b; }
static int popc64(u64 x) { int n = 0; while (x) { x &= x - 1; n++; } return n; }

typedef struct { const u32* cnt; const u64* off; } TokHdr;

static usize layout(Engine* e, const ModelHeader* h, int max_seq, Arena* a, int* ok) {
    int dim = h->dim, hid = h->hidden, nh = h->n_heads, hd = h->head_dim, kvd = h->n_kv * hd;
    int gated = (h->flags & HF_ATTN_GATE) != 0, qd = nh * hd;
    int nlin = h->arch == ARCH_QWEN35 ? (int)h->n_layers - popc64(h->full_mask) : 0, nfull = h->n_layers - nlin;
    int nv = h->lin_nv, dk = h->lin_dk, dv = h->lin_dv, conv_dim = 2 * h->lin_nk * dk + nv * dv;
    int vd = nv * dv;
    int big = imax(imax(dim, hid), imax(qd, vd)); big = (big + 31) & ~31;
    int B = BMAX;
    usize st = (usize)nlin * ((usize)nv * dk * dv + (usize)(h->conv_k > 0 ? h->conv_k - 1 : 0) * conv_dim);
    void* p[64]; int i = 0;
    p[i++] = e->layers = aalloc(a, sizeof(Layer) * h->n_layers);
    p[i++] = e->x = aalloc(a, 4ull * B * dim);
    p[i++] = e->xb = aalloc(a, 4ull * B * dim);
    p[i++] = e->q = aalloc(a, 4ull * B * qd * (gated ? 2 : 1));
    p[i++] = e->k = aalloc(a, 4ull * B * kvd);
    p[i++] = e->v = aalloc(a, 4ull * B * kvd);
    p[i++] = e->att_out = aalloc(a, 4ull * B * imax(qd, vd));
    p[i++] = e->hb = aalloc(a, 4ull * B * hid);
    p[i++] = e->hb2 = aalloc(a, 4ull * B * hid);
    p[i++] = e->logits = aalloc(a, 4ull * (REC_MAX + 1) * h->vocab);
    p[i++] = e->att = aalloc(a, 4ull * nh * max_seq);
    p[i++] = e->tmp = aalloc(a, 4ull * imax(big, 4096));
    p[i++] = e->act.q = aalloc(a, (usize)B * big);
    p[i++] = e->act.s = aalloc(a, 4ull * B * (big / 32));
    p[i++] = e->act.sum = aalloc(a, 4ull * B * (big / 32));
    p[i++] = e->act.nks = aalloc(a, 4ull * B * (big / 4));
    p[i++] = e->act.xsx = aalloc(a, 4ull * B * (big / 4));
    p[i++] = e->hadbuf = aalloc(a, 4ull * B * big);
    p[i++] = e->key_cache = aalloc(a, 2ull * nfull * max_seq * kvd + 64);
    p[i++] = e->val_cache = aalloc(a, 2ull * nfull * max_seq * kvd + 64);
    p[i++] = e->hist = aalloc(a, 4ull * max_seq);
    if (nlin) {
        p[i++] = e->qkv = aalloc(a, 4ull * B * conv_dim);
        p[i++] = e->z = aalloc(a, 4ull * B * vd);
        p[i++] = e->ab = aalloc(a, 4ull * B * nv);
        p[i++] = e->bb = aalloc(a, 4ull * B * nv);
        p[i++] = e->conv_state = aalloc(a, 4 * st);   // conv first, then recurrent: one contiguous state
        e->rec_state = 0;
        for (int c = 0; c < 3; c++) p[i++] = e->ckpt[c] = aalloc(a, 4 * st);
        p[i++] = e->rec_log = aalloc(a, 4ull * REC_MAX * nlin * nv * (1 + dk + dv));
        p[i++] = e->conv_log = aalloc(a, 4ull * REC_MAX * nlin * conv_dim);
    }
    e->textcap = imin(16 * max_seq + 65536, 8 << 20);
    p[i++] = e->textbuf = aalloc(a, e->textcap);
    // only header fields here: eng_mem_required() may be given just the 256-byte header
    u64 hs = 1; while (hs < 2ull * h->n_merges && hs < (1ull << 31)) hs <<= 1;
    e->mmask = (u32)(hs - 1);
    p[i++] = e->mkeys = aalloc(a, 8ull * hs);
    p[i++] = e->mvals = aalloc(a, 4ull * hs);
    p[i++] = e->bpe_scratch = aalloc(a, 4 * 4 * MAX_CHUNK);
    p[i++] = e->bpe_rank = aalloc(a, 4 * MAX_CHUNK);
    p[i++] = e->cps = aalloc(a, 4ull * 4 * NFC_BLOCK);
    p[i++] = e->nfcbuf = aalloc(a, 6ull * NFC_BLOCK);
    p[i++] = e->tokbuf = aalloc(a, 4ull * max_seq);
    *ok = 1;
    for (int j = 0; j < i; j++) if (!p[j]) *ok = 0;
    e->state_floats = st;
    return a->used;
}

usize eng_mem_required(const void* file, int max_seq) {
    const ModelHeader* h = file;
    if (max_seq > (int)h->rope_len) max_seq = h->rope_len;
    Engine tmp; Arena a = {0, 0, ~(usize)0}; int ok;
    return layout(&tmp, h, max_seq, &a, &ok) + 64;
}

static const MatDesc* g_desc; static int g_di, g_dn; static const u8* g_base;
static u64 g_size;
static QMat next_mat(Engine* e, int* bad) {
    QMat m; memset(&m, 0, sizeof m);
    if (g_di >= g_dn) { *bad = 1; return m; }
    const MatDesc* d = &g_desc[g_di++];
    m.type = d->type; m.rows = d->rows; m.cols = d->cols; m.gs = d->gs; m.flags = d->flags;
    m.w = g_base + d->off_w; m.s = (const u16*)(g_base + d->off_s); m.m = d->off_m ? (const u16*)(g_base + d->off_m) : NULL;
    if (d->off_w >= g_size || d->off_s >= g_size || d->off_m >= g_size) *bad = 1;
    if (m.type > WT_T2 || m.cols % 32 || m.gs % 32 || m.gs == 0 || ((m.type == WT_Q1 || m.type == WT_T2) && m.cols % 128)) *bad = 1;
    return m;
}

static const float* had_signs(Engine* e, int width) {
    const ModelHeader* h = e->h;
    if (!h->off_had) return NULL;
    const u32* hb = (const u32*)((const u8*)h + h->off_had);
    int n = hb[1]; const float* s = (const float*)(hb + 2 + n);
    for (int i = 0; i < n; i++) { if ((int)hb[2 + i] == width) return s; s += hb[2 + i]; }
    return NULL;
}

static void build_merges(Engine* e);
static int uclass(Engine* e, u32 c);
static void reset_state(Engine* e);
static int utf8_dec(const u8* s, int n, int* adv);

const char* eng_init(Engine* e, const void* file, u64 size, int max_seq, void* arena, usize arena_size) {
    const ModelHeader* h = file;
    const u8* base = file;
    if (size < 256 || memcmp(h->magic, "QWN3", 4) != 0) return "bad model file (magic)";
    if (h->version != 2 || h->gs != 32) return "unsupported model file version (re-export with export.py)";
    if (h->arch > ARCH_QWEN35) return "unsupported model architecture";
    // shape limits the code relies on (stack arrays, the 64-bit layer mask, head grouping)
    if (!h->n_layers || h->n_layers > 64 || !h->n_kv || h->n_heads % h->n_kv || h->head_dim % 2 ||
        h->dim % 32 || h->hidden % 32 || h->vocab < 256)
        return "model file: unsupported shape";
    if (h->arch == ARCH_QWEN35 && (h->lin_dv > 512 || h->lin_dk > 512 || !h->lin_nk || h->lin_nv % h->lin_nk || h->conv_k < 2))
        return "model file: unsupported linear-attention shape";
    if (h->off_tok + 16 * 4 + 11 * 8 > size || h->off_mats + h->n_mats * sizeof(MatDesc) > size)
        return "model file truncated";
    if (((const u32*)((const u8*)file + h->off_tok))[1] != h->n_merges) return "model file: tokenizer header mismatch";
    if (max_seq > (int)h->rope_len) max_seq = h->rope_len;
    memset(e, 0, sizeof(*e));
    e->h = h; e->arch = h->arch;
    e->dim = h->dim; e->hidden = h->hidden; e->L = h->n_layers; e->nh = h->n_heads; e->nkv = h->n_kv;
    e->hd = h->head_dim; e->V = h->vocab; e->max_seq = max_seq; e->eps = h->eps; e->rope_dim = h->rope_dim;
    e->gated = (h->flags & HF_ATTN_GATE) != 0; e->v_tiled = (h->flags & HF_V_TILED) != 0;
    e->gdn_v_grouped = (h->flags & HF_GDN_V_GROUPED) != 0;
    e->nk = h->lin_nk; e->nv = h->lin_nv; e->dk = h->lin_dk; e->dv = h->lin_dv; e->conv_k = h->conv_k;
    e->conv_dim = 2 * e->nk * e->dk + e->nv * e->dv;
    e->B = BMAX;
    Arena a = {(u8*)(((usize)arena + 63) & ~(usize)63), 0, arena_size - 64};
    int ok; layout(e, h, max_seq, &a, &ok);
    if (!ok) return "not enough memory for engine buffers";
    if (e->arch == ARCH_QWEN35 && e->conv_state)
        e->rec_state = e->conv_state + (usize)(e->L - popc64(h->full_mask)) * (e->conv_k - 1) * e->conv_dim;

    e->rope = (const float*)(base + h->off_rope);
    const float* pw = (const float*)(base + h->off_params);
    int nf = 0, nl = 0;
    for (int l = 0; l < e->L; l++) {
        Layer* ly = &e->layers[l];
        ly->full = e->arch == ARCH_QWEN3 || ((h->full_mask >> l) & 1);
        ly->attn_norm = pw; pw += e->dim;
        ly->ffn_norm = pw; pw += e->dim;
        if (ly->full) { ly->q_norm = pw; pw += e->hd; ly->k_norm = pw; pw += e->hd; ly->ci = nf++; }
        else {
            ly->conv_w = pw; pw += (usize)e->conv_dim * e->conv_k;
            ly->dt_bias = pw; pw += e->nv; ly->A = pw; pw += e->nv; ly->gnorm = pw; pw += e->dv; ly->ci = nl++;
        }
    }
    e->final_norm = pw;
    e->n_full = nf; e->n_lin = nl;
    g_desc = (const MatDesc*)(base + h->off_mats); g_di = 0; g_dn = (int)h->n_mats; g_base = base; g_size = size;
    int bad = 0;
    e->embed = next_mat(e, &bad); e->head = next_mat(e, &bad);
    for (int l = 0; l < e->L; l++) {
        Layer* ly = &e->layers[l];
        if (ly->full) { ly->wq = next_mat(e, &bad); ly->wk = next_mat(e, &bad); ly->wv = next_mat(e, &bad); ly->wo = next_mat(e, &bad); }
        else { ly->in_qkv = next_mat(e, &bad); ly->in_z = next_mat(e, &bad); ly->in_a = next_mat(e, &bad); ly->in_b = next_mat(e, &bad); ly->out = next_mat(e, &bad); }
        ly->w1 = next_mat(e, &bad); ly->w3 = next_mat(e, &bad); ly->w2 = next_mat(e, &bad);
    }
    if (bad) return "model file: bad matrix table";
    if (h->off_mats + h->n_mats * sizeof(MatDesc) > size || h->off_tok > size) return "model file truncated";
    if (h->off_had) {
        const u32* hb = (const u32*)(base + h->off_had);
        e->had_bs = hb[0];
        QMat* all[] = {&e->embed, &e->head};
        for (int j = 0; j < 2; j++) if (all[j]->flags & MF_HAD) all[j]->signs = had_signs(e, e->dim);
        for (int l = 0; l < e->L; l++) {
            Layer* ly = &e->layers[l];
            QMat* ms[] = {&ly->wq, &ly->wk, &ly->wv, &ly->wo, &ly->in_qkv, &ly->in_z, &ly->in_a, &ly->in_b, &ly->out, &ly->w1, &ly->w3, &ly->w2};
            for (int j = 0; j < 12; j++) if (ms[j]->flags & MF_HAD) {
                ms[j]->signs = had_signs(e, ms[j]->cols);
                if (ms[j]->cols % e->had_bs) return "model file: Hadamard block does not divide a matrix";
            }
        }
    }

    // tokenizer
    const u32* tc = (const u32*)(base + h->off_tok); const u64* to = (const u64*)(tc + 16);
    const u8* tb = base + h->off_tok;
    e->vocab_offs = (const u32*)(tb + to[0]); e->vocab_bytes = tb + to[1];
    e->merges = (const u32*)(tb + to[2]); e->bytetok = (const u32*)(tb + to[3]);
    e->uranges = (const u32*)(tb + to[4]); e->n_uranges = tc[2];
    e->ccc = (const u32*)(tb + to[5]); e->n_ccc = tc[3];
    e->decomp = (const u32*)(tb + to[6]); e->n_decomp = tc[4];
    e->decomp_data = (const u32*)(tb + to[7]);
    e->comp = (const u32*)(tb + to[8]); e->n_comp = tc[6];
    e->added = (const u32*)(tb + to[9]); e->n_added = tc[7];
    e->think_sys = tb + to[10]; e->think_sys_len = tc[8];
    e->letter_m = tc[9] & 1;
    if (!(tc[9] & 2)) e->n_ccc = e->n_decomp = e->n_comp = -1;   // no NFC normalizer
    e->chat_style = tc[10];
    for (int c = 0; c < 128; c++) e->asc[c] = (u8)uclass(e, c);
    build_merges(e);
    e->ckpt_pos[0] = e->ckpt_pos[1] = e->ckpt_pos[2] = -1; e->asst_pos = e->content_pos = -1;
    reset_state(e);
    return 0;
}

// ---------------- building blocks ----------------
static void rmsnorm(float* o, const float* x, const float* w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / m_sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * inv * w[i];
}
static inline float sigmoidf(float x) { return 1.0f / (1.0f + m_expf(-x)); }
static inline float siluf(float x) { return x / (1.0f + m_expf(-x)); }
static float softplusf(float x) {   // log(1+e^x), torch threshold 20
    if (x > 20) return x;
    float t = m_expf(x);
    // log1p via atanh series on t/(2+t): accurate for all t >= 0
    float y = t / (2.0f + t), y2 = y * y, s = 0, p = y;
    for (int k = 1; k < 40; k += 2) { s += p / k; p *= y2; if (p < 1e-9f) break; }
    return 2 * s;
}

// block Walsh-Hadamard: x <- H x per block of bs (normalised)
static void fwht(float* x, int n, int bs) {
    float sc = 1.0f / m_sqrtf((float)bs);
    for (int o = 0; o < n; o += bs) {
        float* v = x + o;
        for (int len = 1; len < bs; len <<= 1)
            for (int i = 0; i < bs; i += len << 1)
                for (int j = i; j < i + len; j++) { float a = v[j], b = v[j + len]; v[j] = a + b; v[j + len] = a - b; }
        for (int i = 0; i < bs; i++) v[i] *= sc;
    }
}

static void quant_act(const float* x, int n, i8* q, float* s, float* sum, int* nks, float* xsx) {
    for (int g = 0; g < n / 32; g++) {
        const float* xg = x + g * 32;
        float amax = 0;
        for (int i = 0; i < 32; i++) { float a = xg[i] < 0 ? -xg[i] : xg[i]; if (a > amax) amax = a; }
        float sc = amax / 127.0f, inv = amax > 0 ? 127.0f / amax : 0;
        int tot = 0;
        for (int i = 0; i < 32; i++) { float v = xg[i] * inv; int r = (int)(v + (v >= 0 ? 0.5f : -0.5f)); q[g * 32 + i] = (i8)r; tot += r; }
        s[g] = sc; sum[g] = sc * (float)tot;
        for (int j = 0; j < 8; j++) {
            const i8* qj = q + g * 32 + 4 * j;
            nks[g * 8 + j] = -(qj[0] + qj[1] + qj[2] + qj[3]); xsx[g * 8 + j] = sc;
        }
    }
}

// Several matrices sharing one input (n tokens), run as one parallel job.
typedef struct { int nm, n, lda; const QMat* m[4]; float* out[4]; int ldo[4]; const QAct* a; } MM;
static void mm_task(void* c, int i0, int i1) {
    MM* t = c; int base = 0;
    for (int j = 0; j < t->nm; j++) {
        int rows = t->m[j]->rows;
        int a = i0 > base ? i0 - base : 0, b = i1 - base < rows ? i1 - base : rows;
        if (a < b) g_k.matmat(t->out[j], t->ldo[j], t->m[j], t->a, t->n, t->lda, a, b);
        base += rows;
    }
}
// x: n rows of stride ldx. Outputs: out_j[t*ldo_j + r].
static void matmuls(Engine* e, const float* x, int ldx, int n, int nm, const QMat* m0, float* o0, int l0,
                    const QMat* m1, float* o1, int l1, const QMat* m2, float* o2, int l2, const QMat* m3, float* o3, int l3) {
    int cols = m0->cols, lda = (cols + 31) & ~31;
    for (int t = 0; t < n; t++) {
        const float* xt = x + (usize)t * ldx;
        if (m0->flags & MF_HAD) {
            float* hb = e->hadbuf;
            if (m0->signs) for (int i = 0; i < cols; i++) hb[i] = xt[i] * m0->signs[i];
            else memcpy(hb, xt, 4ull * cols);
            fwht(hb, cols, e->had_bs);
            xt = hb;
        }
        quant_act(xt, cols, e->act.q + (usize)t * lda, e->act.s + (usize)t * (lda / 32), e->act.sum + (usize)t * (lda / 32),
                  e->act.nks + (usize)t * (lda / 4), e->act.xsx + (usize)t * (lda / 4));
    }
    MM t = {nm, n, lda, {m0, m1, m2, m3}, {o0, o1, o2, o3}, {l0, l1, l2, l3}, &e->act};
    int rows = 0; for (int j = 0; j < nm; j++) rows += t.m[j]->rows;
    g_parfor(mm_task, &t, rows);
}
#define MM1(e, x, ldx, n, m, o, lo) matmuls(e, x, ldx, n, 1, m, o, lo, 0, 0, 0, 0, 0, 0, 0, 0, 0)

static void deq_row(Engine* e, const QMat* m, int r, float* out) {
    int c[32];
    for (int b = 0; b < m->cols / 32; b++) {
        blk_codes(m, r, b, c);
        float ws, mm; grp(m, r, b, &ws, &mm);
        for (int i = 0; i < 32; i++) out[b * 32 + i] = ws * c[i] + mm;
    }
    if (m->flags & MF_HAD) {   // embedding table stored in the rotated basis: h = s * (H z)
        fwht(out, m->cols, e->had_bs);
        if (m->signs) for (int i = 0; i < m->cols; i++) out[i] *= m->signs[i];
    }
}

static void rope(float* v, const float* cs, int rd) {
    int half = rd / 2;
    for (int i = 0; i < half; i++) {
        float c = cs[2 * i], s = cs[2 * i + 1];
        float a = v[i], b = v[i + half];
        v[i] = a * c - b * s;
        v[i + half] = b * c + a * s;
    }
}

// ---------------- full attention ----------------
typedef struct { Engine* e; int ci, pos; const float* q; float* out; } AT;
static void att_task(void* c, int h0, int h1) {
    AT* t = c; Engine* e = t->e;
    int hd = e->hd, kvd = e->nkv * hd, grp = e->nh / e->nkv, S = e->max_seq;
    float scale = 1.0f / m_sqrtf((float)hd);
    const u16* kc = e->key_cache + (usize)t->ci * S * kvd;
    const u16* vc = e->val_cache + (usize)t->ci * S * kvd;
    for (int h = h0; h < h1; h++) {
        const float* q = t->q + h * hd;
        float* att = e->att + (usize)h * S;
        int kvh = (h / grp) * hd;
        float mx = -3.0e38f;
        for (int p = 0; p <= t->pos; p++) {
            float s = g_k.dot_h(q, kc + (usize)p * kvd + kvh, hd) * scale;
            att[p] = s; if (s > mx) mx = s;
        }
        float sum = 0;
        for (int p = 0; p <= t->pos; p++) { att[p] = m_expf(att[p] - mx); sum += att[p]; }
        float inv = 1.0f / sum;
        float* o = t->out + h * hd;
        memset(o, 0, 4 * hd);
        for (int p = 0; p <= t->pos; p++) g_k.axpy_h(o, att[p] * inv, vc + (usize)p * kvd + kvh, hd);
    }
}

static void layer_full(Engine* e, Layer* ly, int n, int pos) {
    int dim = e->dim, hd = e->hd, kvd = e->nkv * hd, qd = e->nh * hd, S = e->max_seq;
    int qrow = qd * (e->gated ? 2 : 1);
    matmuls(e, e->xb, dim, n, 3, &ly->wq, e->q, qrow, &ly->wk, e->k, kvd, &ly->wv, e->v, kvd, 0, 0, 0);
    for (int t = 0; t < n; t++) {
        int p = pos + t;
        const float* cs = e->rope + (usize)p * e->rope_dim;
        float* q = e->q + (usize)t * qrow; float* k = e->k + (usize)t * kvd; float* v = e->v + (usize)t * kvd;
        for (int h = 0; h < e->nh; h++) { rmsnorm(q + h * hd, q + h * hd, ly->q_norm, hd, e->eps); rope(q + h * hd, cs, e->rope_dim); }
        for (int h = 0; h < e->nkv; h++) { rmsnorm(k + h * hd, k + h * hd, ly->k_norm, hd, e->eps); rope(k + h * hd, cs, e->rope_dim); }
        u16* kd = e->key_cache + ((usize)ly->ci * S + p) * kvd;
        u16* vd = e->val_cache + ((usize)ly->ci * S + p) * kvd;
        g_k.to_h(kd, k, kvd); g_k.to_h(vd, v, kvd);
        AT at = {e, ly->ci, p, q, e->att_out + (usize)t * qd};
        g_parfor(att_task, &at, e->nh);
        if (e->gated) { float* g = q + qd; float* o = e->att_out + (usize)t * qd; for (int i = 0; i < qd; i++) o[i] *= sigmoidf(g[i]); }
    }
    MM1(e, e->att_out, qd, n, &ly->wo, e->xb, dim);
}

// ---------------- Gated DeltaNet ----------------
typedef struct { Engine* e; Layer* ly; int t; float* conv; float* o; float* log; } GT;
static void gdn_task(void* c, int h0, int h1) {
    GT* t = c; Engine* e = t->e; Layer* ly = t->ly;
    int nk = e->nk, nv = e->nv, dk = e->dk, dv = e->dv, rep = nv / nk;
    float* S0 = e->rec_state + (usize)ly->ci * nv * dk * dv;
    const float* qa = t->conv; const float* ka = t->conv + nk * dk; const float* va = t->conv + 2 * nk * dk;
    const float* z = e->z + (usize)t->t * nv * dv;
    float a_ = 0; (void)a_;
    float kvm[512], dl[512];
    for (int h = h0; h < h1; h++) {
        int kh = e->v_tiled ? h % nk : h / rep;
        const float* q = qa + kh * dk; const float* k = ka + kh * dk; const float* v = va + h * dv;
        float* S = S0 + (usize)h * dk * dv;
        float beta = sigmoidf(e->bb[(usize)t->t * nv + h]);
        float g = ly->A[h] * softplusf(e->ab[(usize)t->t * nv + h] + ly->dt_bias[h]);
        float decay = m_expf(g);
        for (int j = 0; j < dv; j++) kvm[j] = 0;
        for (int i = 0; i < dk; i++) g_k.axpy(kvm, k[i], S + (usize)i * dv, dv);
        for (int j = 0; j < dv; j++) dl[j] = (v[j] - decay * kvm[j]) * beta;
        float* o = t->o + (usize)h * dv;
        for (int j = 0; j < dv; j++) o[j] = 0;
        for (int i = 0; i < dk; i++) {
            float* Si = S + (usize)i * dv;
            for (int j = 0; j < dv; j++) Si[j] = decay * Si[j] + k[i] * dl[j];
            g_k.axpy(o, q[i], Si, dv);
        }
        if (t->log) {   // rollback log: decay, k, delta
            float* lg = t->log + (usize)h * (1 + dk + dv);
            lg[0] = decay; memcpy(lg + 1, k, 4ull * dk); memcpy(lg + 1 + dk, dl, 4ull * dv);
        }
        // gated RMSNorm: norm(o) * w * silu(z)
        float ss = 0; for (int j = 0; j < dv; j++) ss += o[j] * o[j];
        float inv = 1.0f / m_sqrtf(ss / dv + e->eps);
        const float* zh = z + (usize)h * dv;
        for (int j = 0; j < dv; j++) o[j] = o[j] * inv * ly->gnorm[j] * siluf(zh[j]);
    }
}

static void l2n(float* x, int n, float scale) {
    float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = scale / m_sqrtf(ss + 1e-6f);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

static void layer_lin(Engine* e, Layer* ly, int n) {
    int dim = e->dim, nv = e->nv, dv = e->dv, cd = e->conv_dim, K = e->conv_k, nk = e->nk, dk = e->dk;
    matmuls(e, e->xb, dim, n, 2, &ly->in_qkv, e->qkv, cd, &ly->in_z, e->z, nv * dv, 0, 0, 0, 0, 0, 0);
    matmuls(e, e->xb, dim, n, 2, &ly->in_a, e->ab, nv, &ly->in_b, e->bb, nv, 0, 0, 0, 0, 0, 0);
    float* cst = e->conv_state + (usize)ly->ci * (K - 1) * cd;
    float* conv = e->tmp;
    for (int t = 0; t < n; t++) {
        float* x = e->qkv + (usize)t * cd;
        int ri = e->rec_on ? e->rec_n + t : -1;
        if (ri >= 0) memcpy(e->conv_log + ((usize)ri * e->n_lin + ly->ci) * cd, x, 4ull * cd);
        for (int c = 0; c < cd; c++) {
            const float* w = ly->conv_w + (usize)c * K;
            float s = w[K - 1] * x[c];
            for (int j = 0; j < K - 1; j++) s += w[j] * cst[(usize)j * cd + c];
            conv[c] = siluf(s);
        }
        memmove(cst, cst + cd, 4ull * (K - 2) * cd);
        memcpy(cst + (usize)(K - 2) * cd, x, 4ull * cd);
        float qs = 1.0f / m_sqrtf((float)dk);
        for (int h = 0; h < nk; h++) { l2n(conv + h * dk, dk, qs); l2n(conv + nk * dk + h * dk, dk, 1.0f); }
        float* o = e->att_out + (usize)t * nv * dv;
        GT gt = {e, ly, t, conv, o, ri >= 0 ? e->rec_log + ((usize)ri * e->n_lin + ly->ci) * nv * (1 + dk + dv) : 0};
        g_parfor(gdn_task, &gt, nv);
        if (e->gdn_v_grouped && nv != nk) {   // tiled [rep][nk][dv] -> grouped [nk][rep][dv] for the rotated out_proj
            int rep = nv / nk; float* tmp = e->hadbuf;
            for (int r = 0; r < rep; r++) for (int k = 0; k < nk; k++) memcpy(tmp + ((usize)k * rep + r) * dv, o + ((usize)r * nk + k) * dv, 4ull * dv);
            memcpy(o, tmp, 4ull * nv * dv);
        }
    }
    MM1(e, e->att_out, nv * dv, n, &ly->out, e->xb, dim);
}

// ---------------- forward ----------------
static void snapshot(Engine* e, int slot, int pos) {
    if (e->state_floats) memcpy(e->ckpt[slot], e->conv_state, 4 * e->state_floats);
    e->ckpt_pos[slot] = pos;
}
static void restore(Engine* e, int slot) {
    if (e->state_floats) memcpy(e->conv_state, e->ckpt[slot], 4 * e->state_floats);
    e->n_past = e->ckpt_pos[slot];
}
static void reset_state(Engine* e) {
    if (e->state_floats) memset(e->conv_state, 0, 4 * e->state_floats);
    e->n_past = 0;
}

// n <= BMAX tokens at physical pos; logits for the last token (want==1), all tokens into lout (want==2)
static void forward_chunk(Engine* e, const int* toks, int n, int pos, int want, float* lout) {
    int dim = e->dim;
    for (int t = 0; t < n; t++) { deq_row(e, &e->embed, toks[t], e->x + (usize)t * dim); e->hist[pos + t] = toks[t]; }
    for (int l = 0; l < e->L; l++) {
        Layer* ly = &e->layers[l];
        for (int t = 0; t < n; t++) rmsnorm(e->xb + (usize)t * dim, e->x + (usize)t * dim, ly->attn_norm, dim, e->eps);
        if (ly->full) layer_full(e, ly, n, pos); else layer_lin(e, ly, n);
        for (usize i = 0; i < (usize)n * dim; i++) e->x[i] += e->xb[i];
        for (int t = 0; t < n; t++) rmsnorm(e->xb + (usize)t * dim, e->x + (usize)t * dim, ly->ffn_norm, dim, e->eps);
        matmuls(e, e->xb, dim, n, 2, &ly->w1, e->hb, e->hidden, &ly->w3, e->hb2, e->hidden, 0, 0, 0, 0, 0, 0);
        for (usize i = 0; i < (usize)n * e->hidden; i++) e->hb[i] = siluf(e->hb[i]) * e->hb2[i];
        MM1(e, e->hb, e->hidden, n, &ly->w2, e->xb, dim);
        for (usize i = 0; i < (usize)n * dim; i++) e->x[i] += e->xb[i];
        if (g_poll) g_poll();
    }
    if (e->rec_on) e->rec_n += n;
    e->n_past = pos + n;
    if (!want) return;
    int t0 = want == 1 ? n - 1 : 0, nt = want == 1 ? 1 : n;
    for (int t = t0; t < n; t++) rmsnorm(e->x + (usize)t * dim, e->x + (usize)t * dim, e->final_norm, dim, e->eps);
    MM1(e, e->x + (usize)t0 * dim, dim, nt, &e->head, lout, e->V);
}

// physical forward with checkpoint splitting (checkpoint slot 1 = assistant start)
static void forward_phys(Engine* e, const int* toks, int n, int pos, int want, float* lout) {
    while (n > 0) {
        int c = imin(n, BMAX);
        if (e->rec_on) { if (REC_MAX - e->rec_n <= 0) e->rec_on = 0; else c = imin(c, REC_MAX - e->rec_n); }
        if (e->asst_pos > pos && e->asst_pos < pos + c) c = e->asst_pos - pos;
        if (pos == e->asst_pos && e->ckpt_pos[1] != pos) snapshot(e, 1, pos);
        int last = c == n;
        forward_chunk(e, toks, c, pos, want == 2 ? 2 : last ? want : 0, lout);
        if (want == 2) lout += (usize)c * e->V;
        toks += c; pos += c; n -= c;
    }
    if (e->asst_pos == pos && e->ckpt_pos[1] != pos) snapshot(e, 1, pos);
}

#include "engine_api.inc"
#include "tokenizer.inc"
