// AVX2 + FMA + F16C kernels. This file alone is compiled with -mavx2 -mfma -mf16c; the
// platform layer must enable AVX (CR4.OSXSAVE + XCR0 bits 0-2) before calling into it.
// Built a second time with -mavxvnni -DAVXVNNI as kernels_avxvnni() (VEX vpdpbusd).
//
// Per 32-weight block: d = sum(code * x) (u8 codes x s8 activations, 8 int32 lanes), the code
// offset folded in as integers through the activation lane sums (d += c * nks, nks = -sum x),
// then acc += float(d) * (ws * xs). Two accumulators (even/odd blocks) per token; the
// per-(row, token) operation order does not depend on how many tokens run together, so batched
// and single-token results are bit-identical.
#include "engine.h"
#include <immintrin.h>

static inline float hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}
static inline float cvth(u16 h) { return _cvtsh_ss(h); }

#ifdef AVXVNNI
#define DOTU(init, u, x) _mm256_dpbusd_avx_epi32(init, u, x)
#define Q8_BIASED 1          // Q8 as (w ^ 0x80) with c = 128: one vpdpbusd, no sign fix-up
#define KERNELS_FN kernels_avxvnni
#else
#define DOTU(init, u, x) _mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(u, x), _mm256_set1_epi16(1)), init)
#define Q8_BIASED 0          // vpmaddubsw would saturate on 255 * 127 * 2: use |w| and sign(x, w)
#define KERNELS_FN kernels_avx2
#endif

enum { T_Q8, T_Q4S, T_Q4A, T_Q1, T_T2 };

static inline __m256i ld_q4(const u8* p) {
    __m128i v = _mm_loadu_si128((const __m128i*)p), m = _mm_set1_epi8(15);
    return _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(v, 4), m), _mm_and_si128(v, m));
}
static inline __m256i ld_q1(const u8* p) {   // 32 sign bits -> bytes {0, 2}
    u32 v; __builtin_memcpy(&v, p, 4);
    const __m256i shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bits = _mm256_set1_epi64x(0x8040201008040201ll);
    __m256i b = _mm256_shuffle_epi8(_mm256_set1_epi32((int)v), shuf);
    b = _mm256_cmpeq_epi8(_mm256_and_si256(b, bits), bits);
    return _mm256_and_si256(b, _mm256_set1_epi8(2));
}
static inline __m256i ld_t2(const u8* g, int k) {   // block k (0..3) of a 128-weight group
    __m256i v = _mm256_loadu_si256((const __m256i*)g);
    return _mm256_and_si256(_mm256_srl_epi16(v, _mm_cvtsi32_si128(2 * k)), _mm256_set1_epi8(3));
}

static inline int rbytes(int ty, int cols) { return ty == T_Q8 ? cols : (ty == T_Q4S || ty == T_Q4A) ? cols / 2 : ty == T_Q1 ? cols / 8 : cols / 4; }

// One block b of row r for NT tokens into accumulator set A (0/1).
#define BLOCK(b, A)                                                                                           \
    {                                                                                                         \
        __m256i u, wr = _mm256_setzero_si256();                                                               \
        if (TY == T_Q8) { wr = _mm256_loadu_si256((const __m256i*)(w + (b) * 32));                            \
                          u = Q8_BIASED ? _mm256_xor_si256(wr, _mm256_set1_epi8((char)0x80)) : _mm256_sign_epi8(wr, wr); } \
        else if (TY == T_Q4S || TY == T_Q4A) u = ld_q4(w + (b) * 16);                                         \
        else if (TY == T_Q1) u = ld_q1(w + (b) * 4);                                                          \
        else u = ld_t2(w + ((b) >> 2) * 32, (b) & 3);                                                         \
        const __m256 wsv = _mm256_set1_ps(cvth(sc[(b) >> LSPG]));                                             \
        const float mmv = TY == T_Q4A ? cvth(mn[(b) >> LSPG]) : 0;                                            \
        for (int t = 0; t < NT; t++) {                                                                        \
            __m256i xv = _mm256_loadu_si256((const __m256i*)(xq + (usize)t * lda + (b) * 32));                \
            __m256i d;                                                                                        \
            if (TY == T_Q8 && !Q8_BIASED) d = _mm256_madd_epi16(_mm256_maddubs_epi16(u, _mm256_sign_epi8(xv, wr)), _mm256_set1_epi16(1)); \
            else if (TY == T_Q4A) d = DOTU(_mm256_setzero_si256(), u, xv);                                    \
            else {                                                                                            \
                __m256i k = _mm256_loadu_si256((const __m256i*)(xk + (usize)t * ldk + (b) * 8));             \
                if (TY == T_Q8) k = _mm256_slli_epi32(k, 7);                                                  \
                else if (TY == T_Q4S) k = _mm256_slli_epi32(k, 3);                                            \
                d = DOTU(k, u, xv);                                                                           \
            }                                                                                                 \
            __m256 sv = _mm256_mul_ps(wsv, _mm256_broadcast_ss(xs + (usize)t * ldb + (b)));                   \
            acc##A[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), sv, acc##A[t]);                                \
            if (TY == T_Q4A) off##A[t] += mmv * xm[(usize)t * ldb + (b)];                                     \
        }                                                                                                     \
    }

#define KERN(NAME, TY_, NT_, LSPG_)                                                                           \
static inline __attribute__((always_inline)) void NAME(const QMat* m, int r, const QAct* a, int t0, int lda, float* out, int ldo) { \
    enum { TY = TY_, NT = NT_, LSPG = LSPG_ };                                                                \
    const int nb = m->cols / 32, ng = m->cols / m->gs, ldb = lda / 32, ldk = lda / 4;                         \
    const u8* w = m->w + (usize)r * rbytes(TY, m->cols);                                                      \
    const u16* sc = m->s + (usize)r * ng; const u16* mn = TY == T_Q4A ? m->m + (usize)r * ng : 0;            \
    const i8* xq = a->q + (usize)t0 * lda; const float* xs = a->s + (usize)t0 * ldb;                          \
    const float* xm = a->sum + (usize)t0 * ldb; const int* xk = a->nks + (usize)t0 * ldk;                     \
    __m256 acc0[NT], acc1[NT]; float off0[NT], off1[NT];                                                      \
    for (int t = 0; t < NT; t++) { acc0[t] = acc1[t] = _mm256_setzero_ps(); off0[t] = off1[t] = 0; }          \
    int b = 0;                                                                                                \
    for (; b + 1 < nb; b += 2) { BLOCK(b, 0) BLOCK(b + 1, 1) }                                                \
    if (b < nb) BLOCK(b, 0)                                                                                   \
    for (int t = 0; t < NT; t++)                                                                              \
        out[(usize)(t0 + t) * ldo + r] = hsum8(_mm256_add_ps(acc0[t], acc1[t])) + (off0[t] + off1[t]);        \
    (void)xm; (void)mn; (void)xk;                                                                             \
}
#define KSET(TY, TN, L)                                                                                       \
    KERN(k_##TN##_1, TY, 1, L) KERN(k_##TN##_2, TY, 2, L) KERN(k_##TN##_3, TY, 3, L) KERN(k_##TN##_4, TY, 4, L) \
static void mm_##TN(float* out, int ldo, const QMat* m, const QAct* a, int n, int lda, int r0, int r1) {      \
    for (int r = r0; r < r1; r++)                                                                             \
        for (int t0 = 0; t0 < n; t0 += 4) {                                                                   \
            int nt = n - t0 < 4 ? n - t0 : 4;                                                                 \
            if (nt == 4) k_##TN##_4(m, r, a, t0, lda, out, ldo);                                              \
            else if (nt == 3) k_##TN##_3(m, r, a, t0, lda, out, ldo);                                         \
            else if (nt == 2) k_##TN##_2(m, r, a, t0, lda, out, ldo);                                         \
            else k_##TN##_1(m, r, a, t0, lda, out, ldo);                                                      \
        }                                                                                                     \
}
// group sizes: 32 / 64 / 128 (LSPG = log2(gs / 32)); T2 and Q1 are packed per 128 (gs >= 128)
KSET(T_Q8, q8_0, 0) KSET(T_Q8, q8_1, 1) KSET(T_Q8, q8_2, 2)
KSET(T_Q4S, q4s_0, 0) KSET(T_Q4S, q4s_1, 1) KSET(T_Q4S, q4s_2, 2)
KSET(T_Q4A, q4a_0, 0) KSET(T_Q4A, q4a_1, 1) KSET(T_Q4A, q4a_2, 2)
KSET(T_Q1, q1_0, 0) KSET(T_Q1, q1_1, 1)   // gs 128 (the Bonsai models): mm_q1g below
KSET(T_T2, t2_2, 2)

// ---- 1-bit, groups of 128: several rows at once ----
// Decode of the 1-bit models is limited by arithmetic, not memory, so this path trims the work
// per 32-weight block: the group's 16 bytes of sign bits are loaded once and each block's bytes
// are picked with one shuffle; without VNNI the sign is applied to the activations (no code
// offset, no lane sums to add); the activation scale goes in with one FMA per block and the
// weight scale once per group. About 1.4x faster than the generic path for one token.
// NR rows share every activation load. Per (row, token) the arithmetic is the same whatever NR
// and NT are, so batched and single-token results stay bit-identical.
static inline __m256i q1_neg(__m256i g, int k) {   // block k of a group: 0xFF where the bit is 0 (weight -s)
    const __m256i sh = _mm256_add_epi8(_mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                                        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3), _mm256_set1_epi8((char)(4 * k)));
    const __m256i bits = _mm256_set1_epi64x(0x8040201008040201ll);
    return _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(g, sh), bits), _mm256_setzero_si256());
}
// Per row: the operand that carries the signs. Per token: per 4-byte lane, sum(+-x).
#ifdef AVXVNNI
static inline __m256i q1_prep(__m256i neg) { return _mm256_andnot_si256(neg, _mm256_set1_epi8(2)); }   // codes {0, 2}
static inline __m256i q1_dot(__m256i u, __m256i x, const int* nks) {   // 2 * sum(x where bit) - sum(x)
    return _mm256_dpbusd_avx_epi32(_mm256_loadu_si256((const __m256i*)nks), u, x);
}
#else
static inline __m256i q1_prep(__m256i neg) { return _mm256_or_si256(neg, _mm256_set1_epi8(1)); }      // -1 / +1
static inline __m256i q1_dot(__m256i s, __m256i x, const int* nks) {
    (void)nks;
    return _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(1), _mm256_sign_epi8(x, s)), _mm256_set1_epi16(1));
}
#endif
#define Q1KERN(NAME, NR_, NT_)                                                                                 \
static inline __attribute__((always_inline)) void NAME(const QMat* m, int r, const QAct* a, int t0, int lda, float* out, int ldo) { \
    enum { NR = NR_, NT = NT_ };                                                                               \
    const int ng = m->cols / 128, rb = m->cols / 8, ldb = lda / 32, ldk = lda / 4;                             \
    const u8* w = m->w + (usize)r * rb; const u16* sc = m->s + (usize)r * ng;                                  \
    const i8* xq = a->q + (usize)t0 * lda; const float* xx = a->s + (usize)t0 * ldb;                           \
    const int* xk = a->nks + (usize)t0 * ldk; (void)xk;                                                        \
    __m256 acc[NR][NT];                                                                                        \
    for (int i = 0; i < NR; i++) for (int t = 0; t < NT; t++) acc[i][t] = _mm256_setzero_ps();                 \
    for (int g = 0; g < ng; g++) {                                                                             \
        __m256i gb[NR]; __m256 ga[NR][NT];                                                                     \
        for (int i = 0; i < NR; i++) gb[i] = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(w + (usize)i * rb + g * 16))); \
        _Pragma("unroll") for (int k = 0; k < 4; k++) {                                                        \
            __m256i neg[NR];                                                                                   \
            for (int i = 0; i < NR; i++) neg[i] = q1_prep(q1_neg(gb[i], k));                                   \
            for (int t = 0; t < NT; t++) {                                                                     \
                __m256i xv = _mm256_loadu_si256((const __m256i*)(xq + (usize)t * lda + g * 128 + k * 32));     \
                __m256 xs = _mm256_broadcast_ss(xx + (usize)t * ldb + g * 4 + k);                              \
                for (int i = 0; i < NR; i++) {                                                                 \
                    __m256 d = _mm256_cvtepi32_ps(q1_dot(neg[i], xv, xk + (usize)t * ldk + g * 32 + k * 8));   \
                    ga[i][t] = k ? _mm256_fmadd_ps(d, xs, ga[i][t]) : _mm256_mul_ps(d, xs);                    \
                }                                                                                              \
            }                                                                                                  \
        }                                                                                                      \
        for (int i = 0; i < NR; i++) {                                                                         \
            __m256 ws = _mm256_set1_ps(cvth(sc[(usize)i * ng + g]));                                           \
            for (int t = 0; t < NT; t++) acc[i][t] = _mm256_fmadd_ps(ga[i][t], ws, acc[i][t]);                 \
        }                                                                                                      \
    }                                                                                                          \
    for (int i = 0; i < NR; i++) for (int t = 0; t < NT; t++) out[(usize)(t0 + t) * ldo + r + i] = hsum8(acc[i][t]); \
}
Q1KERN(q1g_r2, 2, 1) Q1KERN(q1g_r1, 1, 1)
Q1KERN(q1g_t2, 1, 2) Q1KERN(q1g_t3, 1, 3) Q1KERN(q1g_t4, 1, 4)
static void mm_q1g(float* out, int ldo, const QMat* m, const QAct* a, int n, int lda, int r0, int r1) {
    if (n == 1) {
        int r = r0;
        for (; r + 2 <= r1; r += 2) q1g_r2(m, r, a, 0, lda, out, ldo);
        for (; r < r1; r++) q1g_r1(m, r, a, 0, lda, out, ldo);
        return;
    }
    for (int r = r0; r < r1; r++)
        for (int t0 = 0; t0 < n; t0 += 4) {
            int nt = n - t0 < 4 ? n - t0 : 4;
            if (nt == 4) q1g_t4(m, r, a, t0, lda, out, ldo);
            else if (nt == 3) q1g_t3(m, r, a, t0, lda, out, ldo);
            else if (nt == 2) q1g_t2(m, r, a, t0, lda, out, ldo);
            else q1g_r1(m, r, a, t0, lda, out, ldo);
        }
}

typedef void (*mmfn)(float*, int, const QMat*, const QAct*, int, int, int, int);
static mmfn pick(const QMat* m) {
    int l = m->gs == 32 ? 0 : m->gs == 64 ? 1 : m->gs == 128 ? 2 : -1;
    switch (m->type) {
    case WT_Q8: return l == 0 ? mm_q8_0 : l == 1 ? mm_q8_1 : l == 2 ? mm_q8_2 : 0;
    case WT_Q4: if (m->m) return l == 0 ? mm_q4a_0 : l == 1 ? mm_q4a_1 : l == 2 ? mm_q4a_2 : 0;
                return l == 0 ? mm_q4s_0 : l == 1 ? mm_q4s_1 : l == 2 ? mm_q4s_2 : 0;
    case WT_Q1: return l == 0 ? mm_q1_0 : l == 1 ? mm_q1_1 : l == 2 ? mm_q1g : 0;
    default: return l == 2 ? mm_t2_2 : 0;
    }
}
static mmfn mm_ref;   // portable fallback for unusual group sizes
static void matmat_avx2(float* out, int ldo, const QMat* m, const QAct* a, int n, int lda, int r0, int r1) {
    mmfn f = pick(m);
    (f ? f : mm_ref)(out, ldo, m, a, n, lda, r0, r1);
}

static float dot_f32_avx2(const float* a, const float* b, int n) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    float s = hsum8(_mm256_add_ps(acc0, acc1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
static void axpy_f32_avx2(float* y, float a, const float* x, int n) {
    __m256 av = _mm256_set1_ps(a);
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_fmadd_ps(av, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    for (; i < n; i++) y[i] += a * x[i];
}
static float dot_h_avx2(const float* a, const u16* b, int n) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + i))), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + i + 8))), acc1);
    }
    float s = hsum8(_mm256_add_ps(acc0, acc1));
    for (; i < n; i++) s += a[i] * cvth(b[i]);
    return s;
}
static void axpy_h_avx2(float* y, float a, const u16* x, int n) {
    __m256 av = _mm256_set1_ps(a);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x + i))), _mm256_loadu_ps(y + i)));
    for (; i < n; i++) y[i] += a * cvth(x[i]);
}
static void to_h_avx2(u16* d, const float* s, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm_storeu_si128((__m128i*)(d + i), _mm256_cvtps_ph(_mm256_loadu_ps(s + i), _MM_FROUND_TO_NEAREST_INT));
    for (; i < n; i++) d[i] = _cvtss_sh(s[i], _MM_FROUND_TO_NEAREST_INT);
}

void KERNELS_FN(Kernels* k) {
    if (!mm_ref) mm_ref = k->matmat;
    k->matmat = matmat_avx2; k->dot = dot_f32_avx2; k->axpy = axpy_f32_avx2;
    k->dot_h = dot_h_avx2; k->axpy_h = axpy_h_avx2; k->to_h = to_h_avx2;
}
