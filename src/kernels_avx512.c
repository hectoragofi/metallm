// AVX-512 kernels (F, BW, VL, DQ; optional VNNI). Compiled with -mavx512f -mavx512bw -mavx512vl
// -mavx512dq -mavx512vnni -mf16c -mfma. The platform must set XCR0 |= 0xE7 (x87, SSE, AVX,
// opmask, ZMM_Hi256, Hi16_ZMM) before calling into it.
//
// One zmm covers two 32-weight blocks. Codes are unsigned bytes; the code offset is folded in as
// integers through the activation lane sums (d = c * nks + sum(code * x)), which VNNI does for
// free as vpdpbusd's accumulator input. Q8 is biased (w ^ 0x80, c = 128) with VNNI; without VNNI
// it uses |w| and x negated where w < 0 (255 * 127 * 2 would saturate vpmaddubsw).
// Matrices whose width is not a multiple of 64 use the previously installed (AVX2) path.
#include "engine.h"
#include <immintrin.h>

typedef void (*mmfn)(float*, int, const QMat*, const QAct*, int, int, int, int);
static mmfn mm_fallback;

static inline float cvth(u16 h) { return _cvtsh_ss(h); }
static inline __m512i dot_vnni(__m512i init, __m512i u, __m512i x) { return _mm512_dpbusd_epi32(init, u, x); }
static inline __m512i dot_bw(__m512i init, __m512i u, __m512i x) {
    return _mm512_add_epi32(_mm512_madd_epi16(_mm512_maddubs_epi16(u, x), _mm512_set1_epi16(1)), init);
}

enum { T_Q8, T_Q4S, T_Q4A, T_Q1, T_T2 };

static inline __m512i ld2_q4(const u8* p) {   // 2 blocks x 16 bytes, byte i = q[i] | q[i+16] << 4
    __m512i z = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)p));
    z = _mm512_shuffle_i64x2(z, z, _MM_SHUFFLE(1, 1, 0, 0));   // 128-bit lanes: b0, b0, b1, b1
    const __m512i sh = _mm512_set_epi64(4, 4, 0, 0, 4, 4, 0, 0);
    return _mm512_and_si512(_mm512_srlv_epi64(z, sh), _mm512_set1_epi8(15));
}
static inline __m512i ld2_q1(const u8* p) {   // 64 sign bits -> bytes {0, 2}
    u64 m; __builtin_memcpy(&m, p, 8);
    return _mm512_maskz_mov_epi8(_cvtu64_mask64(m), _mm512_set1_epi8(2));
}
static inline __m512i ld2_t2(const u8* g, int k) {   // blocks k, k+1 (k = 0 or 2) of a 128-weight group
    __m512i z = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)g));
    __m512i sh = _mm512_set_epi64(2 * k + 2, 2 * k + 2, 2 * k + 2, 2 * k + 2, 2 * k, 2 * k, 2 * k, 2 * k);
    return _mm512_and_si512(_mm512_srlv_epi64(z, sh), _mm512_set1_epi8(3));
}
static inline int rbytes(int ty, int cols) { return ty == T_Q8 ? cols : (ty == T_Q4S || ty == T_Q4A) ? cols / 2 : ty == T_Q1 ? cols / 8 : cols / 4; }

// blocks b, b+1 (b even) of row r for NT tokens into accumulator set A
#define PAIR(b, A, DOT)                                                                                          \
    {                                                                                                         \
        __m512i u, wr = _mm512_setzero_si512(); __mmask64 neg = 0;                                            \
        if (TY == T_Q8) { wr = _mm512_loadu_si512(w + (b) * 32);                                              \
                          if (VNNI) u = _mm512_xor_si512(wr, _mm512_set1_epi8((char)0x80));                   \
                          else { u = _mm512_abs_epi8(wr); neg = _mm512_movepi8_mask(wr); } }                  \
        else if (TY == T_Q4S || TY == T_Q4A) u = ld2_q4(w + (b) * 16);                                        \
        else if (TY == T_Q1) u = ld2_q1(w + (b) * 4);                                                         \
        else u = ld2_t2(w + ((b) >> 2) * 32, (b) & 3);                                                        \
        __m512 wsv; float m0 = 0, m1 = 0;                                                                     \
        if (LSPG == 0) {                                                                                      \
            wsv = _mm512_insertf32x8(_mm512_set1_ps(cvth(sc[b])), _mm256_set1_ps(cvth(sc[(b) + 1])), 1);       \
            if (TY == T_Q4A) { m0 = cvth(mn[b]); m1 = cvth(mn[(b) + 1]); }                                    \
        } else {                                                                                              \
            wsv = _mm512_set1_ps(cvth(sc[(b) >> LSPG]));                                                      \
            if (TY == T_Q4A) m0 = m1 = cvth(mn[(b) >> LSPG]);                                                 \
        }                                                                                                     \
        for (int t = 0; t < NT; t++) {                                                                        \
            __m512i xv = _mm512_loadu_si512(xq + (usize)t * lda + (b) * 32);                                  \
            __m512i d;                                                                                        \
            if (TY == T_Q8 && !VNNI) d = DOT(_mm512_setzero_si512(), u, _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv)); \
            else if (TY == T_Q4A) d = DOT(_mm512_setzero_si512(), u, xv);                                     \
            else {                                                                                            \
                __m512i k = _mm512_loadu_si512(xk + (usize)t * ldk + (b) * 8);                                \
                if (TY == T_Q8) k = _mm512_slli_epi32(k, 7);                                                  \
                else if (TY == T_Q4S) k = _mm512_slli_epi32(k, 3);                                            \
                d = DOT(k, u, xv);                                                                            \
            }                                                                                                 \
            __m512 sv = _mm512_mul_ps(wsv, _mm512_loadu_ps(xx + (usize)t * ldk + (b) * 8));                   \
            acc##A[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d), sv, acc##A[t]);                                \
            if (TY == T_Q4A) { const float* m2 = xm + (usize)t * ldb + (b); off##A[t] += m0 * m2[0] + m1 * m2[1]; } \
        }                                                                                                     \
    }

#define KERN(NAME, TY_, NT_, LSPG_, VNNI_, DOT)                                                               \
static inline __attribute__((always_inline)) void NAME(const QMat* m, int r, const QAct* a, int t0, int lda, float* out, int ldo) { \
    enum { TY = TY_, NT = NT_, LSPG = LSPG_, VNNI = VNNI_ };                                                  \
    const int nb = m->cols / 32, ng = m->cols / m->gs, ldb = lda / 32, ldk = lda / 4;                         \
    const u8* w = m->w + (usize)r * rbytes(TY, m->cols);                                                      \
    const u16* sc = m->s + (usize)r * ng; const u16* mn = TY == T_Q4A ? m->m + (usize)r * ng : 0;            \
    const i8* xq = a->q + (usize)t0 * lda; const float* xm = a->sum + (usize)t0 * ldb;                        \
    const int* xk = a->nks + (usize)t0 * ldk; const float* xx = a->xsx + (usize)t0 * ldk;                     \
    __m512 acc0[NT], acc1[NT]; float off0[NT], off1[NT];                                                      \
    for (int t = 0; t < NT; t++) { acc0[t] = acc1[t] = _mm512_setzero_ps(); off0[t] = off1[t] = 0; }          \
    int b = 0;                                                                                                \
    for (; b + 3 < nb; b += 4) { PAIR(b, 0, DOT) PAIR(b + 2, 1, DOT) }                                                  \
    if (b < nb) PAIR(b, 0, DOT)                                                                                  \
    for (int t = 0; t < NT; t++)                                                                              \
        out[(usize)(t0 + t) * ldo + r] = _mm512_reduce_add_ps(_mm512_add_ps(acc0[t], acc1[t])) + (off0[t] + off1[t]); \
    (void)xm; (void)mn; (void)xk;                                                                             \
}
#define KSET(TY, TN, L, V, DOT)                                                                               \
    KERN(k_##TN##_1, TY, 1, L, V, DOT) KERN(k_##TN##_2, TY, 2, L, V, DOT)                                     \
    KERN(k_##TN##_3, TY, 3, L, V, DOT) KERN(k_##TN##_4, TY, 4, L, V, DOT)                                     \
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
#define KALL(S, V, DOT)                                                                                       \
    KSET(T_Q8, q8_0##S, 0, V, DOT) KSET(T_Q8, q8_1##S, 1, V, DOT) KSET(T_Q8, q8_2##S, 2, V, DOT)             \
    KSET(T_Q4S, q4s_0##S, 0, V, DOT) KSET(T_Q4S, q4s_1##S, 1, V, DOT) KSET(T_Q4S, q4s_2##S, 2, V, DOT)       \
    KSET(T_Q4A, q4a_0##S, 0, V, DOT) KSET(T_Q4A, q4a_1##S, 1, V, DOT) KSET(T_Q4A, q4a_2##S, 2, V, DOT)       \
    KSET(T_Q1, q1_0##S, 0, V, DOT) KSET(T_Q1, q1_1##S, 1, V, DOT) KSET(T_Q1, q1_2##S, 2, V, DOT)             \
    KSET(T_T2, t2_2##S, 2, V, DOT)                                                                            \
static mmfn pick##S(const QMat* m) {                                                                          \
    int l = m->gs == 32 ? 0 : m->gs == 64 ? 1 : m->gs == 128 ? 2 : -1;                                        \
    if (l < 0 || (m->cols & 63)) return 0;                                                                    \
    switch (m->type) {                                                                                        \
    case WT_Q8: return l == 0 ? mm_q8_0##S : l == 1 ? mm_q8_1##S : mm_q8_2##S;                                \
    case WT_Q4: if (m->m) return l == 0 ? mm_q4a_0##S : l == 1 ? mm_q4a_1##S : mm_q4a_2##S;                  \
                return l == 0 ? mm_q4s_0##S : l == 1 ? mm_q4s_1##S : mm_q4s_2##S;                             \
    case WT_Q1: return l == 0 ? mm_q1_0##S : l == 1 ? mm_q1_1##S : mm_q1_2##S;                                \
    default: return l == 2 ? mm_t2_2##S : 0;                                                                  \
    }                                                                                                         \
}                                                                                                             \
static void matmat##S(float* out, int ldo, const QMat* m, const QAct* a, int n, int lda, int r0, int r1) {    \
    mmfn f = pick##S(m);                                                                                      \
    (f ? f : mm_fallback)(out, ldo, m, a, n, lda, r0, r1);                                                    \
}
KALL(_v, 1, dot_vnni)
KALL(_b, 0, dot_bw)

static float dot_f32_512(const float* a, const float* b, int n) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    if (i < n) { __mmask16 k = (__mmask16)((1u << (n - i)) - 1); acc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k, a + i), _mm512_maskz_loadu_ps(k, b + i), acc); }
    return _mm512_reduce_add_ps(acc);
}
static void axpy_f32_512(float* y, float a, const float* x, int n) {
    __m512 av = _mm512_set1_ps(a);
    int i = 0;
    for (; i + 16 <= n; i += 16) _mm512_storeu_ps(y + i, _mm512_fmadd_ps(av, _mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i)));
    for (; i < n; i++) y[i] += a * x[i];
}
static float dot_h_512(const float* a, const u16* b, int n) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(b + i))), acc);
    float s = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) s += a[i] * cvth(b[i]);
    return s;
}
static void axpy_h_512(float* y, float a, const u16* x, int n) {
    __m512 av = _mm512_set1_ps(a);
    int i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(y + i, _mm512_fmadd_ps(av, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(x + i))), _mm512_loadu_ps(y + i)));
    for (; i < n; i++) y[i] += a * cvth(x[i]);
}

void kernels_avx512(Kernels* k, int vnni) {
    mm_fallback = k->matmat;   // whatever was installed before (AVX2 / AVX-VNNI)
    k->matmat = vnni ? matmat_v : matmat_b;
    k->dot = dot_f32_512; k->axpy = axpy_f32_512; k->dot_h = dot_h_512; k->axpy_h = axpy_h_512;
}
