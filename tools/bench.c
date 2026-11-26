// Test-only (linked into qwen.dll, not the EFI app): kernel micro-benchmark and cross-check.
// bench_matmat(type, rows, cols, gs, n, asym, mem, out): fills `mem` (caller-allocated, see
// bench_mem) with pseudo-random weights/activations, runs g_k.matmat once into out[n*rows].
#include "engine.h"

static u64 rs = 88172645463325252ull;
static u32 rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (u32)rs; }
static int wbytes(int type, int cols) { return type == WT_Q8 ? cols : type == WT_Q4 ? cols / 2 : type == WT_Q1 ? cols / 8 : cols / 4; }

usize bench_mem(int type, int rows, int cols, int gs, int n) {
    return (usize)rows * wbytes(type, cols) + 4ull * rows * (cols / gs) + (usize)n * cols + 8ull * n * (cols / 32) + 2ull * n * cols + 256;
}
// setup=1: fill memory; then time the call from the caller with repeated setup=0 calls.
void bench_matmat(int type, int rows, int cols, int gs, int n, int asym, u8* mem, float* out, int setup) {
    usize wb = (usize)rows * wbytes(type, cols), ng = (usize)rows * (cols / gs);
    QMat m = {0};
    m.w = mem; m.s = (const u16*)(mem + wb); m.m = asym ? (const u16*)(mem + wb + 2 * ng) : 0;
    m.type = type; m.rows = rows; m.cols = cols; m.gs = gs;
    i8* q = (i8*)(mem + wb + 4 * ng);
    float* s = (float*)(q + (usize)n * cols); float* sum = s + (usize)n * (cols / 32);
    int* nks = (int*)(sum + (usize)n * (cols / 32)); float* xsx = (float*)(nks + (usize)n * (cols / 4));
    if (setup) {
        for (usize i = 0; i < wb; i++) mem[i] = (u8)rnd();
        if (type == WT_Q8) for (usize i = 0; i < wb; i++) if (mem[i] == 0x80) mem[i] = 0x81;
        u16* sc = (u16*)(mem + wb);
        for (usize i = 0; i < 2 * ng; i++) sc[i] = f2h(((int)(rnd() % 2001) - 1000) / 64000.0f);
        for (usize i = 0; i < (usize)n * cols; i++) { int v = (int)(rnd() % 255) - 127; q[i] = (i8)v; }
        for (usize b = 0; b < (usize)n * (cols / 32); b++) {
            s[b] = (rnd() % 1000 + 1) / 50000.0f; int t = 0;
            for (int i = 0; i < 32; i++) t += q[b * 32 + i];
            sum[b] = s[b] * t;
            for (int j = 0; j < 8; j++) { const i8* qj = q + b * 32 + 4 * j; nks[b * 8 + j] = -(qj[0] + qj[1] + qj[2] + qj[3]); xsx[b * 8 + j] = s[b]; }
        }
    }
    QAct a = {q, s, sum, nks, xsx};
    g_k.matmat(out, rows, &m, &a, n, cols, 0, rows);
}
