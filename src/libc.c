// Tiny replacements for the few libc symbols the compiler may emit calls to.
#include "common.h"

int _fltused = 0;  // required by MS-ABI toolchains when floating point is used

void* memcpy(void* d, const void* s, usize n) {
    u8* dp = d; const u8* sp = s;
    if (((usize)dp | (usize)sp | n) % 8 == 0) {
        u64* d8 = (u64*)dp; const u64* s8 = (const u64*)sp;
        for (usize i = 0; i < n / 8; i++) d8[i] = s8[i];
        return d;
    }
    while (n--) *dp++ = *sp++;
    return d;
}
void* memmove(void* d, const void* s, usize n) {
    u8* dp = d; const u8* sp = s;
    if (dp < sp) { while (n--) *dp++ = *sp++; }
    else { dp += n; sp += n; while (n--) *--dp = *--sp; }
    return d;
}
void* memset(void* d, int c, usize n) {
    u8* p = d;
    while (n--) *p++ = (u8)c;
    return d;
}
int memcmp(const void* a, const void* b, usize n) {
    const u8 *x = a, *y = b;
    for (usize i = 0; i < n; i++) if (x[i] != y[i]) return x[i] - y[i];
    return 0;
}

static inline float bits_f(u32 b) { float f; __builtin_memcpy(&f, &b, 4); return f; }

// expf: Cody-Waite range reduction + degree-6 polynomial (~1e-7 rel. error)
float m_expf(float x) {
    if (x > 88.72f) return bits_f(0x7f800000);
    if (x < -103.0f) return 0.0f;
    float t = x * 1.4426950408889634f;
    int n = (int)(t + (t >= 0 ? 0.5f : -0.5f));
    float r = x - (float)n * 0.693145751953125f;   // ln2 hi
    r = r - (float)n * 1.428606765330187e-06f;      // ln2 lo
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (1.0f / 6 + r * (1.0f / 24 + r * (1.0f / 120 + r * (1.0f / 720))))));
    if (n < -126) return p * bits_f((u32)(n + 127 + 64) << 23) * bits_f((127 - 64) << 23);
    return p * bits_f((u32)(n + 127) << 23);
}
