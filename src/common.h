// Freestanding basics shared by every translation unit. No libc anywhere.
#pragma once
typedef unsigned char u8;
typedef signed char i8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef int i32;
typedef unsigned long long u64;
typedef long long i64;
typedef unsigned long long usize;

#define NULL ((void*)0)

void* memcpy(void* d, const void* s, usize n);
void* memset(void* d, int c, usize n);
void* memmove(void* d, const void* s, usize n);
int memcmp(const void* a, const void* b, usize n);

float m_expf(float x);
static inline float m_sqrtf(float x) { return __builtin_sqrtf(x); }  // -fno-math-errno => sqrtss

// ---- parallel-for hook (platform supplies a threaded version) ----
typedef void (*task_fn)(void* ctx, int i0, int i1);
typedef void (*parfor_fn)(task_fn fn, void* ctx, int n);
extern parfor_fn g_parfor;
// ---- optional hook the engine calls between layers (the app polls keyboard/mouse there) ----
extern void (*g_poll)(void);
