// CPU identification, clocks (turbo / P-states), topology and memory bandwidth.
#pragma once
#include "common.h"

#define MAX_THREADS 64

typedef struct {
    char vendor[13], brand[49];
    int intel, amd, family, model;
    int avx2, avx512, fma;
    int hwp, eist, turbo, epb, aperf;   // Intel power features (CPUID)
    int hwpstate, cpb, cppc;            // AMD
    int smt_shift;                      // APIC id >> smt_shift = physical core id
    u32 apic[MAX_THREADS];              // APIC id per worker slot (0 = this boot core)
    const char* boost_how;              // what cpu_boost did ("" if nothing applied)
    int boost_ratio;                    // requested multiplier / perf level (0 if unknown)
    int msr_faults;                     // MSR accesses the CPU refused (handled, not fatal)
} CpuInfo;
extern CpuInfo cpu;

void cpu_detect(void);                  // boot core, once
// Per core (boot core and every worker at start): exception guard for MSR probes, AVX state,
// clocks up if boost != 0, and records the core's APIC id in cpu.apic[w].
void cpu_core_init(int w, int boost);
void cpu_boost(int w, int on);          // this core: clocks to max (on) or back to what firmware set
u32 cpu_mhz(u64 tsc_hz);                // effective clock of this core under load, MHz
// Memory read bandwidth (bytes/s) with the current worker set; buf >= 64 MiB.
u64 cpu_bandwidth(void* buf, u64 bytes, u64 tsc_hz);
static inline u64 rdtsc(void) { u32 lo, hi; __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi)); return (u64)hi << 32 | lo; }
static inline void cpu_pause(void) { __asm__ volatile("pause" ::: "memory"); }
void cpuid(u32 leaf, u32 sub, u32* a, u32* b, u32* c, u32* d);
int msr_read(u32 m, u64* v);            // 1 = ok, 0 = the CPU refused (#GP caught)
int msr_write(u32 m, u64 v);
