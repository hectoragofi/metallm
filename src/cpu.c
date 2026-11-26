// CPU features, clock control, topology and a memory bandwidth probe.
//
// Clocks: firmware without an OS often leaves the cores in a low P-state. With the boost on we
// ask for the highest performance level the CPU reports: Intel HWP (highest perf, EPP 0) or the
// legacy PERF_CTL ratio (max turbo), AMD P0 plus CPPC max perf. Every MSR access goes through a
// #GP guard, so a CPU or hypervisor that refuses one just reports it instead of crashing.
#include "cpu.h"

CpuInfo cpu;

void cpuid(u32 leaf, u32 sub, u32* a, u32* b, u32* c, u32* d) {
    __asm__ volatile("cpuid" : "=a"(*a), "=b"(*b), "=c"(*c), "=d"(*d) : "a"(leaf), "c"(sub));
}

// ---------------------------------------------------------------- #GP guard
// The handler skips a faulting rdmsr/wrmsr (2 bytes) and sets r8 = 1; anything else goes to the
// firmware's handler. msr_read/msr_write clear r8 before the instruction and test it after.
u64 gp_orig;
__asm__(
    ".text\n"
    ".globl gp_guard\n"
    "gp_guard:\n"
    "    push %rax\n"
    "    mov 16(%rsp), %rax\n"          // faulting RIP (above saved rax and the error code)
    "    cmpw $0x320F, (%rax)\n"        // rdmsr
    "    je 1f\n"
    "    cmpw $0x300F, (%rax)\n"        // wrmsr
    "    je 1f\n"
    "    pop %rax\n"
    "    jmp *gp_orig(%rip)\n"
    "1:  add $2, %rax\n"
    "    mov %rax, 16(%rsp)\n"
    "    pop %rax\n"
    "    mov $1, %r8d\n"
    "    add $8, %rsp\n"                // drop the error code
    "    iretq\n");
extern u8 gp_guard[];

typedef struct __attribute__((packed)) { u16 limit; u64 base; } DTR;
static u8 idt_fw[4096] __attribute__((aligned(16)));   // copy of the firmware IDT + our #GP entry
static u8 idt_own[4096] __attribute__((aligned(16)));  // for cores we started (own GDT): #GP only
static u16 fw_cs, own_cs; static volatile int idt_lock;

static void idt_set(u8* idt, int vec, u64 fn, u16 cs) {
    u8* e = idt + vec * 16;
    *(u16*)(e + 0) = (u16)fn; *(u16*)(e + 2) = cs; e[4] = 0; e[5] = 0x8E;
    *(u16*)(e + 6) = (u16)(fn >> 16); *(u32*)(e + 8) = (u32)(fn >> 32); *(u32*)(e + 12) = 0;
}
static void guard_install(int boot_core) {
    DTR d; u16 cs;
    __asm__ volatile("sidt %0" : "=m"(d));
    __asm__ volatile("mov %%cs, %0" : "=r"(cs));
    if (boot_core) {
        usize n = (usize)d.limit + 1; if (n > sizeof idt_fw) n = sizeof idt_fw;
        memcpy(idt_fw, (void*)(usize)d.base, n);
        u8* e = idt_fw + 13 * 16;
        gp_orig = (u64)*(u16*)(e + 0) | (u64)*(u16*)(e + 6) << 16 | (u64)*(u32*)(e + 8) << 32;
        idt_set(idt_fw, 13, (u64)(usize)gp_guard, cs);
        fw_cs = cs;
    }
    u8* t = idt_fw;
    if (cs != fw_cs) {
        while (__atomic_exchange_n(&idt_lock, 1, __ATOMIC_ACQUIRE)) cpu_pause();
        if (own_cs != cs) { memset(idt_own, 0, sizeof idt_own); idt_set(idt_own, 13, (u64)(usize)gp_guard, cs); own_cs = cs; }
        __atomic_store_n(&idt_lock, 0, __ATOMIC_RELEASE);
        t = idt_own;
    }
    DTR nd = {4095, (u64)(usize)t};
    __asm__ volatile("lidt %0" ::"m"(nd));
}

int msr_read(u32 m, u64* v) {
    u32 lo = 0, hi = 0; register u64 f __asm__("r8") = 0;
    __asm__ volatile("rdmsr" : "+a"(lo), "+d"(hi), "+r"(f) : "c"(m) : "memory");
    if (f) { __atomic_fetch_add(&cpu.msr_faults, 1, __ATOMIC_RELAXED); *v = 0; return 0; }
    *v = (u64)hi << 32 | lo;
    return 1;
}
int msr_write(u32 m, u64 v) {
    register u64 f __asm__("r8") = 0;
    __asm__ volatile("wrmsr" : "+r"(f) : "c"(m), "a"((u32)v), "d"((u32)(v >> 32)) : "memory");
    if (f) { __atomic_fetch_add(&cpu.msr_faults, 1, __ATOMIC_RELAXED); return 0; }
    return 1;
}

// ---------------------------------------------------------------- detection
static u32 apic_id(void) {
    u32 a, b, c, d;
    cpuid(0, 0, &a, &b, &c, &d);
    if (a >= 0xB) { cpuid(0xB, 0, &a, &b, &c, &d); if (b) return d; }
    cpuid(1, 0, &a, &b, &c, &d);
    return b >> 24;
}

void cpu_detect(void) {
    u32 a, b, c, d, maxl, maxe;
    cpuid(0, 0, &maxl, &b, &c, &d);
    memcpy(cpu.vendor, &b, 4); memcpy(cpu.vendor + 4, &d, 4); memcpy(cpu.vendor + 8, &c, 4); cpu.vendor[12] = 0;
    cpu.intel = !memcmp(cpu.vendor, "GenuineIntel", 12);
    cpu.amd = !memcmp(cpu.vendor, "AuthenticAMD", 12) || !memcmp(cpu.vendor, "HygonGenuine", 12);
    cpuid(0x80000000, 0, &maxe, &b, &c, &d);
    if (maxe >= 0x80000004) {
        u32* p = (u32*)cpu.brand;
        for (u32 l = 0; l < 3; l++) cpuid(0x80000002 + l, 0, p + l * 4, p + l * 4 + 1, p + l * 4 + 2, p + l * 4 + 3);
        cpu.brand[48] = 0;
        char* s = cpu.brand; while (*s == ' ') s++;   // some CPUs pad the front
        if (s != cpu.brand) memmove(cpu.brand, s, 49 - (s - cpu.brand));
    }
    cpuid(1, 0, &a, &b, &c, &d);
    int fam = (a >> 8) & 15, mod = (a >> 4) & 15;
    if (fam == 15) fam += (a >> 20) & 255;
    if (fam >= 6) mod |= ((a >> 16) & 15) << 4;
    cpu.family = fam; cpu.model = mod;
    int xsave = (c >> 26) & 1, avx = (c >> 28) & 1;
    cpu.fma = (c >> 12) & 1;
    cpu.eist = cpu.intel && ((c >> 7) & 1);
    if (maxl >= 7) {
        u32 b7, c7, d7; cpuid(7, 0, &a, &b7, &c7, &d7);
        cpu.avx2 = xsave && avx && cpu.fma && ((c >> 29) & 1) && ((b7 >> 5) & 1);
        if (xsave && maxl >= 0xD) {
            u32 xa; cpuid(0xD, 0, &xa, &b, &c, &d);
            cpu.avx512 = cpu.avx2 && ((b7 >> 16) & 1) && ((b7 >> 30) & 1) && (xa & 0xE0) == 0xE0;  // F + BW + ZMM state
        }
    }
    if (maxl >= 6) {
        cpuid(6, 0, &a, &b, &c, &d);
        cpu.turbo = cpu.intel && ((a >> 1) & 1);
        cpu.hwp = cpu.intel && ((a >> 7) & 1);
        cpu.aperf = c & 1;
        cpu.epb = cpu.intel && ((c >> 3) & 1);
    }
    if (cpu.amd && maxe >= 0x80000007) {
        cpuid(0x80000007, 0, &a, &b, &c, &d);
        cpu.hwpstate = (d >> 7) & 1; cpu.cpb = (d >> 9) & 1;
        if (maxe >= 0x80000008) { cpuid(0x80000008, 0, &a, &b, &c, &d); cpu.cppc = (b >> 27) & 1; }
    }
    // SMT: bits of the APIC id that number threads within a core
    cpu.smt_shift = 0;
    u32 leaf = maxl >= 0x1F ? 0x1F : maxl >= 0xB ? 0xB : 0;
    if (leaf) {
        cpuid(leaf, 0, &a, &b, &c, &d);
        if (((c >> 8) & 255) == 1 && b) cpu.smt_shift = a & 31;
    }
    if (!leaf && cpu.amd && maxe >= 0x8000001E) {
        cpuid(0x8000001E, 0, &a, &b, &c, &d);
        int tpc = ((b >> 8) & 255) + 1;
        while ((1 << cpu.smt_shift) < tpc) cpu.smt_shift++;
    }
    cpu.boost_how = "";
}

static void enable_vector_state(void) {
    if (!cpu.avx2) return;
    u64 cr4; __asm__ volatile("mov %%cr4, %0" : "=r"(cr4));
    cr4 |= (1ull << 18) | (1ull << 9) | (1ull << 10);  // OSXSAVE, OSFXSR, OSXMMEXCPT
    __asm__ volatile("mov %0, %%cr4" ::"r"(cr4));
    u32 lo, hi; __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    lo |= 7;                       // x87 | SSE | AVX
    if (cpu.avx512) lo |= 0xE0;    // opmask | ZMM_Hi256 | Hi16_ZMM
    __asm__ volatile("xsetbv" ::"a"(lo), "d"(hi), "c"(0));
}

// ---------------------------------------------------------------- clocks
static struct { u64 perf_ctl, hwp_req, epb, pstate, cppc_req; u8 have[5]; } saved[MAX_THREADS];

void cpu_boost(int w, int on) {
    if (w < 0 || w >= MAX_THREADS) return;
    u64 v;
    if (!on) {  // back to what firmware had set on this core
        if (saved[w].have[0]) msr_write(0x199, saved[w].perf_ctl);
        if (saved[w].have[1]) msr_write(0x774, saved[w].hwp_req);
        if (saved[w].have[2]) msr_write(0x1B0, saved[w].epb);
        if (saved[w].have[3]) msr_write(0xC0010062, saved[w].pstate);
        if (saved[w].have[4]) msr_write(0xC00102B3, saved[w].cppc_req);
        if (w == 0) cpu.boost_how = "off";
        return;
    }
    if (cpu.intel) {
        if (msr_read(0x1A0, &v)) {  // IA32_MISC_ENABLE: turbo allowed, SpeedStep on
            u64 nv = v & ~(1ull << 38);
            if (cpu.eist) nv |= 1ull << 16;
            if (nv != v) msr_write(0x1A0, nv);
        }
        if (cpu.hwp) {
            u64 caps;
            if (msr_write(0x770, 1) && msr_read(0x771, &caps) && msr_read(0x774, &v)) {
                if (!saved[w].have[1]) { saved[w].hwp_req = v; saved[w].have[1] = 1; }
                u64 hi = caps & 255;
                if (hi && msr_write(0x774, hi | hi << 8)) {  // min = max = highest, desired auto, EPP 0
                    cpu.boost_ratio = (int)hi; if (w == 0) cpu.boost_how = "HWP";
                }
            }
        } else if (cpu.eist) {
            u64 plat, trl; int ratio = 0;
            if (msr_read(0xCE, &plat)) ratio = (int)((plat >> 8) & 255);
            if (msr_read(0x1AD, &trl) && (int)(trl & 255) > ratio) ratio = (int)(trl & 255);
            if (ratio && msr_read(0x199, &v)) {
                if (!saved[w].have[0]) { saved[w].perf_ctl = v; saved[w].have[0] = 1; }
                if (msr_write(0x199, (v & ~0x10000FFFFull) | (u64)ratio << 8)) {
                    cpu.boost_ratio = ratio; if (w == 0) cpu.boost_how = "P-state";
                }
            }
        }
        if (cpu.epb && msr_read(0x1B0, &v)) {
            if (!saved[w].have[2]) { saved[w].epb = v; saved[w].have[2] = 1; }
            msr_write(0x1B0, v & ~15ull);
        }
    } else if (cpu.amd) {
        if (cpu.cpb && msr_read(0xC0010015, &v) && (v >> 25) & 1) msr_write(0xC0010015, v & ~(1ull << 25));
        if (cpu.hwpstate && msr_read(0xC0010062, &v)) {
            if (!saved[w].have[3]) { saved[w].pstate = v; saved[w].have[3] = 1; }
            if (msr_write(0xC0010062, 0) && w == 0) cpu.boost_how = "P0";
        }
        u64 en, cap;
        if (cpu.cppc && msr_read(0xC00102B1, &en) && (en & 1) && msr_read(0xC00102B0, &cap) && msr_read(0xC00102B3, &v)) {
            if (!saved[w].have[4]) { saved[w].cppc_req = v; saved[w].have[4] = 1; }
            u64 hi = (cap >> 24) & 255;
            if (hi && msr_write(0xC00102B3, hi | hi << 8)) { cpu.boost_ratio = (int)hi; if (w == 0) cpu.boost_how = "CPPC"; }
        }
    }
    if (w == 0 && !cpu.boost_how[0]) cpu.boost_how = "no clock control";
}

void cpu_core_init(int w, int boost) {
    guard_install(w == 0);
    enable_vector_state();
    if (w >= 0 && w < MAX_THREADS) cpu.apic[w] = apic_id();
    if (boost) cpu_boost(w, 1);}

// 100 dependent adds per iteration: one add per cycle on every x86 core.
static void add_chain(u64 iters) {
    u64 x = 0;
    __asm__ volatile("1:\n.rept 100\nadd $1, %0\n.endr\ndec %1\njnz 1b" : "+r"(x), "+r"(iters));
}
u32 cpu_mhz(u64 tsc_hz) {
    u64 t0 = rdtsc();
    while (rdtsc() - t0 < tsc_hz / 50) add_chain(1000);  // 20 ms warm-up so the clock can ramp
    u64 a0, m0, a1, m1;
    if (cpu.aperf && msr_read(0xE8, &a0) && msr_read(0xE7, &m0)) {
        t0 = rdtsc();
        while (rdtsc() - t0 < tsc_hz / 50) add_chain(1000);
        if (msr_read(0xE8, &a1) && msr_read(0xE7, &m1) && m1 > m0 && a1 > a0) {
            u64 mhz = (tsc_hz / 1000000) * (a1 - a0) / (m1 - m0);
            if (mhz > 100 && mhz < 10000) return (u32)mhz;
        }
    }
    u64 n = 20000;
    t0 = rdtsc(); add_chain(n); u64 dt = rdtsc() - t0;
    return dt ? (u32)(n * 100 * (tsc_hz / 1000) / dt / 1000) : 0;
}

// ---------------------------------------------------------------- bandwidth
#define BW_CHUNK (64 * 1024)
typedef long long v2i __attribute__((vector_size(16)));
static volatile v2i bw_sink[MAX_THREADS * 4];  // keeps the loads alive
static void bw_task(void* ctx, int i0, int i1) {
    v2i a0 = {0, 0}, a1 = a0, a2 = a0, a3 = a0;
    for (int c = i0; c < i1; c++) {
        const v2i* p = (const v2i*)((const u8*)ctx + (u64)c * BW_CHUNK);
        for (int k = 0; k < BW_CHUNK / 16; k += 4) { a0 += p[k]; a1 += p[k + 1]; a2 += p[k + 2]; a3 += p[k + 3]; }
    }
    bw_sink[(i0 & (MAX_THREADS - 1)) * 4] = a0 + a1 + a2 + a3;
}
u64 cpu_bandwidth(void* buf, u64 bytes, u64 tsc_hz) {
    int chunks = (int)(bytes / BW_CHUNK);
    g_parfor(bw_task, buf, chunks);  // touch once
    u64 t0 = rdtsc(), read = 0;
    do { g_parfor(bw_task, buf, chunks); read += (u64)chunks * BW_CHUNK; } while (rdtsc() - t0 < tsc_hz / 12);
    u64 dt = rdtsc() - t0;
    return dt ? read * (tsc_hz >> 10) / dt << 10 : 0;
}
