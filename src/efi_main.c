// metallm: a single UEFI application that chats with a small LLM. No OS, no libc.
// The firmware provides: memory pages, the USB keyboard (ConIn), the framebuffer (GOP),
// FAT32 file access, and (if present) MP services to run our worker loop on other cores.
#include "uefi.h"
#include "engine.h"
#include "console.h"
#include "cpu.h"
#include "fs.h"
#include "md.h"
#include "settings.h"

static EFI_SYSTEM_TABLE* ST;
static EFI_BOOT_SERVICES* BS;
static EFI_GOP* gop;
static EFI_HANDLE image;
static Settings S;
static int use_avx2;

// ---------------- string building ----------------
typedef struct { char* b; int n, cap; } Str;
#define STR(name, size) char name##_buf[size]; Str name = {name##_buf, 0, size}; name##_buf[0] = 0
static void s_s(Str* s, const char* t) { while (*t && s->n < s->cap - 1) s->b[s->n++] = *t++; s->b[s->n] = 0; }
static void s_c(Str* s, char c) { char t[2] = {c, 0}; s_s(s, t); }
static void s_i(Str* s, i64 v) {
    char t[24]; int n = 0;
    if (v < 0) { s_c(s, '-'); v = -v; }
    do t[n++] = '0' + v % 10; while ((v /= 10));
    while (n) s_c(s, t[--n]);
}
static void s_f(Str* s, float v, int d) {
    if (v < 0) { s_c(s, '-'); v = -v; }
    i64 m = 1; for (int i = 0; i < d; i++) m *= 10;
    i64 x = (i64)(v * m + 0.5f);
    s_i(s, x / m);
    if (d) { s_c(s, '.'); i64 f = x % m; for (i64 k = m / 10; k > 0; k /= 10) s_c(s, '0' + (f / k) % 10); }
}
static void s_bytes(Str* s, u64 b) {
    if (b >= (1ull << 30)) { s_f(s, (float)(i64)b / 1073741824.0f, 1); s_s(s, " GiB"); }
    else { s_i(s, (i64)(b >> 20)); s_s(s, " MiB"); }
}
static void s_mhz(Str* s, u32 mhz) {
    if (!mhz) { s_s(s, "?"); return; }
    s_f(s, mhz / 1000.0f, 2); s_s(s, " GHz");
}

// ---------------- multicore ----------------
// Workers spin on job.gen. A parfor cuts n items into several chunks per active slot
// (slot[w] >= 0) and the active cores take chunks from a shared counter until none are left, so
// a slower core (an E-core on Intel hybrid CPUs, an SMT sibling, a core at a lower clock) just
// takes fewer chunks instead of holding everyone up at the end.
// A broadcast runs fn once on every started core (used for per-core settings like clocks).
#define CHUNKS_PER_THREAD 8
static struct {
    volatile task_fn fn; void* volatile ctx; volatile int n, nchunks;
    volatile u32 gen, done, ready; volatile int bcast;
    volatile int next __attribute__((aligned(64)));   // next chunk to take (own cache line)
    int nthreads, ncores;
    volatile int slot[MAX_THREADS];
} job;
static int g_boost_start;

static void run_chunks(void) {
    int n = job.n, nc = job.nchunks;
    for (;;) {
        int c = __atomic_fetch_add(&job.next, 1, __ATOMIC_RELAXED);
        if (c >= nc) break;
        int a = (int)((i64)n * c / nc), b = (int)((i64)n * (c + 1) / nc);
        if (a < b) job.fn(job.ctx, a, b);
    }
}
static void EFIAPI ap_worker(void* arg) {
    int w = (int)(usize)arg;
    cpu_core_init(w, g_boost_start);
    __atomic_fetch_add(&job.ready, 1, __ATOMIC_SEQ_CST);
    u32 seen = 0;
    for (;;) {
        u32 g;
        while ((g = __atomic_load_n(&job.gen, __ATOMIC_ACQUIRE)) == seen) cpu_pause();
        seen = g;
        if (job.bcast) job.fn(job.ctx, w, w + 1);
        else { if (job.slot[w] < 0) continue; run_chunks(); }
        __atomic_fetch_add(&job.done, 1, __ATOMIC_RELEASE);
    }
}
static void parfor_mp(task_fn fn, void* ctx, int n) {
    if (job.nthreads < 2 || n < job.nthreads * 2) { fn(ctx, 0, n); return; }
    int nc = job.nthreads * CHUNKS_PER_THREAD; if (nc > n) nc = n;
    job.fn = fn; job.ctx = ctx; job.n = n; job.nchunks = nc; job.bcast = 0;
    __atomic_store_n(&job.next, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&job.done, 0, __ATOMIC_RELAXED);
    __atomic_fetch_add(&job.gen, 1, __ATOMIC_RELEASE);
    run_chunks();
    while (__atomic_load_n(&job.done, __ATOMIC_ACQUIRE) < (u32)(job.nthreads - 1)) cpu_pause();
}
static void broadcast(task_fn fn, void* ctx) {
    if (job.ncores > 1) {
        job.fn = fn; job.ctx = ctx; job.bcast = 1;
        __atomic_store_n(&job.done, 0, __ATOMIC_RELAXED);
        __atomic_fetch_add(&job.gen, 1, __ATOMIC_RELEASE);
    }
    fn(ctx, 0, 1);
    if (job.ncores > 1) {
        while (__atomic_load_n(&job.done, __ATOMIC_ACQUIRE) < (u32)(job.ncores - 1)) cpu_pause();
        job.bcast = 0;
    }
}
static void boost_task(void* ctx, int w, int w1) { (void)w1; cpu_boost(w, (int)(usize)ctx); }

// Topology: physical core id = APIC id >> smt_shift. Order: boot core, the other first threads
// of each physical core, then their SMT siblings.
static int n_physical;
static int order[MAX_THREADS];
static void build_order(void) {
    int k = 0, seen[MAX_THREADS]; n_physical = 0;
    for (int pass = 0; pass < 2; pass++)
        for (int w = 0; w < job.ncores; w++) {
            u32 core = cpu.apic[w] >> cpu.smt_shift;
            int first = 1;
            for (int j = 0; j < w; j++) if ((cpu.apic[j] >> cpu.smt_shift) == core) first = 0;
            if (pass == 0 && first) { order[k++] = w; n_physical++; }
            if (pass == 1 && !first) order[k++] = w;
        }
    (void)seen;
}
static void set_threads(int n) {
    if (n < 1) n = 1;
    if (n > job.ncores) n = job.ncores;
    for (int w = 0; w < MAX_THREADS; w++) job.slot[w] = -1;
    for (int i = 0; i < n; i++) job.slot[order[i]] = i;
    job.nthreads = n;
}

static void* alloc_pages(u64 bytes);
// ---- fallback: wake the other cores ourselves (INIT-SIPI-SIPI), like an OS kernel does ----
extern u8 ap_tramp_start[], ap_tramp_end[], tramp_gdt[], tramp_pm32[], tramp_lm64[];
extern u8 tramp_gdtr_base[], tramp_far32[], tramp_far64[], tramp_cr3[], tramp_count[];
extern u8 tramp_stack_base[], tramp_stack_size[], tramp_entry[], tramp_max[];

static u64 rdmsr(u32 m) { u32 lo, hi; __asm__ volatile("rdmsr" : "=a"(lo), "=d"(hi) : "c"(m)); return (u64)hi << 32 | lo; }
static void wrmsr(u32 m, u64 v) { __asm__ volatile("wrmsr" ::"c"(m), "a"((u32)v), "d"((u32)(v >> 32))); }

static void EFIAPI sipi_entry(u64 idx) { ap_worker((void*)(usize)(idx + 1)); }

static void send_ipi(int x2, volatile u32* apic, u32 icr) {
    if (x2) { wrmsr(0x830, icr); return; }
    apic[0x310 / 4] = 0;
    apic[0x300 / 4] = icr;
    for (int i = 0; i < 100000 && (apic[0x300 / 4] & (1u << 12)); i++) cpu_pause();
}

static int sipi_start_cores(int max_threads) {
    // trampoline page below 1 MiB, page tables below 4 GiB (APs load CR3 in 32-bit mode)
    EFI_PHYSICAL_ADDRESS tp = 0x9F000, pt = 0xFFFFFFFF;
    if (BS->AllocatePages(AllocateMaxAddress, EfiLoaderData, 1, &tp) != EFI_SUCCESS) return 1;
    if (BS->AllocatePages(AllocateMaxAddress, EfiLoaderData, 18, &pt) != EFI_SUCCESS) return 1;
    u64* pml4 = (u64*)(usize)pt; u64* pdpt = pml4 + 512; u64* pd = pml4 + 1024;
    memset(pml4, 0, 18 * 4096);
    pml4[0] = (pt + 4096) | 3;
    for (int i = 0; i < 16; i++) pdpt[i] = (pt + 8192 + i * 4096ull) | 3;         // identity map 16 GiB
    for (u64 i = 0; i < 16 * 512; i++) pd[i] = (i << 21) | 0x83;                  // 2 MiB pages, P|RW|PS
    u64 stack_size = 64 * 1024;
    u8* stacks = alloc_pages(stack_size * (max_threads + 1));
    if (!stacks) return 1;

    u8* B = (u8*)(usize)tp;
    usize len = (usize)(ap_tramp_end - ap_tramp_start);
    memcpy(B, ap_tramp_start, len);
#define FIELD(sym) (B + ((sym) - ap_tramp_start))
    *(u32*)FIELD(tramp_gdtr_base) = (u32)(tp + (tramp_gdt - ap_tramp_start));
    *(u32*)FIELD(tramp_far32) = (u32)(tp + (tramp_pm32 - ap_tramp_start));
    *(u32*)FIELD(tramp_far64) = (u32)(tp + (tramp_lm64 - ap_tramp_start));
    *(u64*)FIELD(tramp_cr3) = pt;
    *(u64*)FIELD(tramp_count) = 0;
    *(u64*)FIELD(tramp_stack_base) = (u64)(usize)stacks;
    *(u64*)FIELD(tramp_stack_size) = stack_size;
    *(u64*)FIELD(tramp_entry) = (u64)(usize)sipi_entry;
    *(u64*)FIELD(tramp_max) = (u64)(max_threads - 1);  // worker w = index + 1 must stay < max_threads
    volatile u64* count = (volatile u64*)FIELD(tramp_count);
#undef FIELD
    __atomic_thread_fence(__ATOMIC_SEQ_CST);

    u64 apic_base = rdmsr(0x1B);
    int x2 = (apic_base >> 10) & 1;
    volatile u32* apic = (volatile u32*)(usize)(apic_base & 0xFFFFF000ull);
    u32 vec = (u32)(tp >> 12);
    job.nthreads = 1;
    send_ipi(x2, apic, 0x000C4500);            // INIT, all excluding self (also resets any half-started APs)
    BS->Stall(10000);
    __atomic_store_n(&job.ready, 0, __ATOMIC_SEQ_CST);
    send_ipi(x2, apic, 0x000C4600 | vec);      // SIPI
    BS->Stall(200);
    send_ipi(x2, apic, 0x000C4600 | vec);      // SIPI again (per spec)
    BS->Stall(100000);
    // wait for the cores that arrived to finish entering the worker loop
    // (cores past max_threads - 1 are counted but parked, so wait for at most that many)
    for (int t = 0; t < 1000; t++) {
        u64 arrived = *count; if (arrived > (u64)(max_threads - 1)) arrived = (u64)(max_threads - 1);
        if (__atomic_load_n(&job.ready, __ATOMIC_SEQ_CST) >= arrived) break;
        BS->Stall(1000);
    }
    int up = 1 + (int)__atomic_load_n(&job.ready, __ATOMIC_SEQ_CST);
    if (up > max_threads) up = max_threads;
    return up;
}

static int start_cores_mp(int max_threads) {
    EFI_MP* mp = 0;
    if (BS->LocateProtocol(&MP_GUID, 0, (void**)&mp) != EFI_SUCCESS || !mp) return 1;
    UINTN total = 0, enabled = 0, me = 0;
    if (mp->GetNumberOfProcessors(mp, &total, &enabled) != EFI_SUCCESS || enabled < 2) return 1;
    mp->WhoAmI(mp, &me);
    int want = 1;
    for (UINTN i = 0; i < total && want < max_threads; i++) {
        if (i == me) continue;
        EFI_PROCESSOR_INFORMATION info;
        if (mp->GetProcessorInfo(mp, i, &info) != EFI_SUCCESS || !(info.StatusFlag & 2)) continue;
        EFI_EVENT ev;
        if (BS->CreateEvent(0, 0, 0, 0, &ev) != EFI_SUCCESS) continue;
        if (mp->StartupThisAP(mp, ap_worker, i, ev, 0, (void*)(usize)want, 0) == EFI_SUCCESS) want++;
    }
    // wait (bounded) for workers to come up
    for (int t = 0; t < 2000 && __atomic_load_n(&job.ready, __ATOMIC_SEQ_CST) < (u32)(want - 1); t++) BS->Stall(1000);
    int up = 1 + (int)__atomic_load_n(&job.ready, __ATOMIC_SEQ_CST);
    return up == want ? up : -1;  // -1: some AP didn't start (stragglers would desync)
}

static const char* cores_how = "";
static int start_cores(void) {
    int n = 1;
    for (int w = 0; w < MAX_THREADS; w++) job.slot[w] = -1;
    job.nthreads = job.ncores = 1;
#ifndef FORCE_SIPI
    n = start_cores_mp(MAX_THREADS);
    if (n > 1) cores_how = "firmware MP services";
    if (n < 0) n = 1;
#endif
    if (n <= 1) {
        n = sipi_start_cores(MAX_THREADS);
        cores_how = n > 1 ? "started directly via APIC" : "only one core available";
    }
    job.ncores = n;
    g_parfor = parfor_mp;
    build_order();
    set_threads(n);
    return n;
}

// ---------------- display ----------------
static u32* shadow; static int scr_w, scr_h;
static u32* hist_buf; static usize hist_bytes;
static void blit(const u32* sh, int x, int y, int w, int h, int pitch) {
    gop->Blt(gop, (void*)(sh + (usize)y * pitch + x), EfiBltBufferToVideo, 0, 0, x, y, w, h, (UINTN)pitch * 4);
}
static void console_setup(void) {
    con_init(shadow, scr_w, scr_h, blit, S.font, S.theme);
    if (hist_buf) con_history(hist_buf, hist_bytes);
}

static void fatal_text(const char* msg) {  // used before the graphics console exists
    CHAR16 buf[256]; int i = 0;
    while (*msg && i < 254) buf[i++] = (CHAR16)(u8)*msg++;
    buf[i++] = '\r'; buf[i] = 0;
    ST->ConOut->OutputString(ST->ConOut, buf);
}
static void halt(const char* msg) {
    con_color(COL_ERR); con_puts(msg); con_puts("\n"); con_flush();
    for (;;) BS->Stall(1000000);
}

// ---------------- time ----------------
static u64 tsc_hz = 1;
static void calibrate_tsc(void) { u64 a = rdtsc(); BS->Stall(50000); tsc_hz = (rdtsc() - a) * 20; if (!tsc_hz) tsc_hz = 1; }
static float secs(u64 t) { return (float)t / (float)tsc_hz; }
static void s_date(Str* s) {
    EFI_TIME t;
    if (!ST->RuntimeServices || ST->RuntimeServices->GetTime(&t, 0) != EFI_SUCCESS || t.Year < 2000) return;
    s_i(s, t.Year); s_c(s, '-'); if (t.Month < 10) s_c(s, '0'); s_i(s, t.Month); s_c(s, '-');
    if (t.Day < 10) s_c(s, '0'); s_i(s, t.Day); s_c(s, ' ');
    if (t.Hour < 10) s_c(s, '0'); s_i(s, t.Hour); s_c(s, ':'); if (t.Minute < 10) s_c(s, '0'); s_i(s, t.Minute);
}

// ---------------- memory ----------------
typedef struct { void* p; u64 pages; } Block;
static void* alloc_pages(u64 bytes) {
    // below 16 GiB: cores started via APIC only map that much
    EFI_PHYSICAL_ADDRESS a = (16ull << 30) - 1;
    if (BS->AllocatePages(AllocateMaxAddress, EfiLoaderData, (bytes + 4095) / 4096, &a) != EFI_SUCCESS) return 0;
    return (void*)(usize)a;
}
static Block alloc_block(u64 bytes) { Block b = {alloc_pages(bytes), (bytes + 4095) / 4096}; if (!b.p) b.pages = 0; return b; }
static void free_block(Block* b) { if (b->p) BS->FreePages((EFI_PHYSICAL_ADDRESS)(usize)b->p, b->pages); b->p = 0; b->pages = 0; }

static u64 ram_total, ram_free, ram_largest;  // free / largest: conventional memory below 16 GiB
static void mem_stats(void) {
    static u8 map[48 * 1024];
    UINTN size = sizeof map, key, dsz; u32 ver;
    ram_total = ram_free = ram_largest = 0;
    if (BS->GetMemoryMap(&size, map, &key, &dsz, &ver) != EFI_SUCCESS || dsz < sizeof(EFI_MEMORY_DESCRIPTOR)) return;
    u64 lim = 16ull << 30;
    for (UINTN o = 0; o + dsz <= size; o += dsz) {
        EFI_MEMORY_DESCRIPTOR* d = (EFI_MEMORY_DESCRIPTOR*)(map + o);
        u64 b = d->NumberOfPages * 4096, s = d->PhysicalStart;
        if ((d->Type >= 1 && d->Type <= 7) || d->Type == 9 || d->Type == 10) ram_total += b;
        if (d->Type != EfiConventionalMemory || s >= lim) continue;
        if (s + b > lim) b = lim - s;
        ram_free += b;
        if (b > ram_largest) ram_largest = b;
    }
}

// ---------------- model files ----------------
// Every *.BIN in the root of a FAT volume is a model. Files over 4 GiB are split for FAT32:
// NAME.BIN is part 0, then NAME.B01, NAME.B02, ... (plain byte split).
#define MAX_MODELS 9
typedef struct { EFI_FILE* root; char file[32]; char name[64]; u64 size; int parts; u8 hdr[256]; } Model;
static Model models[MAX_MODELS];
static int n_models, cur_model = -1;

static void wname(const char* s, CHAR16* w) { int i = 0; w[i++] = '\\'; while (*s && i < 62) w[i++] = (u8)*s++; w[i] = 0; }
static u64 file_size(EFI_FILE* f) {
    u8 info[512]; UINTN n = sizeof info;
    if (f->GetInfo(f, &FILE_INFO_GUID, &n, info) != EFI_SUCCESS) return 0;
    return *(u64*)(info + FILE_INFO_FILESIZE);
}
static void part_name(const Model* m, int k, char* out) {  // k >= 1: NAME.Bkk
    int dot = 0; for (int j = 0; m->file[j]; j++) if (m->file[j] == '.') dot = j;
    int i = 0; for (; i < dot; i++) out[i] = m->file[i];
    out[i++] = '.'; out[i++] = 'B'; out[i++] = '0' + k / 10; out[i++] = '0' + k % 10; out[i] = 0;
}
static EFI_FILE* open_part(const Model* m, int k) {
    char n[40]; CHAR16 w[64]; EFI_FILE* f = 0;
    if (k == 0) wname(m->file, w); else { part_name(m, k, n); wname(n, w); }
    return m->root->Open(m->root, &f, w, EFI_FILE_MODE_READ, 0) == EFI_SUCCESS ? f : 0;
}

static void scan_volume(EFI_HANDLE dev) {
    EFI_SFS* fs; EFI_FILE* root;
    if (BS->HandleProtocol(dev, &SFS_GUID, (void**)&fs) != EFI_SUCCESS) return;
    if (fs->OpenVolume(fs, &root) != EFI_SUCCESS) return;
    static u8 info[1024];
    for (;;) {
        UINTN sz = sizeof info;
        if (root->Read(root, &sz, info) != EFI_SUCCESS || sz == 0) break;
        if (*(u64*)(info + FILE_INFO_ATTR) & EFI_FILE_DIRECTORY) continue;
        const CHAR16* fn = (const CHAR16*)(info + FILE_INFO_NAME);
        int len = 0; while (fn[len] && len < 62) len++;
        if (len < 5 || len > 28 || fn[len - 4] != '.') continue;
        CHAR16 a = fn[len - 3] | 32, b = fn[len - 2] | 32, c = fn[len - 1] | 32;
        if (a != 'b' || b != 'i' || c != 'n' || n_models >= MAX_MODELS) continue;
        Model* m = &models[n_models];
        memset(m, 0, sizeof *m);
        m->root = root;
        for (int i = 0; i < len; i++) { CHAR16 ch = fn[i]; m->file[i] = (char)(ch >= 'a' && ch <= 'z' ? ch - 32 : ch < 128 ? ch : '_'); }
        for (int i = 0; i < len - 4; i++) m->name[i] = m->file[i];
        EFI_FILE* f = open_part(m, 0);
        if (!f) continue;
        UINTN hn = sizeof m->hdr;
        if (f->Read(f, &hn, m->hdr) != EFI_SUCCESS || hn != sizeof m->hdr || memcmp(m->hdr, "QWN3", 4)) { f->Close(f); continue; }
        m->size = file_size(f); f->Close(f);
        m->parts = 1;
        for (int k = 1; k < 100; k++) {
            EFI_FILE* p = open_part(m, k);
            if (!p) break;
            m->size += file_size(p); m->parts++;
            p->Close(p);
        }
        if (m->hdr[192]) {  // display name stored in the header
            int i = 0; for (; i < 63 && m->hdr[192 + i]; i++) m->name[i] = (char)m->hdr[192 + i];
            m->name[i] = 0;
        }
        n_models++;
    }
}
static EFI_FILE* boot_root;
static void find_models(void) {
    EFI_LOADED_IMAGE* li = 0;
    if (BS->HandleProtocol(image, &LOADED_IMAGE_GUID, (void**)&li) == EFI_SUCCESS) scan_volume(li->DeviceHandle);
    if (n_models) return;
    UINTN n = 0; EFI_HANDLE* hs = 0;
    if (BS->LocateHandleBuffer(ByProtocol, &SFS_GUID, 0, &n, &hs) != EFI_SUCCESS) return;
    for (UINTN i = 0; i < n; i++) scan_volume(hs[i]);
}
static void open_boot_volume(void) {
    EFI_LOADED_IMAGE* li = 0; EFI_SFS* fs = 0;
    if (BS->HandleProtocol(image, &LOADED_IMAGE_GUID, (void**)&li) != EFI_SUCCESS) return;
    if (BS->HandleProtocol(li->DeviceHandle, &SFS_GUID, (void**)&fs) != EFI_SUCCESS) return;
    if (fs->OpenVolume(fs, &boot_root) != EFI_SUCCESS) boot_root = 0;
    fs_init(BS, boot_root);
}

// ---------------- machine facts (picker, info, log) ----------------
static u64 bandwidth;              // bytes/s, all cores
static u32 mhz_boot, mhz_now;
static int threads_auto;           // what "auto" picked
static float bench_all, bench_phys;
static float last_tps; static int last_pos;
static int writable = -1;          // -1 unknown, 0 read-only, 1 ok

static u64 kv_min_bytes(const Model* m) { return eng_mem_required(m->hdr, 2048); }
static Block model_mem, arena_mem;
static int model_fits(const Model* m) {
    u64 held = model_mem.pages * 4096 + arena_mem.pages * 4096;
    u64 largest = ram_largest > model_mem.pages * 4096 ? ram_largest : model_mem.pages * 4096;
    return largest >= m->size && ram_free + held >= m->size + kv_min_bytes(m) + (32ull << 20);
}
static float model_est(const Model* m) { return bandwidth && m->size ? 0.8f * (float)(i64)bandwidth / (float)(i64)m->size : 0; }
static int default_model(void) {
    for (int i = 0; i < n_models; i++)
        if (S.model[0] && !memcmp(models[i].file, S.model, sizeof S.model) && model_fits(&models[i])) return i;
    int best = -1;
    for (int i = 0; i < n_models; i++)
        if (model_fits(&models[i]) && model_est(&models[i]) >= 8 && (best < 0 || models[i].size > models[best].size)) best = i;
    if (best >= 0) return best;
    for (int i = 0; i < n_models; i++) if (best < 0 || models[i].size < models[best].size) best = i;
    return best;
}

// ---------------- keyboard ----------------
#define SCAN_UP 0x01
#define SCAN_DOWN 0x02
#define SCAN_RIGHT 0x03
#define SCAN_LEFT 0x04
#define SCAN_HOME 0x05
#define SCAN_END 0x06
#define SCAN_PGUP 0x09
#define SCAN_PGDN 0x0A
#define SCAN_ESC 0x17
static EFI_SIMPLE_POINTER* ptrs[8]; static int n_ptrs;
static EFI_EVENT wait_events[9]; static int n_wait;
static void show_stats(float tps, int pos);

static void init_input(void) {
    wait_events[n_wait++] = ST->ConIn->WaitForKey;
    UINTN n = 0; EFI_HANDLE* hs = 0;
    if (BS->LocateHandleBuffer(ByProtocol, &POINTER_GUID, 0, &n, &hs) != EFI_SUCCESS) return;
    for (UINTN i = 0; i < n && n_ptrs < 8; i++) {
        EFI_SIMPLE_POINTER* p;
        if (BS->HandleProtocol(hs[i], &POINTER_GUID, (void**)&p) != EFI_SUCCESS || !p) continue;
        p->Reset(p, 0);
        ptrs[n_ptrs++] = p;
        if (p->WaitForInput) wait_events[n_wait++] = p->WaitForInput;
    }
}
static void scroll_view(int delta) {
    int before = con_scroll(0), after = con_scroll(delta);
    if (before != after) show_stats(last_tps, last_pos);
    con_flush();
}
static int wheel(void) {  // lines (+ = away from you = back)
    int lines = 0;
    for (int i = 0; i < n_ptrs; i++) {
        EFI_SIMPLE_POINTER_STATE st;
        if (ptrs[i]->GetState(ptrs[i], &st) != EFI_SUCCESS || !st.RelativeMovementZ) continue;
        u64 res = ptrs[i]->Mode && ptrs[i]->Mode->ResolutionZ ? ptrs[i]->Mode->ResolutionZ : 1;
        i64 z = st.RelativeMovementZ / (i64)res; if (!z) z = st.RelativeMovementZ > 0 ? 1 : -1;
        lines += (int)z * 3;
    }
    return lines;
}
// Scroll keys work everywhere in the chat (also while a reply is streaming). Returns 1 if k was one.
static int scroll_key(const EFI_INPUT_KEY* k) {
    int page = con_rows() - 2;
    switch (k->ScanCode) {
    case SCAN_UP: scroll_view(1); return 1;
    case SCAN_DOWN: scroll_view(-1); return 1;
    case SCAN_PGUP: scroll_view(page); return 1;
    case SCAN_PGDN: scroll_view(-page); return 1;
    case SCAN_HOME: scroll_view(1 << 30); return 1;
    case SCAN_END: scroll_view(-(1 << 30)); return 1;
    }
    return 0;
}
static int read_key(EFI_INPUT_KEY* k, int wait) {
    for (;;) {
        if (wait) { UINTN idx; BS->WaitForEvent(n_wait, wait_events, &idx); }
        int z = wheel(); if (z) scroll_view(z);
        if (ST->ConIn->ReadKeyStroke(ST->ConIn, k) == EFI_SUCCESS) {
            if (scroll_key(k)) return 0;  // caller redraws (cursor etc.)
            return 1;
        }
        if (!wait) return 0;
    }
}
// For menus: every key, nothing intercepted; the wheel moves the selection.
static void read_key_raw(EFI_INPUT_KEY* k) {
    for (;;) {
        UINTN idx; BS->WaitForEvent(n_wait, wait_events, &idx);
        int z = wheel();
        if (z) { k->UnicodeChar = 0; k->ScanCode = z > 0 ? SCAN_UP : SCAN_DOWN; return; }
        if (ST->ConIn->ReadKeyStroke(ST->ConIn, k) == EFI_SUCCESS) return;
    }
}
static int utf8_put(u8* o, u32 c) {
    if (c < 0x80) { o[0] = (u8)c; return 1; }
    if (c < 0x800) { o[0] = 0xC0 | c >> 6; o[1] = 0x80 | (c & 63); return 2; }
    o[0] = 0xE0 | c >> 12; o[1] = 0x80 | ((c >> 6) & 63); o[2] = 0x80 | (c & 63); return 3;
}
// Line editor with echo. Returns length (bytes, UTF-8).
static int read_line(u8* buf, int max) {
    int n = 0;
    ConMark m = con_mark();
    for (;;) {
        con_rewind(m); con_color(COL_USER);
        for (int i = 0; i < n; i++) con_putb(buf[i]);
        con_cursor(1); con_flush();
        EFI_INPUT_KEY k;
        if (!read_key(&k, 1)) continue;
        if (con_scroll(0)) { con_scroll(-(1 << 30)); show_stats(last_tps, last_pos); }  // typing jumps back down
        con_cursor(0);
        if (k.UnicodeChar == '\r' || k.UnicodeChar == '\n') { con_flush(); return n; }
        if (k.UnicodeChar == 8) { while (n > 0 && (buf[n - 1] & 0xC0) == 0x80) n--; if (n > 0) n--; continue; }
        if (k.ScanCode == SCAN_ESC) { n = 0; continue; }
        if (k.UnicodeChar >= 32 && k.UnicodeChar != 127 && n + 3 < max) n += utf8_put(buf + n, k.UnicodeChar);
    }
}

// ---------------- engine ----------------
static Engine E;
static int toks[8192];
static int threads_up = 1;
static const char* model_name = "";
static float t_load, t_init;
static u64 kv_bytes;

static void progress(const char* what, u64 done, u64 total) {
    static ConMark m; static int started; static u64 last;
    if (!started || done < last) { m = con_mark(); started = 1; }  // new load (also after a failed one)
    last = done;
    con_rewind(m);
    con_color(COL_DIM); con_puts(what); con_puts(" [");
    int width = con_cols() - 32; if (width > 40) width = 40; if (width < 10) width = 10;
    int filled = (int)(done * width / (total ? total : 1));
    con_color(COL_ACCENT);
    for (int i = 0; i < width; i++) con_puts(i < filled ? "█" : "░");
    con_color(COL_DIM); con_puts("] "); con_putn((i64)(done >> 20)); con_puts(" / "); con_putn((i64)(total >> 20)); con_puts(" MiB");
    con_flush();
    if (done >= total) { started = 0; con_puts("\n"); }
}

// Engine buffers + KV cache for the context setting (largest that fits when auto).
static const char* init_engine(void) {
    free_block(&arena_mem);
    const ModelHeader* h = (const ModelHeader*)model_mem.p;
    int max_ctx = 256; while (max_ctx * 2 <= (int)h->rope_len) max_ctx *= 2;
    int ctx = S.ctx ? S.ctx : max_ctx; if (ctx > max_ctx) ctx = max_ctx;
    for (; ctx >= 256; ctx /= 2) {
        usize need = eng_mem_required(model_mem.p, ctx);
        arena_mem = alloc_block(need);
        if (!arena_mem.p) continue;
        const char* err = eng_init(&E, model_mem.p, models[cur_model].size, ctx, arena_mem.p, need);
        if (err) { free_block(&arena_mem); return err; }
        kv_bytes = 4ull * E.n_full * E.max_seq * h->n_kv * h->head_dim;  // fp16 K and V, full-attention layers
        return 0;
    }
    return "not enough memory for the context";
}

// Loads model i (freeing the current one first). Returns 0 or an error message.
static const char* load_model(int i) {
    Model* m = &models[i];
    free_block(&arena_mem); free_block(&model_mem);
    cur_model = -1;
    u64 t0 = rdtsc();
    model_mem = alloc_block(m->size);
    if (!model_mem.p) return "not enough memory for this model";
    u8* dst = model_mem.p; u64 off = 0;
    for (int k = 0; k < m->parts; k++) {
        EFI_FILE* f = open_part(m, k);
        if (!f) { free_block(&model_mem); return "can't open the model file"; }
        for (;;) {
            UINTN chunk = m->size - off > (8u << 20) ? (8u << 20) : (UINTN)(m->size - off);
            if (!chunk) break;
            if (f->Read(f, &chunk, dst + off) != EFI_SUCCESS) { f->Close(f); free_block(&model_mem); return "read error"; }
            if (chunk == 0) break;  // end of this part
            off += chunk;
            progress("loading", off, m->size);
        }
        f->Close(f);
    }
    if (off != m->size) { free_block(&model_mem); return "model file shorter than expected"; }
    t_load = secs(rdtsc() - t0);
    cur_model = i;
    model_name = m->name;
    memcpy(S.model, m->file, sizeof S.model);
    t0 = rdtsc();
    const char* err = init_engine();
    t_init = secs(rdtsc() - t0);
    if (err) { free_block(&model_mem); cur_model = -1; }
    return err;
}

// Threads "auto": decode is memory bound, so one thread per physical core is often as fast as
// all SMT threads. Time a few real decode steps both ways and keep the cheaper setting.
static float time_decode(int threads) {
    set_threads(threads);
    eng_forward(&E, (int)E.h->tok_im_start, 0, 1);
    u64 t0 = rdtsc();
    for (int p = 1; p <= 3; p++) eng_forward(&E, (int)E.h->tok_im_start, p, 1);
    return secs(rdtsc() - t0) / 3;
}
static void pick_threads(void) {
    threads_auto = job.ncores;
    if (n_physical < job.ncores && n_physical > 0) {
        bench_all = time_decode(job.ncores);
        bench_phys = time_decode(n_physical);
        if (bench_phys <= bench_all * 1.02f) threads_auto = n_physical;
    }
    set_threads(S.threads ? S.threads : threads_auto);
    threads_up = job.nthreads;
}

static void set_boost(int on) {
    broadcast(boost_task, (void*)(usize)on);
    mhz_now = cpu_mhz(tsc_hz);
}

// ---------------- status bar ----------------
static void status(const char* right) { con_status(model_name, right); }
static void show_stats(float tps, int pos) {
    last_tps = tps; last_pos = pos;
    STR(sb, 160);
    int back = con_scroll(0);
    if (back) { s_s(&sb, "scrolled up "); s_i(&sb, back); s_s(&sb, ", End to return    "); }
    if (tps > 0) { s_f(&sb, tps, 1); s_s(&sb, " tok/s    "); }
    s_i(&sb, threads_up); s_s(&sb, threads_up == 1 ? " thread    " : " threads    ");
    s_s(&sb, "context "); s_i(&sb, pos); s_s(&sb, "/"); s_i(&sb, E.max_seq);
    status(sb.b); con_flush();
}

// ---------------- info + log ----------------
#define INFO_MAX 16
static char info_txt[INFO_MAX][2][96]; static int n_info;
static void info_add(const char* k, const char* v) {
    if (n_info >= INFO_MAX) return;
    int i = 0; for (; k[i] && i < 95; i++) info_txt[n_info][0][i] = k[i]; info_txt[n_info][0][i] = 0;
    for (i = 0; v[i] && i < 95; i++) info_txt[n_info][1][i] = v[i]; info_txt[n_info][1][i] = 0;
    n_info++;
}
static int isa_used;
static void build_info(void) {
    n_info = 0;
    STR(v, 96);
    info_add("CPU", cpu.brand[0] ? cpu.brand : cpu.vendor);
    v.n = 0; s_i(&v, n_physical); s_s(&v, n_physical == 1 ? " core, " : " cores, "); s_i(&v, job.ncores);
    s_s(&v, " threads, using "); s_i(&v, job.nthreads); info_add("Cores", v.b);
    v.n = 0; s_s(&v, cpu.avx2 ? "AVX2, FMA" : "no AVX2, slow path");
    if (isa_used & ISA_VNNI512) s_s(&v, ", AVX-512 VNNI");
    else if (isa_used & ISA_AVX512) s_s(&v, ", AVX-512");
    if (isa_used & ISA_AVXVNNI) s_s(&v, ", AVX-VNNI");
    info_add("Vector", v.b);
    v.n = 0; s_mhz(&v, mhz_boot); s_s(&v, " at boot, "); s_mhz(&v, mhz_now); s_s(&v, " now");
    if (S.boost && cpu.boost_how[0]) { s_s(&v, " ("); s_s(&v, cpu.boost_how); s_s(&v, ")"); }
    info_add("Clock", v.b);
    v.n = 0; s_bytes(&v, ram_total); s_s(&v, ", "); s_bytes(&v, ram_free); s_s(&v, " free"); info_add("Memory", v.b);
    v.n = 0; s_f(&v, (float)(i64)bandwidth / 1e9f, 1); s_s(&v, " GB/s read"); info_add("Bandwidth", v.b);
    if (cur_model >= 0) {
        v.n = 0; s_s(&v, models[cur_model].name); s_s(&v, ", "); s_bytes(&v, models[cur_model].size); info_add("Model", v.b);
        v.n = 0; s_i(&v, E.max_seq); s_s(&v, " tokens, "); s_bytes(&v, kv_bytes); s_s(&v, " cache"); info_add("Context", v.b);
    }
    v.n = 0; if (last_tps > 0) { s_f(&v, last_tps, 1); s_s(&v, " tok/s last reply"); } else s_s(&v, "no reply yet");
    info_add("Speed", v.b);
    v.n = 0; s_i(&v, scr_w); s_c(&v, 'x'); s_i(&v, scr_h); s_s(&v, ", "); s_i(&v, con_font_w()); s_c(&v, 'x'); s_i(&v, con_font_h()); s_s(&v, " font");
    info_add("Screen", v.b);
    info_add("USB", writable == 0 ? "read-only" : writable == 1 ? "writable" : "not written yet");
}

static void write_log(void) {
    static char lb[8192]; Str L = {lb, 0, sizeof lb}; lb[0] = 0;
    s_s(&L, "metallm log  "); s_date(&L); s_s(&L, "\n\n");
    build_info();
    for (int i = 0; i < n_info - 1; i++) { s_s(&L, info_txt[i][0]); s_s(&L, ": "); s_s(&L, info_txt[i][1]); s_s(&L, "\n"); }
    s_s(&L, "\nvendor "); s_s(&L, cpu.vendor); s_s(&L, ", family "); s_i(&L, cpu.family); s_s(&L, " model "); s_i(&L, cpu.model);
    s_s(&L, "\nfeatures:"); if (cpu.avx2) s_s(&L, " avx2"); if (cpu.fma) s_s(&L, " fma"); if (cpu.avx512) s_s(&L, " avx512");
    if (cpu.hwp) s_s(&L, " hwp"); if (cpu.eist) s_s(&L, " eist"); if (cpu.turbo) s_s(&L, " turbo"); if (cpu.epb) s_s(&L, " epb");
    if (cpu.aperf) s_s(&L, " aperf"); if (cpu.hwpstate) s_s(&L, " hwpstate"); if (cpu.cpb) s_s(&L, " cpb"); if (cpu.cppc) s_s(&L, " cppc");
    s_s(&L, "\nboost: "); s_s(&L, S.boost ? cpu.boost_how : "off"); s_s(&L, ", level "); s_i(&L, cpu.boost_ratio);
    s_s(&L, ", refused MSR accesses "); s_i(&L, cpu.msr_faults);
    s_s(&L, "\ncores: "); s_s(&L, cores_how); s_s(&L, ", smt shift "); s_i(&L, cpu.smt_shift); s_s(&L, ", apic ids");
    for (int w = 0; w < job.ncores; w++) { s_c(&L, ' '); s_i(&L, cpu.apic[w]); }
    s_s(&L, "\nthreads: auto = "); s_i(&L, threads_auto);
    if (bench_all > 0) { s_s(&L, " (decode step "); s_f(&L, bench_all * 1000, 1); s_s(&L, " ms all threads, "); s_f(&L, bench_phys * 1000, 1); s_s(&L, " ms one per core)"); }
    s_s(&L, ", setting "); if (S.threads) s_i(&L, S.threads); else s_s(&L, "auto");
    s_s(&L, "\ntsc: "); s_i(&L, (i64)(tsc_hz / 1000000)); s_s(&L, " MHz");
    s_s(&L, "\nmemory: largest free block "); s_bytes(&L, ram_largest);
    if (cur_model >= 0) {
        const Model* m = &models[cur_model];
        s_s(&L, "\nmodel file: "); s_s(&L, m->file); s_s(&L, ", "); s_i(&L, m->parts); s_s(&L, m->parts == 1 ? " part, " : " parts, ");
        s_i(&L, (i64)m->size); s_s(&L, " bytes");
        s_s(&L, "\nload: "); s_f(&L, t_load, 2); s_s(&L, " s ("); s_f(&L, t_load > 0 ? (float)(i64)m->size / t_load / 1048576.0f : 0, 0);
        s_s(&L, " MiB/s), init "); s_f(&L, t_init, 2); s_s(&L, " s");
    }
    for (int i = 0; i < n_models; i++) {
        s_s(&L, "\nmodel "); s_s(&L, models[i].file); s_s(&L, ": "); s_s(&L, models[i].name);
        s_s(&L, model_fits(&models[i]) ? ", fits" : ", too big"); s_s(&L, ", estimate "); s_f(&L, model_est(&models[i]), 1); s_s(&L, " tok/s");
    }
    s_s(&L, "\n");
    writable = fs_write("\\LOG.TXT", lb, L.n) == 0;
}

// ---------------- transcript ----------------
// The conversation as text (what goes into CHATS\CHATnnnn.TXT) plus where each turn starts,
// so the screen can be redrawn after a font change.
#define TR_CAP (1u << 20)
static char* tr; static u32 tr_n;
static struct { u32 off, len; u8 user; } seg[4096]; static int n_seg;
static char chat_path[40];
static void tr_add(const char* s, u32 n) { if (!tr) return; if (tr_n + n > TR_CAP) n = TR_CAP - tr_n; memcpy(tr + tr_n, s, n); tr_n += n; }
static void tr_new(void) {
    tr_n = 0; n_seg = 0; chat_path[0] = 0;
    STR(h, 160); s_s(&h, "metallm, "); s_s(&h, model_name);
    STR(d, 32); s_date(&d); if (d.n) { s_s(&h, ", "); s_s(&h, d.b); }
    s_s(&h, "\n\n");
    tr_add(h.b, h.n);
}
static void tr_turn(int user, const u8* s, u32 n) {
    if (n_seg >= 4096 || !tr) return;
    if (user) tr_add("> ", 2);
    seg[n_seg].off = tr_n; seg[n_seg].user = (u8)user;
    tr_add((const char*)s, n);
    seg[n_seg].len = tr_n - seg[n_seg].off;   // what actually fit in the buffer
    tr_add("\n\n", 2);
    n_seg++;
}
static void tr_drop_last_turn(void) {  // /retry: forget the last question + answer
    while (n_seg && !seg[n_seg - 1].user) n_seg--;
    if (n_seg) { n_seg--; tr_n = seg[n_seg].off - 2; }
}
static int save_chat(void) {
    if (!n_seg) return 0;
    if (!chat_path[0] && fs_next_name("\\CHATS", "CHAT", ".TXT", chat_path, sizeof chat_path)) { writable = 0; return 1; }
    int err = fs_write(chat_path, tr, tr_n);
    writable = !err;
    return err;
}
static void say(int color, const char* a, const char* b) {
    con_color(color); con_puts(a); if (b) con_puts(b); con_puts("\n");
}
static void write_failed(void) { say(COL_ERR, "not saved: ", fs_error()); }

static void replay(void) {  // redraw the conversation from the transcript
    for (int i = 0; i < n_seg; i++) {
        const char* s = tr + seg[i].off;
        if (seg[i].user) {
            con_color(COL_ACCENT); con_puts("\n> "); con_color(COL_USER);
            for (u32 k = 0; k < seg[i].len; k++) con_putb((u8)s[k]);
            con_puts("\n");
        } else {
            md_reset(COL_TEXT);
            for (u32 k = 0; k < seg[i].len; k++) md_putb((u8)s[k]);
            md_finish();
        }
    }
}

// ---------------- memory (system prompt) ----------------
static u8* memory_txt; static u64 memory_len;
static void load_memory(void) {
    fs_free(memory_txt); memory_txt = 0; memory_len = 0;
    if (fs_read("\\MEMORY.TXT", &memory_txt, &memory_len) != 0) { memory_txt = 0; memory_len = 0; }
    while (memory_len && (memory_txt[memory_len - 1] == '\n' || memory_txt[memory_len - 1] == '\r' || memory_txt[memory_len - 1] == ' ')) memory_len--;
    if (memory_len > 6000) memory_len = 6000;  // keep the prompt bounded
}
static void show_file(const char* path, const char* empty) {
    u8* d; u64 n;
    if (fs_read(path, &d, &n) != 0 || !n) { say(COL_DIM, empty, 0); fs_free(d); return; }
    con_color(COL_TEXT);
    for (u64 i = 0; i < n; i++) con_putb(d[i]);
    if (d[n - 1] != '\n') con_puts("\n");
    fs_free(d);
}
static void append_line(const char* path, const u8* text, int n, const char* done) {
    u8 buf[2100]; if (n > 2048) n = 2048;
    memcpy(buf, text, n); buf[n] = '\n';
    if (fs_append(path, buf, n + 1)) { writable = 0; write_failed(); return; }
    writable = 1;
    say(COL_DIM, done, 0);
}

static void save_settings(void) {
    char buf[1024]; int n = settings_format(&S, buf, sizeof buf);
    if (fs_write("\\SETTINGS.TXT", buf, n) == 0) writable = 1; else writable = 0;
}

// ---------------- model picker ----------------
// Full-screen list. Returns the chosen model, or -1 on Esc (when cancel is allowed).
static int pick_model(int cancel) {
    int sel = default_model();
    for (;;) {
        con_ov_clear();
        int r = 1, c = 2;
        con_ov_text(c, r, "Model", COL_HEAD, BG_NONE, FACE_BOLD); r += 2;
        int namew = 10;
        for (int i = 0; i < n_models; i++) { int l = 0; while (models[i].name[l]) l++; if (l > namew) namew = l; }
        for (int i = 0; i < n_models; i++, r++) {
            int fits = model_fits(&models[i]);
            int bg = i == sel ? BG_SEL : BG_NONE, col = fits ? COL_TEXT : COL_DIM;
            con_ov_fill(c - 1, r, namew + 44, bg);
            STR(t, 96);
            s_i(&t, i + 1);
            con_ov_text(c, r, t.b, fits ? COL_ACCENT : COL_DIM, bg, 0);
            con_ov_text(c + 3, r, models[i].name, col, bg, i == sel ? FACE_BOLD : 0);
            t.n = 0; s_bytes(&t, models[i].size);
            con_ov_text(c + 5 + namew, r, t.b, COL_DIM, bg, 0);
            t.n = 0;
            if (!fits) s_s(&t, "does not fit in memory");
            else if (bandwidth) { s_s(&t, "about "); s_i(&t, (i64)(model_est(&models[i]) + 0.5f)); s_s(&t, " tok/s"); }
            con_ov_text(c + 16 + namew, r, t.b, COL_DIM, bg, 0);
        }
        r++;
        con_ov_text(c, r, cancel ? "Up Down choose    Enter load    Esc back" : "Up Down choose    Enter load", COL_DIM, BG_NONE, 0);
        con_flush();
        EFI_INPUT_KEY k; read_key_raw(&k);
        if (k.ScanCode == SCAN_UP) sel = (sel + n_models - 1) % n_models;
        else if (k.ScanCode == SCAN_DOWN) sel = (sel + 1) % n_models;
        else if (k.ScanCode == SCAN_ESC && cancel) { con_ov_end(); return -1; }
        else if (k.UnicodeChar >= '1' && k.UnicodeChar < '1' + n_models) sel = k.UnicodeChar - '1';
        else if (k.UnicodeChar == '\r' && model_fits(&models[sel])) { con_ov_end(); return sel; }
        if (k.UnicodeChar >= '1' && k.UnicodeChar < '1' + n_models && model_fits(&models[sel])) { con_ov_end(); return sel; }
    }
}

// ---------------- settings menu ----------------
enum { I_HEAD_CHAT, I_TEMP, I_TOPP, I_TOPK, I_THINK, I_MAXLEN, I_MEMORY, I_AUTOSAVE,
       I_HEAD_MODEL, I_MODEL, I_CTX, I_HEAD_PERF, I_THREADS, I_BOOST,
       I_HEAD_SCREEN, I_FONT, I_THEME, I_INFO, N_ITEMS };
static const char* const item_label[N_ITEMS] = {
    "Chat", "Temperature", "Top-p", "Top-k", "Thinking", "Reply length limit", "Use MEMORY.TXT", "Autosave chats",
    "Model", "Loaded", "Context size", "Performance", "Threads", "CPU boost",
    "Screen", "Font size", "Theme", "Info"};
static int is_head(int i) { return i == I_HEAD_CHAT || i == I_HEAD_PERF || i == I_HEAD_SCREEN || i == I_HEAD_MODEL; }
static int item_enabled(int i) { return !is_head(i); }

static const int temps[] = {-1, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150};
static const int topps[] = {-1, 50, 60, 70, 80, 90, 95, 100};
static const int topks[] = {1, 5, 10, 20, 40, 64};  // the sampler keeps at most 64
static const int lens[] = {0, 256, 512, 1024, 2048, 4096, 8192};
static const int ctxs[] = {0, 2048, 4096, 8192, 16384, 32768, 65536};
static int step(const int* v, int n, int cur, int d) {
    int i = 0; while (i < n && v[i] != cur) i++;
    if (i == n) i = 0;
    i += d; if (i < 0) i = 0; if (i >= n) i = n - 1;
    return v[i];
}
#define STEP(arr, cur, d) step(arr, (int)(sizeof arr / sizeof arr[0]), cur, d)

static int pending_ctx_change, pending_model = -1;
static void item_value(int i, Str* v) {
    static const char* const fonts[] = {"small", "medium", "large"};
    static const char* const themes[] = {"dark", "light", "high contrast"};
    switch (i) {
    case I_TEMP: if (S.temp100 < 0) s_s(v, "auto"); else s_f(v, S.temp100 / 100.0f, 1); break;
    case I_TOPP: if (S.topp100 < 0) s_s(v, "auto"); else s_f(v, S.topp100 / 100.0f, 2); break;
    case I_TOPK: s_i(v, S.topk); break;
    case I_THINK: s_s(v, S.think ? "on" : "off"); break;
    case I_MAXLEN: if (S.max_reply) { s_i(v, S.max_reply); s_s(v, " tokens"); } else s_s(v, "none"); break;
    case I_MEMORY: s_s(v, S.memory ? "on" : "off"); break;
    case I_AUTOSAVE: s_s(v, S.autosave ? "on" : "off"); break;
    case I_CTX:
        if (S.ctx) { s_i(v, S.ctx); s_s(v, " tokens"); } else s_s(v, "largest that fits");
        if (pending_ctx_change) s_s(v, ", new chat");
        break;
    case I_THREADS: if (S.threads) s_i(v, S.threads); else { s_s(v, "auto ("); s_i(v, threads_auto); s_s(v, ")"); } break;
    case I_BOOST: s_s(v, S.boost ? "on" : "off"); break;
    case I_FONT: s_s(v, S.font < 0 ? "auto" : fonts[S.font]); break;
    case I_THEME: s_s(v, themes[S.theme]); break;
    case I_MODEL: s_s(v, cur_model >= 0 ? models[cur_model].name : "none"); break;
    case I_INFO: s_s(v, "Enter"); break;
    }
}

static void item_change(int i, int d) {
    switch (i) {
    case I_TEMP: S.temp100 = STEP(temps, S.temp100, d); break;
    case I_TOPP: S.topp100 = STEP(topps, S.topp100, d); break;
    case I_TOPK: S.topk = STEP(topks, S.topk, d); break;
    case I_THINK: S.think = !S.think; break;
    case I_MAXLEN: S.max_reply = STEP(lens, S.max_reply, d); break;
    case I_MEMORY: S.memory = !S.memory; break;
    case I_AUTOSAVE: S.autosave = !S.autosave; break;
    case I_CTX: S.ctx = STEP(ctxs, S.ctx, d); pending_ctx_change = 1; break;
    case I_THREADS:
        S.threads += d;
        if (S.threads < 0) S.threads = 0;
        if (S.threads > job.ncores) S.threads = job.ncores;
        set_threads(S.threads ? S.threads : threads_auto); threads_up = job.nthreads;
        break;
    case I_BOOST: S.boost = !S.boost; set_boost(S.boost); break;
    case I_FONT: S.font += d; if (S.font < -1) S.font = -1; if (S.font > 2) S.font = 2; break;
    case I_THEME: S.theme = (S.theme + (d < 0 ? NTHEMES - 1 : 1)) % NTHEMES; con_theme(S.theme); show_stats(last_tps, last_pos); break;
    }
}

static void draw_info(int c, int r) {
    build_info();
    con_ov_text(c, r, "metallm", COL_HEAD, BG_NONE, FACE_BOLD); r += 2;
    for (int i = 0; i < n_info; i++, r++) {
        con_ov_text(c, r, info_txt[i][0], COL_DIM, BG_NONE, 0);
        con_ov_text(c + 11, r, info_txt[i][1], COL_TEXT, BG_NONE, 0);
    }
}
static void info_page(void) {
    con_ov_clear();
    draw_info(2, 1);
    con_ov_text(2, n_info + 4, "Esc back", COL_DIM, BG_NONE, 0);
    con_flush();
    EFI_INPUT_KEY k; do read_key_raw(&k); while (k.ScanCode != SCAN_ESC && k.UnicodeChar != '\r');
}

static void settings_menu(void) {
    Settings before = S;
    int sel = I_TEMP, scroll = 0;
    pending_ctx_change = 0; pending_model = -1;
    for (;;) {
        int wide = con_cols() >= 110;
        int rows = con_rows() - 3;  // room for items (top margin + hint line)
        if (sel < scroll) scroll = sel;
        if (sel >= scroll + rows) scroll = sel - rows + 1;
        if (scroll > 0 && is_head(scroll - 1)) scroll--;
        con_ov_clear();
        int lw = 22, vw = 26;
        for (int i = scroll, r = 1; i < N_ITEMS && r <= rows; i++, r++) {
            if (i == I_INFO && wide) break;
            if (is_head(i)) {
                if (i != scroll) r++;
                if (r > rows) break;
                con_ov_text(2, r, item_label[i], COL_HEAD, BG_NONE, FACE_BOLD);
                continue;
            }
            int bg = i == sel ? BG_SEL : BG_NONE, col = item_enabled(i) ? COL_TEXT : COL_DIM;
            con_ov_fill(3, r, lw + vw + 2, bg);
            con_ov_text(4, r, item_label[i], col, bg, 0);
            STR(v, 64); item_value(i, &v);
            con_ov_text(4 + lw, r, v.b, item_enabled(i) ? COL_ACCENT : COL_DIM, bg, 0);
        }
        if (wide) draw_info(lw + vw + 10, 1);
        con_ov_text(2, con_rows() - 1, con_cols() >= 64 ? "Up Down choose    Left Right change    Enter open    Esc back"
                                                      : "Arrows choose/change  Esc back", COL_DIM, BG_NONE, 0);
        con_flush();

        EFI_INPUT_KEY k; read_key_raw(&k);
        int last = wide ? I_INFO - 1 : I_INFO;
        if (k.ScanCode == SCAN_ESC) break;
        if (k.ScanCode == SCAN_UP) { do sel = sel > I_TEMP ? sel - 1 : last; while (is_head(sel) || !item_enabled(sel)); }
        else if (k.ScanCode == SCAN_DOWN) { do sel = sel < last ? sel + 1 : I_TEMP; while (is_head(sel) || !item_enabled(sel)); }
        else if (k.ScanCode == SCAN_LEFT) item_change(sel, -1);
        else if (k.ScanCode == SCAN_RIGHT) item_change(sel, 1);
        else if (k.UnicodeChar == '\r' || k.UnicodeChar == ' ') {
            if (sel == I_MODEL) { int m = pick_model(1); if (m >= 0 && m != cur_model) { pending_model = m; break; } }
            else if (sel == I_INFO) info_page();
            else item_change(sel, 1);
        }
    }
    con_ov_end();
    if (S.ctx == before.ctx) pending_ctx_change = 0;
    if (memcmp(&before, &S, sizeof S) || pending_model >= 0) save_settings();
    if (S.font != before.font) {  // new cell size: redraw the conversation in it
        console_setup();
        replay();
    }
}

// ---------------- chat ----------------
static int streq(const u8* a, int n, const char* b) { int i = 0; for (; i < n && b[i]; i++) if (a[i] != (u8)b[i]) return 0; return i == n && !b[i]; }
static int starts(const u8* a, int n, const char* b) { int i = 0; for (; b[i]; i++) if (i >= n || a[i] != (u8)b[i]) return 0; return n == i || a[i] == ' '; }
// "0.7" -> 0.7f; returns -1 on junk
static float parse_float(const u8* s, int n) {
    float v = 0, scale = 0; int any = 0;
    for (int i = 0; i < n; i++) {
        if (s[i] == '.' && !scale) { scale = 1; continue; }
        if (s[i] < '0' || s[i] > '9') return -1;
        any = 1;
        if (scale) { scale *= 10; v += (s[i] - '0') / scale; } else v = v * 10 + (s[i] - '0');
    }
    return any ? v : -1;
}

static const char HELP[] =
    "/clear            new conversation\n"
    "/retry            new answer to your last message\n"
    "/think            step-by-step reasoning on/off\n"
    "/temp 0.3         randomness, 0 = always the same answer\n"
    "/settings         settings and system info\n"
    "/model            switch model\n"
    "/save             save this chat now\n"
    "/note text        add a line to NOTES.TXT\n"
    "/notes            show NOTES.TXT\n"
    "/remember text    add a line to MEMORY.TXT, the model reads it in new chats\n"
    "/memory           show MEMORY.TXT\n"
    "/forget           clear MEMORY.TXT\n"
    "Esc               stop a reply\n"
    "PgUp PgDn, arrows, mouse wheel   scroll\n"
    "Home End          top, bottom\n";

static int pos, first, autosave_failed;
static int last_n;  // length of the last user message, for /retry (0 = nothing to retry)
static void new_chat(void) { pos = 0; first = 1; last_n = 0; tr_new(); }

static void switch_model(int m) {
    int old = cur_model;
    con_color(COL_DIM); con_puts("\n");
    const char* err = load_model(m);
    if (err) {
        say(COL_ERR, err, 0);
        if (old >= 0 && old != m && (err = load_model(old))) halt(err);
    }
    show_stats(0, 0);
    new_chat();
    write_log();
}

static void chat(void) {
    u64 rng = rdtsc() | 1;
    static u8 line[2048], last[2048];
    int last_pos0 = 0, last_first = 1;
    new_chat();
    show_stats(0, 0);
    for (;;) {
        con_color(COL_ACCENT); con_puts("\n> "); con_flush();
        int n = read_line(line, sizeof line);
        con_puts("\n");
        if (n == 0) continue;
        if (streq(line, n, "/clear") || streq(line, n, "/reset")) { new_chat(); last_n = 0; con_clear(); show_stats(0, 0); continue; }
        if (streq(line, n, "/think")) { S.think = !S.think; save_settings(); say(COL_DIM, S.think ? "thinking on" : "thinking off", 0); continue; }
        if (starts(line, n, "/temp")) {
            if (n > 6) {
                float t = parse_float(line + 6, n - 6);
                if (t >= 0 && t <= 2) { S.temp100 = (int)(t * 100 + 0.5f); save_settings(); }
                else { say(COL_DIM, "use a number from 0 to 2", 0); continue; }
            }
            STR(v, 32); if (S.temp100 < 0) s_s(&v, "auto"); else s_f(&v, S.temp100 / 100.0f, 2);
            say(COL_DIM, "temperature ", v.b);
            continue;
        }
        if (streq(line, n, "/settings")) {
            settings_menu();
            if (pending_model >= 0) switch_model(pending_model);
            else if (pending_ctx_change) {
                const char* err = init_engine();
                if (err) halt(err);
                new_chat(); last_n = 0;
                STR(v, 64); s_s(&v, "context "); s_i(&v, E.max_seq); s_s(&v, " tokens, new conversation");
                say(COL_DIM, v.b, 0);
            }
            show_stats(last_tps, pos);
            continue;
        }
        if (streq(line, n, "/model")) {
            int m = pick_model(1);
            if (m >= 0 && m != cur_model) { switch_model(m); save_settings(); last_n = 0; }
            continue;
        }
        if (streq(line, n, "/save")) {
            if (!n_seg) { say(COL_DIM, "nothing to save yet", 0); continue; }
            if (save_chat()) write_failed(); else say(COL_DIM, "saved to ", chat_path);
            continue;
        }
        if (starts(line, n, "/note") && n > 6) { append_line("\\NOTES.TXT", line + 6, n - 6, "added to NOTES.TXT"); continue; }
        if (streq(line, n, "/notes")) { show_file("\\NOTES.TXT", "no notes yet"); continue; }
        if (starts(line, n, "/remember") && n > 10) {
            append_line("\\MEMORY.TXT", line + 10, n - 10, "added to MEMORY.TXT, used from the next conversation");
            load_memory();
            continue;
        }
        if (streq(line, n, "/memory")) { show_file("\\MEMORY.TXT", "MEMORY.TXT is empty"); continue; }
        if (streq(line, n, "/forget")) {
            if (!memory_len) { say(COL_DIM, "MEMORY.TXT is empty", 0); continue; }
            con_color(COL_TEXT); con_puts("clear MEMORY.TXT? y/n "); con_flush();
            EFI_INPUT_KEY k; do read_key_raw(&k); while (!k.UnicodeChar && k.ScanCode != SCAN_ESC);
            con_puts("\n");
            if ((k.UnicodeChar | 32) == 'y') {
                if (fs_delete("\\MEMORY.TXT")) write_failed(); else { load_memory(); say(COL_DIM, "MEMORY.TXT cleared", 0); }
            } else say(COL_DIM, "kept", 0);
            continue;
        }
        if (streq(line, n, "/retry")) {
            if (!last_n) { say(COL_DIM, "nothing to retry", 0); continue; }
            memcpy(line, last, last_n); n = last_n; pos = last_pos0; first = last_first;
            tr_drop_last_turn();
        } else if (line[0] == '/') {
            con_color(COL_DIM); con_puts(HELP);
            continue;
        }
        memcpy(last, line, n); last_n = n; last_pos0 = pos; last_first = first;

        const u8* sys = S.memory && memory_len ? memory_txt : 0; int sys_len = sys ? (int)memory_len : 0;
        int nt = eng_chat_turn(&E, sys, sys_len, line, n, first, S.think, toks, 8192);
        if (pos + nt + 256 >= E.max_seq) {
            say(COL_DIM, "context full, starting a new conversation", 0);
            pos = 0; first = 1; last_pos0 = 0; last_first = 1;
            nt = eng_chat_turn(&E, sys, sys_len, line, n, 1, S.think, toks, 8192);
            if (nt + 256 >= E.max_seq) { say(COL_ERR, "message too long", 0); continue; }
        }
        first = 0;
        tr_turn(1, line, n);
        // prefill
        float* lg = eng_forward_batch(&E, toks, nt, pos, 0);
        pos += nt;
        // generate
        float temp = S.temp100 >= 0 ? S.temp100 / 100.0f : S.think ? 0.6f : 0.7f;
        float top_p = S.topp100 >= 0 ? S.topp100 / 100.0f : S.think ? 0.95f : 0.8f;
        u64 t0 = rdtsc(); int gen = 0, thinking = E.think_open;   // the prompt may already open <think>
        u32 reply0 = tr_n;
        md_reset(thinking ? COL_DIM : COL_TEXT);
        for (;;) {
            int t = eng_sample(&E, lg, temp, top_p, S.topk, &rng);
            if (t == (int)E.h->tok_im_end || t == (int)E.h->tok_eot) break;
            if (t == (int)E.h->tok_think) { md_reset(COL_DIM); thinking = 1; }
            else if (t == (int)E.h->tok_think_end) { md_reset(COL_TEXT); thinking = 0; }
            else if (t < (int)E.h->tok_eot) {
                int len; const u8* b = eng_token_bytes(&E, t, &len);
                for (int i = 0; i < len; i++) md_putb(b[i]);
                if (!thinking) tr_add((const char*)b, (u32)len);
            }
            gen++;
            con_flush();
            if (gen % 8 == 0) show_stats(gen / secs(rdtsc() - t0), pos);
            EFI_INPUT_KEY k;
            if (read_key(&k, 0) && k.ScanCode == SCAN_ESC) { md_finish(); say(COL_DIM, "stopped", 0); break; }
            if (pos >= E.max_seq - 1) { md_finish(); say(COL_DIM, "context full", 0); break; }
            if (S.max_reply && gen >= S.max_reply) { md_finish(); say(COL_DIM, "reply length limit", 0); break; }
            lg = eng_forward(&E, t, pos++, 1);
        }
        md_finish();
        // the reply text went straight into the transcript; register it as a turn
        u32 rl = tr_n - reply0; tr_n = reply0;
        {
            static char tmp[TR_CAP / 4]; if (rl > sizeof tmp) rl = sizeof tmp;
            memcpy(tmp, tr + reply0, rl);
            while (rl && (tmp[rl - 1] == '\n' || tmp[rl - 1] == ' ')) rl--;
            u32 s0 = 0; while (s0 < rl && (tmp[s0] == '\n' || tmp[s0] == ' ')) s0++;
            tr_turn(0, (const u8*)tmp + s0, rl - s0);
        }
        show_stats(gen ? gen / secs(rdtsc() - t0) : 0, pos);
        if (S.autosave && !autosave_failed && save_chat()) { write_failed(); autosave_failed = 1; say(COL_DIM, "autosave is off until the next start", 0); }
    }
}

// ---------------- boot ----------------
EFI_STATUS EFIAPI efi_main(EFI_HANDLE img, EFI_SYSTEM_TABLE* st) {
    ST = st; BS = st->BootServices; image = img;
    BS->SetWatchdogTimer(0, 0, 0, 0);  // otherwise firmware reboots us after 5 minutes
    ST->ConIn->Reset(ST->ConIn, 0);

    if (BS->LocateProtocol(&GOP_GUID, 0, (void**)&gop) != EFI_SUCCESS || !gop) {
        fatal_text("No graphics output (GOP) found."); for (;;) BS->Stall(1000000);
    }
    scr_w = gop->Mode->Info->HorizontalResolution; scr_h = gop->Mode->Info->VerticalResolution;
    shadow = alloc_pages((u64)scr_w * scr_h * 4);
    if (!shadow) { fatal_text("Out of memory for screen buffer."); for (;;) BS->Stall(1000000); }

    open_boot_volume();
    settings_default(&S);
    { u8* d; u64 n; if (fs_read("\\SETTINGS.TXT", &d, &n) == 0) { settings_parse(&S, d, n); fs_free(d); } }

    hist_bytes = 4000ull * (scr_w / 10) * 4;  // 4000 lines of scrollback at the smallest font
    hist_buf = alloc_pages(hist_bytes);
    if (!hist_buf) hist_bytes = 0;
    console_setup();
    init_input();
    status("");
    con_flush();
    tr = alloc_pages(TR_CAP);

    calibrate_tsc();
    cpu_detect();
    use_avx2 = cpu.avx2;
    mhz_boot = cpu_mhz(tsc_hz);
    g_boost_start = S.boost;
    cpu_core_init(0, S.boost);
    eng_use_avx2(use_avx2);
    if (!use_avx2) say(COL_DIM, "no AVX2 on this CPU, this will be slow", 0);
    threads_up = start_cores();
    if (use_avx2) isa_used = eng_set_isa(eng_detect_isa());  // every core's XCR0 is set by now
    mhz_now = cpu_mhz(tsc_hz);

    {   // memory read bandwidth with every core, for the speed estimates
        Block b = alloc_block(256ull << 20);
        if (!b.p) b = alloc_block(64ull << 20);
        if (b.p) { bandwidth = cpu_bandwidth(b.p, b.pages * 4096, tsc_hz); free_block(&b); }
    }
    mem_stats();
    find_models();
    load_memory();
    if (!n_models) halt("No model found. Put a .BIN model file next to the EFI folder.");
    int pick = n_models > 1 ? pick_model(0) : 0;
    if (!model_fits(&models[pick])) say(COL_DIM, "this model may not fit in memory", 0);
    model_name = models[pick].name;
    status("");
    const char* err = load_model(pick);
    if (err) halt(err);
    pick_threads();
    mem_stats();
    write_log();
    con_clear();
    say(COL_DIM, "type / for commands", 0);
    chat();
    return EFI_SUCCESS;
}
