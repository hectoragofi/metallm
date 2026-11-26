// Test-only: a Windows thread pool for g_parfor, so qwen.dll benchmarks can run multi-threaded
// from Python. Same spin-and-split scheme as efi_main's parfor_mp. Built by build.sh into
// build/parfor_win.dll; qeng.set_threads(n) installs it.
typedef unsigned long long u64;
typedef unsigned int u32;
typedef void (*task_fn)(void* ctx, int i0, int i1);
__declspec(dllimport) void* __stdcall CreateThread(void*, u64, u32(__stdcall*)(void*), void*, u32, u32*);
__declspec(dllimport) void __stdcall Sleep(u32);

static struct { task_fn fn; void* ctx; int n, nthreads; volatile u32 gen, done; } job;
static int started;

static void run_chunk(int w) {
    int per = (job.n + job.nthreads - 1) / job.nthreads, i0 = w * per, i1 = i0 + per;
    if (i1 > job.n) i1 = job.n;
    if (i0 < i1) job.fn(job.ctx, i0, i1);
}
static u32 __stdcall worker(void* arg) {
    int w = (int)(u64)arg; u32 seen = 0;
    for (;;) {
        u32 g;
        u32 spins = 0;   // spin while busy, back off to Sleep(1) when idle so idle tests don't burn cores
        while ((g = __atomic_load_n(&job.gen, __ATOMIC_ACQUIRE)) == seen) { if (++spins > 2000000) Sleep(1); else __builtin_ia32_pause(); }
        seen = g;
        if (w >= job.nthreads) continue;
        run_chunk(w);
        __atomic_fetch_add(&job.done, 1, __ATOMIC_RELEASE);
    }
}
__declspec(dllexport) void parfor_win(task_fn fn, void* ctx, int n) {
    if (job.nthreads <= 1 || n < job.nthreads * 2) { fn(ctx, 0, n); return; }
    job.fn = fn; job.ctx = ctx; job.n = n;
    __atomic_store_n(&job.done, 0, __ATOMIC_RELAXED);
    __atomic_fetch_add(&job.gen, 1, __ATOMIC_RELEASE);
    run_chunk(0);
    while (__atomic_load_n(&job.done, __ATOMIC_ACQUIRE) < (u32)(job.nthreads - 1)) __builtin_ia32_pause();
}
// n threads total (caller + n-1 spinning workers); workers are created once, up to 64.
__declspec(dllexport) void parfor_win_threads(int n) {
    if (n < 1) n = 1; if (n > 64) n = 64;
    for (; started < n - 1; started++) CreateThread(0, 0, worker, (void*)(u64)(started + 1), 0, 0);
    job.nthreads = n;
}
int __stdcall _DllMainCRTStartup(void* h, u32 r, void* p) { (void)h; (void)r; (void)p; return 1; }
