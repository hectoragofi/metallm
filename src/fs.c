// Reading and writing small text files on the boot volume.
#include "fs.h"

static EFI_BOOT_SERVICES* BS;
static EFI_FILE* root;
static EFI_STATUS last;

#define EFI_WRITE_PROTECTED EFI_ERR(8)
#define EFI_VOLUME_FULL EFI_ERR(11)
#define EFI_NOT_FOUND EFI_ERR(14)
#define EFI_MEDIA_CHANGED EFI_ERR(13)

void fs_init(EFI_BOOT_SERVICES* bs, EFI_FILE* r) { BS = bs; root = r; }
int fs_ready(void) { return root != 0; }

const char* fs_error(void) {
    if (!root) return "no USB volume";
    if (last == EFI_WRITE_PROTECTED) return "the USB stick is read-only";
    if (last == EFI_VOLUME_FULL) return "the USB stick is full";
    if (last == EFI_NOT_FOUND) return "file not found";
    return "the USB stick could not be written";
}

static void wpath(const char* p, CHAR16* w) { int i = 0; while (p[i] && i < 126) { w[i] = (u8)p[i]; i++; } w[i] = 0; }

static EFI_FILE* open(const char* path, u64 mode, u64 attr) {
    if (!root) { last = EFI_NOT_FOUND; return 0; }
    CHAR16 w[128]; wpath(path, w);
    EFI_FILE* f = 0;
    last = root->Open(root, &f, w, mode, attr);
    return EFI_ERROR(last) ? 0 : f;
}

static u64 file_size(EFI_FILE* f) {
    u8 info[512]; UINTN n = sizeof info;
    if (f->GetInfo(f, &FILE_INFO_GUID, &n, info) != EFI_SUCCESS) return 0;
    return *(u64*)(info + FILE_INFO_FILESIZE);
}

int fs_read(const char* path, u8** data, u64* len) {
    *data = 0; *len = 0;
    EFI_FILE* f = open(path, EFI_FILE_MODE_READ, 0);
    if (!f) return -1;
    u64 size = file_size(f);
    u8* buf = 0;
    if (BS->AllocatePool(EfiLoaderData, size + 1, (void**)&buf) != EFI_SUCCESS) { f->Close(f); return 1; }
    UINTN n = size;
    last = size ? f->Read(f, &n, buf) : EFI_SUCCESS;
    f->Close(f);
    if (EFI_ERROR(last)) { BS->FreePool(buf); return 1; }
    buf[n] = 0;
    *data = buf; *len = n;
    return 0;
}
void fs_free(void* p) { if (p) BS->FreePool(p); }

static int write_all(EFI_FILE* f, const void* d, u64 len) {
    const u8* p = d;
    while (len) {
        UINTN n = len > (1u << 20) ? (1u << 20) : (UINTN)len;
        last = f->Write(f, &n, p);
        if (EFI_ERROR(last) || n == 0) { if (!EFI_ERROR(last)) last = EFI_VOLUME_FULL; return 1; }
        p += n; len -= n;
    }
    return 0;
}

int fs_write(const char* path, const void* d, u64 len) {
    EFI_FILE* f = open(path, EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE | EFI_FILE_MODE_CREATE, 0);
    if (!f) return 1;
    int err = write_all(f, d, len);
    if (!err) {
        // shrink if the old file was longer
        u8 info[512]; UINTN n = sizeof info;
        if (f->GetInfo(f, &FILE_INFO_GUID, &n, info) == EFI_SUCCESS && *(u64*)(info + FILE_INFO_FILESIZE) > len) {
            *(u64*)(info + FILE_INFO_FILESIZE) = len;
            if (f->SetInfo(f, &FILE_INFO_GUID, n, info) != EFI_SUCCESS) {
                f->Close(f); fs_delete(path);  // fall back to delete + recreate
                f = open(path, EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE | EFI_FILE_MODE_CREATE, 0);
                if (!f) return 1;
                err = write_all(f, d, len);
            }
        }
    }
    EFI_STATUS fl = f->Flush(f);
    f->Close(f);
    if (!err && EFI_ERROR(fl)) { last = fl; err = 1; }
    return err;
}

int fs_append(const char* path, const void* d, u64 len) {
    EFI_FILE* f = open(path, EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE | EFI_FILE_MODE_CREATE, 0);
    if (!f) return 1;
    last = f->SetPosition(f, ~0ull);  // end of file
    int err = EFI_ERROR(last) ? 1 : write_all(f, d, len);
    EFI_STATUS fl = f->Flush(f);
    f->Close(f);
    if (!err && EFI_ERROR(fl)) { last = fl; err = 1; }
    return err;
}

int fs_delete(const char* path) {
    EFI_FILE* f = open(path, EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE, 0);
    if (!f) return last == EFI_NOT_FOUND ? 0 : 1;
    last = f->Delete(f);  // also closes
    return last == EFI_SUCCESS ? 0 : 1;
}

int fs_mkdir(const char* path) {
    EFI_FILE* f = open(path, EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE | EFI_FILE_MODE_CREATE, EFI_FILE_DIRECTORY);
    if (!f) return 1;
    f->Close(f);
    return 0;
}

int fs_next_name(const char* dir, const char* prefix, const char* ext, char* out, int max) {
    if (fs_mkdir(dir)) return 1;
    EFI_FILE* d = open(dir, EFI_FILE_MODE_READ, 0);
    if (!d) return 1;
    int plen = 0; while (prefix[plen]) plen++;
    int best = 0;
    static u8 info[1024];
    for (;;) {
        UINTN sz = sizeof info;
        if (d->Read(d, &sz, info) != EFI_SUCCESS || sz == 0) break;
        const CHAR16* fn = (const CHAR16*)(info + FILE_INFO_NAME);
        int i = 0;
        while (i < plen && (fn[i] | 32) == (prefix[i] | 32)) i++;
        if (i < plen) continue;
        int v = 0, digits = 0;
        while (fn[i] >= '0' && fn[i] <= '9') { v = v * 10 + (fn[i] - '0'); i++; digits++; }
        if (digits && v > best) best = v;
    }
    d->Close(d);
    int n = best + 1, k = 0;
    for (const char* s = dir; *s && k < max - 1; s++) out[k++] = *s;
    if (k < max - 1) out[k++] = '\\';
    for (const char* s = prefix; *s && k < max - 1; s++) out[k++] = *s;
    char num[12]; int nd = 0; int v = n; do num[nd++] = '0' + v % 10; while ((v /= 10));
    while (nd < 4) num[nd++] = '0';
    while (nd && k < max - 1) out[k++] = num[--nd];
    for (const char* s = ext; *s && k < max - 1; s++) out[k++] = *s;
    out[k] = 0;
    return 0;
}
