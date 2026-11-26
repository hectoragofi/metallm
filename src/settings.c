#include "settings.h"
#include "console.h"

void settings_default(Settings* s) {
    memset(s, 0, sizeof *s);
    s->temp100 = -1; s->topp100 = -1; s->topk = 20; s->memory = 1; s->autosave = 1; s->boost = 1;
    s->font = -1; s->theme = THEME_DARK;
}

static int lower(int c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }
static int eq(const u8* a, int n, const char* b) {
    int i = 0; for (; i < n && b[i]; i++) if (lower(a[i]) != b[i]) return 0;
    return i == n && !b[i];
}
// "0.7" -> 70 (x100); "20" -> 2000. -1 on junk.
static int fixed100(const u8* s, int n) {
    int v = 0, frac = -1, any = 0;
    for (int i = 0; i < n; i++) {
        if (s[i] == '.' && frac < 0) { frac = 0; continue; }
        if (s[i] < '0' || s[i] > '9') return -1;
        any = 1;
        if (frac < 0) v = v * 10 + (s[i] - '0');
        else if (frac < 2) { v = v * 10 + (s[i] - '0'); frac++; }
    }
    if (!any || v > 10000000) return -1;
    if (frac < 0) frac = 0;
    while (frac++ < 2) v *= 10;
    return v;
}
static int onoff(const u8* v, int n) {
    if (eq(v, n, "on") || eq(v, n, "yes") || eq(v, n, "true") || eq(v, n, "1")) return 1;
    if (eq(v, n, "off") || eq(v, n, "no") || eq(v, n, "false") || eq(v, n, "0")) return 0;
    return -1;
}

static void set(Settings* s, const u8* k, int kn, const u8* v, int vn) {
    int b = onoff(v, vn), x = fixed100(v, vn), auto_ = eq(v, vn, "auto") || eq(v, vn, "default");
    if (eq(k, kn, "temperature")) { if (auto_) s->temp100 = -1; else if (x >= 0 && x <= 200) s->temp100 = x; }
    else if (eq(k, kn, "top_p")) { if (auto_) s->topp100 = -1; else if (x > 0 && x <= 100) s->topp100 = x; }
    else if (eq(k, kn, "top_k")) { if (x >= 100 && x <= 6400) s->topk = x / 100; }
    else if (eq(k, kn, "thinking")) { if (b >= 0) s->think = b; }
    else if (eq(k, kn, "max_reply")) { if (b == 0 || eq(v, vn, "none")) s->max_reply = 0; else if (x >= 1600) s->max_reply = x / 100; }
    else if (eq(k, kn, "memory")) { if (b >= 0) s->memory = b; }
    else if (eq(k, kn, "autosave")) { if (b >= 0) s->autosave = b; }
    else if (eq(k, kn, "context")) { if (auto_) s->ctx = 0; else if (x >= 25600) s->ctx = x / 100; }
    else if (eq(k, kn, "threads")) { if (auto_) s->threads = 0; else if (x >= 100 && x <= 6400) s->threads = x / 100; }
    else if (eq(k, kn, "boost")) { if (b >= 0) s->boost = b; }
    else if (eq(k, kn, "font")) {
        if (auto_) s->font = -1; else if (eq(v, vn, "small")) s->font = 0;
        else if (eq(v, vn, "medium")) s->font = 1; else if (eq(v, vn, "large")) s->font = 2;
    } else if (eq(k, kn, "theme")) {
        if (eq(v, vn, "dark")) s->theme = THEME_DARK; else if (eq(v, vn, "light")) s->theme = THEME_LIGHT;
        else if (eq(v, vn, "contrast") || eq(v, vn, "high-contrast")) s->theme = THEME_CONTRAST;
    } else if (eq(k, kn, "model")) {
        if (vn > 0 && vn < (int)sizeof s->model) { memcpy(s->model, v, vn); s->model[vn] = 0; }
    }
}

void settings_parse(Settings* s, const u8* t, u64 len) {
    u64 i = 0;
    while (i < len) {
        u64 e = i; while (e < len && t[e] != '\n') e++;
        u64 a = i, b = e;
        while (a < b && (t[a] == ' ' || t[a] == '\t' || t[a] == 0xEF || t[a] == 0xBB || t[a] == 0xBF)) a++;  // also a BOM
        while (b > a && (t[b - 1] == ' ' || t[b - 1] == '\t' || t[b - 1] == '\r')) b--;
        u64 q = a; while (q < b && t[q] != '=') q++;
        if (q < b && t[a] != '#') {
            u64 ke = q; while (ke > a && (t[ke - 1] == ' ' || t[ke - 1] == '\t')) ke--;
            u64 vs = q + 1; while (vs < b && (t[vs] == ' ' || t[vs] == '\t')) vs++;
            set(s, t + a, (int)(ke - a), t + vs, (int)(b - vs));
        }
        i = e + 1;
    }
}

typedef struct { char* b; int n, max; } Out;
static void o_s(Out* o, const char* s) { while (*s && o->n < o->max - 1) o->b[o->n++] = *s++; }
static void o_i(Out* o, int v) { char t[12]; int n = 0; do t[n++] = '0' + v % 10; while ((v /= 10)); while (n && o->n < o->max - 1) o->b[o->n++] = t[--n]; }
static void o_x100(Out* o, int v) { o_i(o, v / 100); o_s(o, "."); o_i(o, v / 10 % 10); o_i(o, v % 10); }
static void o_kv(Out* o, const char* k) { o_s(o, k); o_s(o, "="); }

int settings_format(const Settings* s, char* out, int max) {
    Out o = {out, 0, max};
    static const char* const fonts[] = {"small", "medium", "large"};
    static const char* const themes[] = {"dark", "light", "contrast"};
    o_kv(&o, "temperature"); if (s->temp100 < 0) o_s(&o, "auto"); else o_x100(&o, s->temp100); o_s(&o, "\n");
    o_kv(&o, "top_p"); if (s->topp100 < 0) o_s(&o, "auto"); else o_x100(&o, s->topp100); o_s(&o, "\n");
    o_kv(&o, "top_k"); o_i(&o, s->topk); o_s(&o, "\n");
    o_kv(&o, "thinking"); o_s(&o, s->think ? "on\n" : "off\n");
    o_kv(&o, "max_reply"); if (s->max_reply) o_i(&o, s->max_reply); else o_s(&o, "off"); o_s(&o, "\n");
    o_kv(&o, "memory"); o_s(&o, s->memory ? "on\n" : "off\n");
    o_kv(&o, "autosave"); o_s(&o, s->autosave ? "on\n" : "off\n");
    o_kv(&o, "context"); if (s->ctx) o_i(&o, s->ctx); else o_s(&o, "auto"); o_s(&o, "\n");
    o_kv(&o, "threads"); if (s->threads) o_i(&o, s->threads); else o_s(&o, "auto"); o_s(&o, "\n");
    o_kv(&o, "boost"); o_s(&o, s->boost ? "on\n" : "off\n");
    o_kv(&o, "font"); o_s(&o, s->font < 0 ? "auto" : fonts[s->font]); o_s(&o, "\n");
    o_kv(&o, "theme"); o_s(&o, themes[s->theme]); o_s(&o, "\n");
    o_kv(&o, "model"); o_s(&o, s->model); o_s(&o, "\n");
    out[o.n] = 0;
    return o.n;
}
