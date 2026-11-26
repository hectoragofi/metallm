// Framebuffer text console: anti-aliased glyphs, word wrap, scrolling, status bar, themes.
// Renders into a RAM shadow buffer (BGRx) and pushes dirty rectangles through a blit callback.
#include "console.h"
#include "font.h"

#define MAX_COLS 400
#define MAX_ROWS 200
#define NATTR (NCOLORS * NBG)

static u32* fb; static int W, H; static blit_fn blit;
static const Font* F; static int font_idx = -1;
static u8 glyphs[FONT_NGLYPHS * 40 * 20];   // unpacked coverage of the font in use
static int cols, rows, top, left;           // text area: `left`/`top` pixel offsets
static u32 grid[MAX_ROWS][MAX_COLS];        // glyph index | attr << 16, attr = color + NCOLORS * bg
static int cx, cy, cur_attr, cur_face;
static i64 scrolled;                        // total lines scrolled (for marks)
static int dx0, dy0, dx1, dy1;              // dirty rect (pixels)
static u32 lut[NATTR + 1][256];             // blended colors per alpha (+1: status bar)
static u32 utf_cp; static int utf_need;
static u32 wrap_cells[16]; static int wrap_n;
// scrollback: ring of lines that scrolled off the top, each `cols` cells wide
static u32* hist; static int hist_cap, hist_n, hist_head;
static int view;                            // how many lines the view is scrolled back (0 = live)

typedef struct { u32 bg[NBG], bar, fg[NCOLORS]; } Theme;
static const Theme themes[NTHEMES] = {
    // text user dim accent err status bold code math head
    {{0x1b1e24, 0x262a33, 0x33435a}, 0x2c313c,
     {0xdde3ec, 0x8fd0e0, 0x7d8696, 0xf0c674, 0xe06c75, 0xc8ced8, 0xffffff, 0xc5cbd6, 0xb4a7f0, 0xf0c674}},
    {{0xfafaf8, 0xeceef1, 0xcfe0f5}, 0xe3e5e8,
     {0x24292f, 0x0b6a8a, 0x6e7781, 0x9a5b00, 0xc0302e, 0x30353c, 0x000000, 0x2b3036, 0x5a3fb0, 0x8a4f00}},
    {{0x000000, 0x1c1c1c, 0x003a70}, 0x262626,
     {0xffffff, 0x40e0ff, 0xc8c8c8, 0xffe14d, 0xff6b6b, 0xffffff, 0xffffff, 0xffffff, 0xc9b8ff, 0xffe14d}},
};
static const Theme* T = &themes[0];

static u32 blend(u32 bg, u32 fg, int a) {
    u32 r = 0;
    for (int s = 0; s < 24; s += 8) {
        int b = (bg >> s) & 255, f = (fg >> s) & 255;
        // mild gamma boost so thin strokes don't wash out
        int aa = a == 0 ? 0 : 255 - (255 - a) * (255 - a) / 255;
        aa = (a * 2 + aa) / 3;
        r |= (u32)(b + (f - b) * aa / 255) << s;
    }
    return r;
}
static void build_luts(void) {
    for (int bgi = 0; bgi < NBG; bgi++)
        for (int c = 0; c < NCOLORS; c++)
            for (int a = 0; a < 256; a++) lut[c + NCOLORS * bgi][a] = blend(T->bg[bgi], T->fg[c], a);
    for (int a = 0; a < 256; a++) lut[NATTR][a] = blend(T->bar, T->fg[COL_STATUS], a);
}

static void dirty(int x0, int y0, int x1, int y1) {
    if (dx0 >= dx1) { dx0 = x0; dy0 = y0; dx1 = x1; dy1 = y1; return; }
    if (x0 < dx0) dx0 = x0; if (y0 < dy0) dy0 = y0; if (x1 > dx1) dx1 = x1; if (y1 > dy1) dy1 = y1;
}

static int glyph_index(u32 cp) {
    int lo = 0, hi = FONT_NGLYPHS - 1;
    while (lo <= hi) { int m = (lo + hi) / 2; if (font_codepoints[m] == cp) return m; if (font_codepoints[m] < cp) lo = m + 1; else hi = m - 1; }
    return FONT_FALLBACK_INDEX;
}
static int glyph_for(u32 cp, int face) {
    if (face && ((cp > 0x20 && cp < 0x7F) || (cp > 0xA0 && cp <= 0xFF)))
        return glyph_index(cp + (face == FACE_BOLD ? FONT_FACE_BOLD : FONT_FACE_ITALIC));
    return glyph_index(cp);
}
static const int SPACE = 0;  // font_codepoints[0] == ' '

static void unpack_font(void) {
    const u8* s = F->rle; usize n = F->rle_len, o = 0, max = (usize)FONT_NGLYPHS * F->w * F->h;
    for (usize i = 0; i < n && o < max; i++) {
        if (s[i]) glyphs[o++] = s[i];
        else { int run = s[++i]; while (run-- && o < max) glyphs[o++] = 0; }
    }
}

static void draw_glyph_px(int px, int py, int g, const u32* lt) {
    const u8* a = glyphs + (usize)g * F->w * F->h;
    for (int y = 0; y < F->h; y++) {
        u32* row = fb + (usize)(py + y) * W + px;
        for (int x = 0; x < F->w; x++) row[x] = lt[a[y * F->w + x]];
    }
}
static const u32* line_at(int r) {    // screen row r of the current view
    int l = hist_n - view + r;
    return l < hist_n ? hist + (usize)((hist_head + l) % hist_cap) * cols : grid[l - hist_n];
}
static void draw_at(int c, int r, u32 v) {
    int px = left + c * F->w, py = top + r * F->h;
    draw_glyph_px(px, py, v & 0xffff, lut[v >> 16]);
    dirty(px, py, px + F->w, py + F->h);
}
static void draw_cell(int c, int r) { if (!view) draw_at(c, r, grid[r][c]); }
static void redraw_all(void) {
    for (int r = 0; r < rows; r++) { const u32* l = line_at(r); for (int c = 0; c < cols; c++) draw_at(c, r, l[c]); }
}
static void push_hist(const u32* line) {
    if (!hist_cap) return;
    int slot;
    if (hist_n < hist_cap) slot = (hist_head + hist_n++) % hist_cap;
    else { slot = hist_head; hist_head = (hist_head + 1) % hist_cap; }
    memcpy(hist + (usize)slot * cols, line, 4ull * cols);
}
static void paint_bg(void) {
    for (usize i = 0; i < (usize)W * H; i++) fb[i] = T->bg[0];
    dirty(0, 0, W, H);
}

void con_init(u32* shadow, int width, int height, blit_fn b, int font, int theme) {
    fb = shadow; W = width; H = height; blit = b;
    if (font < 0 || font > 2) {
        font = 0;
        for (int i = 2; i >= 0; i--) if (width / fonts[i].w >= 100 && height / fonts[i].h >= 30) { font = i; break; }
    }
    while (font > 0 && (width / fonts[font].w < 40 || height / fonts[font].h < 12)) font--;
    if (font != font_idx) { F = &fonts[font]; font_idx = font; unpack_font(); }
    T = &themes[theme >= 0 && theme < NTHEMES ? theme : 0];
    cols = W / F->w - 2; if (cols > MAX_COLS) cols = MAX_COLS;  // one cell of margin each side
    left = (W - cols * F->w) / 2;
    top = F->h + F->h / 2;
    rows = (H - top) / F->h; if (rows > MAX_ROWS) rows = MAX_ROWS;
    build_luts();
    paint_bg();
    for (int r = 0; r < MAX_ROWS; r++) for (int c = 0; c < MAX_COLS; c++) grid[r][c] = SPACE;
    cx = cy = 0; cur_attr = COL_TEXT; cur_face = 0; wrap_n = 0; view = 0; hist_n = hist_head = 0; hist_cap = 0;
}
void con_theme(int theme) {
    T = &themes[theme >= 0 && theme < NTHEMES ? theme : 0];
    build_luts();
    paint_bg();
    redraw_all();
}
int con_font(void) { return font_idx; }
int con_font_w(void) { return F->w; }
int con_font_h(void) { return F->h; }

int con_cols(void) { return cols; }
int con_rows(void) { return rows; }
int con_col(void) { return cx; }
void con_history(u32* buf, usize bytes) { hist = buf; hist_cap = (int)(bytes / (4ull * cols)); hist_n = hist_head = 0; }

int con_scroll(int delta) {
    int v = view + delta;
    if (v > hist_n) v = hist_n;
    if (v < 0) v = 0;
    if (v != view) { view = v; redraw_all(); }
    return view;
}
void con_color(int c) { cur_attr = c; cur_face = 0; }
void con_style(int color, int bg, int face) { cur_attr = color + NCOLORS * bg; cur_face = face; }
u32 con_cell(u32 cp, int color, int bg, int face) { return (u32)glyph_for(cp, face) | (u32)(color + NCOLORS * bg) << 16; }
void con_wrap_prefix(const u32* cells, int n) {
    if (n > 16) n = 16;
    if (n > cols / 2) n = cols / 2;
    for (int i = 0; i < n; i++) wrap_cells[i] = cells[i];
    wrap_n = n;
}

void con_clear(void) {
    for (int r = 0; r <= cy && r < rows; r++) push_hist(grid[r]);
    view = 0;
    for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) grid[r][c] = SPACE;
    cx = cy = 0; scrolled += rows;
    redraw_all();
}

static void scroll(void) {
    push_hist(grid[0]);
    memmove(grid[0], grid[1], sizeof(grid[0]) * (rows - 1));
    for (int c = 0; c < MAX_COLS; c++) grid[rows - 1][c] = SPACE;
    scrolled++;
    if (view) { if (view < hist_n) view++; else redraw_all(); }  // keep a scrolled-back view still
    else redraw_all();
}

static void newline(void) {
    cx = 0;
    if (++cy >= rows) { cy = rows - 1; scroll(); }
}

static u32 blank(void) { return (u32)SPACE | (u32)(NCOLORS * (cur_attr / NCOLORS)) << 16; }

static void put_glyph(int g) {
    if (cx >= cols) {
        // word wrap: carry the unfinished word to the next line (after the wrap prefix)
        int k = cols - 1;
        while (k > 0 && (grid[cy][k] & 0xffff) != SPACE) k--;
        int carry = (k > 0 && cols - 1 - k < cols / 3 && g != SPACE) ? cols - 1 - k : 0;
        u32 tmp[MAX_COLS], bl = blank();
        for (int i = 0; i < carry; i++) { tmp[i] = grid[cy][k + 1 + i]; grid[cy][k + 1 + i] = bl; draw_cell(k + 1 + i, cy); }
        newline();
        for (int i = 0; i < wrap_n; i++) { grid[cy][i] = wrap_cells[i]; draw_cell(i, cy); }
        cx = wrap_n;
        if (g == SPACE && !carry) return;  // swallow the space that caused the wrap
        for (int i = 0; i < carry; i++) { grid[cy][cx] = tmp[i]; draw_cell(cx, cy); cx++; }
    }
    grid[cy][cx] = (u32)g | (u32)cur_attr << 16;
    draw_cell(cx, cy);
    cx++;
}

void con_put(u32 cp) {
    if (cp == '\n') { wrap_n = 0; newline(); return; }
    if (cp == '\r') return;
    if (cp == '\t') { do put_glyph(SPACE); while (cx % 4); return; }
    if (cp < 32) return;
    if (cp == 0x200B || cp == 0xFEFF || cp == 0x200D) return;
    put_glyph(glyph_for(cp, cur_face));
}

void con_fill_eol(void) {
    u32 bl = blank();
    for (int x = cx; x < cols; x++) if (grid[cy][x] != bl) { grid[cy][x] = bl; draw_cell(x, cy); }
}

void con_putb(u8 b) {
    if (utf_need) {
        if ((b & 0xC0) == 0x80) { utf_cp = utf_cp << 6 | (b & 0x3F); if (--utf_need == 0) con_put(utf_cp); return; }
        utf_need = 0; con_put(0xFFFD);
    }
    if (b < 0x80) con_put(b);
    else if ((b & 0xE0) == 0xC0) { utf_cp = b & 0x1F; utf_need = 1; }
    else if ((b & 0xF0) == 0xE0) { utf_cp = b & 0x0F; utf_need = 2; }
    else if ((b & 0xF8) == 0xF0) { utf_cp = b & 0x07; utf_need = 3; }
    else con_put(0xFFFD);
}

void con_puts(const char* s) { while (*s) con_putb((u8)*s++); }

void con_putn(i64 v) {
    char b[24]; int n = 0;
    if (v < 0) { con_putb('-'); v = -v; }
    do b[n++] = '0' + v % 10; while ((v /= 10));
    while (n) con_putb(b[--n]);
}
void con_putf(float v, int d) {
    if (v < 0) { con_putb('-'); v = -v; }
    i64 m = 1; for (int i = 0; i < d; i++) m *= 10;
    i64 x = (i64)(v * m + 0.5f);
    con_putn(x / m);
    if (d) { con_putb('.'); i64 f = x % m; for (i64 k = m / 10; k > 0; k /= 10) { con_putb('0' + (f / k) % 10); } }
}

void con_cursor(int on) {
    if (view || cy >= rows || cx >= cols) return;
    draw_cell(cx, cy);
    if (on) {
        int px = left + cx * F->w, py = top + cy * F->h;
        int th = F->h / 10 < 2 ? 2 : F->h / 10;
        for (int y = F->h - th - 1; y < F->h - 1; y++)
            for (int x = 1; x < F->w - 1; x++) fb[(usize)(py + y) * W + px + x] = T->fg[COL_ACCENT];
    }
}

ConMark con_mark(void) { ConMark m = {scrolled + cy, cx}; return m; }
int con_mark_visible(ConMark m) { return m.line - scrolled >= 0; }

void con_rewind(ConMark m) {
    i64 r = m.line - scrolled; int c = m.col;
    if (r < 0) { r = 0; c = 0; }
    if (r >= rows) { r = rows - 1; c = 0; }   // e.g. after a font change shrank the grid
    if (c >= cols) c = cols - 1;
    for (int y = (int)r; y < rows; y++)
        for (int x = (y == r ? c : 0); x < cols; x++)
            if (grid[y][x] != SPACE) { grid[y][x] = SPACE; draw_cell(x, y); }
    cy = (int)r; cx = c; wrap_n = 0;
}

// ---- overlay ----
void con_ov_clear(void) {
    for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) draw_at(c, r, SPACE);
}
int con_ov_text(int col, int row, const char* s, int color, int bg, int face) {
    if (row < 0 || row >= rows) return 0;
    int c = col; u32 attr = (u32)(color + NCOLORS * bg) << 16;
    while (*s && c < cols) {
        u8 ch = (u8)*s++; u32 cp = ch;
        if (ch >= 0xC0) {
            int n = ch >= 0xF0 ? 3 : ch >= 0xE0 ? 2 : 1; cp = ch & (0x3F >> n);
            while (n-- && *s) cp = cp << 6 | ((u8)*s++ & 0x3F);
        }
        if (c >= 0) draw_at(c, row, (u32)glyph_for(cp, face) | attr);
        c++;
    }
    return c - col;
}
void con_ov_fill(int col, int row, int n, int bg) {
    if (row < 0 || row >= rows) return;
    for (int c = col; c < col + n && c < cols; c++) if (c >= 0) draw_at(c, row, (u32)SPACE | (u32)(NCOLORS * bg) << 16);
}
void con_ov_end(void) { redraw_all(); }

// ---- status bar ----
static void bar_text(int x, const char* s) {
    int py = (top - F->h) / 2 - F->h / 4; if (py < 0) py = 0;
    while (*s && x + F->w <= W) {
        u8 ch = (u8)*s++; u32 cp = ch;
        if (ch >= 0xC0) {  // tiny UTF-8 decode for the bar
            int n = ch >= 0xF0 ? 3 : ch >= 0xE0 ? 2 : 1; cp = ch & (0x3F >> n);
            while (n-- && *s) cp = cp << 6 | ((u8)*s++ & 0x3F);
        }
        draw_glyph_px(x, py, glyph_index(cp), lut[NATTR]);
        x += F->w;
    }
}
static int u8len(const char* s) { int n = 0; while (*s) n += ((u8)*s++ & 0xC0) != 0x80; return n; }

void con_status(const char* l, const char* r) {
    int bh = top - F->h / 2;
    for (int y = 0; y < bh; y++) for (int x = 0; x < W; x++) fb[(usize)y * W + x] = T->bar;
    bar_text(F->w, l);
    // right text: right-aligned, but never left of the left text (bar_text clips at the edge)
    int xr = W - (u8len(r) + 1) * F->w, minx = (u8len(l) + 3) * F->w;
    bar_text(xr > minx ? xr : minx, r);
    dirty(0, 0, W, bh);
}

void con_flush(void) {
    if (dx0 >= dx1) return;
    if (dx0 < 0) dx0 = 0; if (dy0 < 0) dy0 = 0; if (dx1 > W) dx1 = W; if (dy1 > H) dy1 = H;
    blit(fb, dx0, dy0, dx1 - dx0, dy1 - dy0, W);
    dx0 = dx1 = 0;
}
