#pragma once
#include "common.h"

enum { COL_TEXT, COL_USER, COL_DIM, COL_ACCENT, COL_ERR, COL_STATUS, COL_BOLD, COL_CODE, COL_MATH, COL_HEAD, NCOLORS };
enum { BG_NONE, BG_CODE, BG_SEL, NBG };
enum { FACE_REG, FACE_BOLD, FACE_ITALIC };
enum { THEME_DARK, THEME_LIGHT, THEME_CONTRAST, NTHEMES };

typedef void (*blit_fn)(const u32* shadow, int x, int y, int w, int h, int pitch);

// font: 0 small, 1 medium, 2 large, -1 = pick from the screen size. Keeps the scrollback buffer
// pointer but empties it (call con_history again after a re-init).
void con_init(u32* shadow, int width, int height, blit_fn blit, int font, int theme);
void con_theme(int theme);              // recolor everything on screen
int con_font(void);                     // font index in use
int con_font_w(void);
int con_font_h(void);
void con_color(int c);                  // plain text in color c (no background, regular face)
void con_style(int color, int bg, int face);
void con_putb(u8 byte);                 // UTF-8 byte stream
void con_put(u32 cp);                   // one codepoint
void con_puts(const char* s);
void con_putn(i64 v);
void con_putf(float v, int decimals);
void con_fill_eol(void);                // paint the rest of the line in the current background
// Cells repeated at the start of each wrapped continuation line (list indent, quote bar).
// Cleared by the next '\n'.
u32 con_cell(u32 cp, int color, int bg, int face);
void con_wrap_prefix(const u32* cells, int n);
int con_col(void);
void con_flush(void);
void con_clear(void);
void con_cursor(int on);
void con_status(const char* left, const char* right);
int con_cols(void);
int con_rows(void);
// Scrollback: give the console a buffer for lines that scroll off the top.
void con_history(u32* buf, usize bytes);
// Move the view back (+) or forward (-) by lines. Returns lines scrolled back (0 = live).
int con_scroll(int delta);
// Marks let the caller rewrite text in place (handles wrapping + scrolling).
typedef struct { i64 line; int col; } ConMark;
ConMark con_mark(void);
int con_mark_visible(ConMark m);        // still on screen, so con_rewind can reach it
void con_rewind(ConMark m);
// Overlay: draw straight onto the text area without touching the conversation underneath.
// con_ov_end() puts the conversation back.
void con_ov_clear(void);
int con_ov_text(int col, int row, const char* utf8, int color, int bg, int face);  // returns columns used
void con_ov_fill(int col, int row, int n, int bg);
void con_ov_end(void);
