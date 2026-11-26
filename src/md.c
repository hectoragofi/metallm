// Streaming markdown + LaTeX formatter. The model's output arrives a few bytes at a time; this
// decides line by line what each line is (heading, list, quote, rule, code fence, table row)
// from a short buffered prefix, then styles the rest inline (**bold**, *italic*, `code`) and
// turns $..$ / $$..$$ / \(..\) / \[..\] math into Unicode as soon as each piece is complete.
#include "md.h"
#include "console.h"

// ---------------------------------------------------------------- small helpers
static int is_alpha(u32 c) { return (c | 32) >= 'a' && (c | 32) <= 'z'; }
static int is_digit(u32 c) { return c >= '0' && c <= '9'; }
static int is_space(u32 c) { return c == ' ' || c == '\t'; }
static int is_punct(u32 c) { return c > 32 && c < 127 && !is_alpha(c) && !is_digit(c); }

static u32 utf8_next(const char** p, const char* e) {
    const u8* s = (const u8*)*p; u32 c = *s++; int n = 0;
    if (c >= 0xF0) { c &= 7; n = 3; } else if (c >= 0xE0) { c &= 15; n = 2; } else if (c >= 0xC0) { c &= 31; n = 1; }
    while (n-- && (const char*)s < e && (*s & 0xC0) == 0x80) c = c << 6 | (*s++ & 63);
    *p = (const char*)s;
    return c;
}
static int utf8_enc(char* o, u32 c) {
    if (c < 0x80) { o[0] = (char)c; return 1; }
    if (c < 0x800) { o[0] = (char)(0xC0 | c >> 6); o[1] = (char)(0x80 | (c & 63)); return 2; }
    if (c < 0x10000) { o[0] = (char)(0xE0 | c >> 12); o[1] = (char)(0x80 | ((c >> 6) & 63)); o[2] = (char)(0x80 | (c & 63)); return 3; }
    o[0] = (char)(0xF0 | c >> 18); o[1] = (char)(0x80 | ((c >> 12) & 63)); o[2] = (char)(0x80 | ((c >> 6) & 63)); o[3] = (char)(0x80 | (c & 63)); return 4;
}
static u32 first_cp(const char* s) { const char* p = s; return utf8_next(&p, s + 4); }

// ---------------------------------------------------------------- LaTeX to Unicode
// Converted math is a list of codepoints; bit 31 marks letters to draw in the italic face.
#define IT 0x80000000u
#define MO_MAX 256
typedef struct { u32 c[MO_MAX]; int n; } MOut;
static void mo_put(MOut* o, u32 c) {
    if ((c == ')' || c == ']') && o->n && o->c[o->n - 1] == ' ') o->n--;  // "( x )" -> "(x)"
    if (o->n < MO_MAX) o->c[o->n++] = c;
}
static void mo_cat(MOut* o, const MOut* a) { for (int i = 0; i < a->n; i++) mo_put(o, a->c[i]); }
static void mo_str(MOut* o, const char* s) { const char* e = s; while (*e) e++; while (s < e) mo_put(o, utf8_next(&s, e)); }

static const struct { const char* name; const char* out; } SYM[] = {
    {"alpha", "α"}, {"beta", "β"}, {"gamma", "γ"}, {"delta", "δ"}, {"epsilon", "ε"}, {"varepsilon", "ε"},
    {"zeta", "ζ"}, {"eta", "η"}, {"theta", "θ"}, {"vartheta", "ϑ"}, {"iota", "ι"}, {"kappa", "κ"},
    {"lambda", "λ"}, {"mu", "μ"}, {"nu", "ν"}, {"xi", "ξ"}, {"omicron", "ο"}, {"pi", "π"}, {"varpi", "ϖ"},
    {"rho", "ρ"}, {"varrho", "ϱ"}, {"sigma", "σ"}, {"varsigma", "ς"}, {"tau", "τ"}, {"upsilon", "υ"},
    {"phi", "φ"}, {"varphi", "φ"}, {"chi", "χ"}, {"psi", "ψ"}, {"omega", "ω"},
    {"Gamma", "Γ"}, {"Delta", "Δ"}, {"Theta", "Θ"}, {"Lambda", "Λ"}, {"Xi", "Ξ"}, {"Pi", "Π"},
    {"Sigma", "Σ"}, {"Upsilon", "Υ"}, {"Phi", "Φ"}, {"Psi", "Ψ"}, {"Omega", "Ω"},
    {"cdot", "·"}, {"cdotp", "·"}, {"times", "×"}, {"div", "÷"}, {"pm", "±"}, {"mp", "∓"},
    {"le", "≤"}, {"leq", "≤"}, {"leqslant", "≤"}, {"ge", "≥"}, {"geq", "≥"}, {"geqslant", "≥"},
    {"neq", "≠"}, {"ne", "≠"}, {"approx", "≈"}, {"equiv", "≡"}, {"sim", "∼"}, {"simeq", "≃"},
    {"cong", "≅"}, {"propto", "∝"}, {"infty", "∞"}, {"lt", "<"}, {"gt", ">"}, {"ll", "≪"}, {"gg", "≫"},
    {"sum", "∑"}, {"prod", "∏"}, {"coprod", "∐"}, {"int", "∫"}, {"iint", "∬"}, {"iiint", "∭"},
    {"oint", "∮"}, {"partial", "∂"}, {"nabla", "∇"},
    {"in", "∈"}, {"notin", "∉"}, {"ni", "∋"}, {"subset", "⊂"}, {"subseteq", "⊆"}, {"supset", "⊃"},
    {"supseteq", "⊇"}, {"cup", "∪"}, {"cap", "∩"}, {"bigcup", "⋃"}, {"bigcap", "⋂"}, {"setminus", "∖"},
    {"emptyset", "∅"}, {"varnothing", "∅"},
    {"forall", "∀"}, {"exists", "∃"}, {"nexists", "∄"}, {"neg", "¬"}, {"lnot", "¬"},
    {"land", "∧"}, {"wedge", "∧"}, {"lor", "∨"}, {"vee", "∨"},
    {"to", "→"}, {"rightarrow", "→"}, {"longrightarrow", "→"}, {"Rightarrow", "⇒"}, {"implies", "⇒"},
    {"Longrightarrow", "⇒"}, {"leftarrow", "←"}, {"gets", "←"}, {"longleftarrow", "←"},
    {"Leftarrow", "⇐"}, {"impliedby", "⇐"}, {"leftrightarrow", "↔"}, {"Leftrightarrow", "⇔"},
    {"iff", "⇔"}, {"mapsto", "↦"}, {"uparrow", "↑"}, {"downarrow", "↓"},
    {"ldots", "…"}, {"dots", "…"}, {"cdots", "⋯"}, {"vdots", "⋮"}, {"ddots", "⋱"},
    {"circ", "∘"}, {"degree", "°"}, {"angle", "∠"}, {"perp", "⊥"}, {"parallel", "∥"},
    {"mid", "|"}, {"vert", "|"}, {"Vert", "‖"}, {"prime", "′"}, {"star", "⋆"}, {"ast", "∗"},
    {"bullet", "•"}, {"oplus", "⊕"}, {"otimes", "⊗"}, {"hbar", "ℏ"}, {"ell", "ℓ"},
    {"Re", "ℜ"}, {"Im", "ℑ"}, {"aleph", "ℵ"}, {"therefore", "∴"}, {"because", "∵"},
    {"langle", "⟨"}, {"rangle", "⟩"}, {"lfloor", "⌊"}, {"rfloor", "⌋"}, {"lceil", "⌈"}, {"rceil", "⌉"},
    {"lbrace", "{"}, {"rbrace", "}"}, {"colon", ":"}, {"quad", "  "}, {"qquad", "    "},
    {"top", "⊤"}, {"bot", "⊥"}, {"vdash", "⊢"}, {"models", "⊨"}, {"triangle", "△"},
    {"dagger", "†"}, {"checkmark", "✓"}, {"bmod", " mod "}, {"mod", " mod "},
};
// dropped entirely (sizing and spacing hints)
static const char* const DROP[] = {
    "left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr",
    "Biggl", "Biggr", "middle", "displaystyle", "textstyle", "scriptstyle", "limits", "nolimits",
    "nonumber", "notag", "not",
};
// argument shown as plain text (spaces kept, no conversion)
static const char* const TEXTARG[] = {"text", "textrm", "textnormal", "mbox", "hbox", "textsf", "texttt", "textup"};
// argument converted but drawn upright
static const char* const UPRIGHT[] = {
    "mathrm", "mathbf", "mathit", "mathsf", "mathtt", "operatorname", "textbf", "textit", "emph",
    "boldsymbol", "bm", "mathcal", "mathscr", "mathfrak", "overline", "underline", "hat", "bar", "vec",
    "tilde", "dot", "ddot", "widehat", "widetilde", "overrightarrow", "overleftarrow", "cancel", "boxed",
};
// superscript / subscript forms (same order in each pair of strings)
static const char SUP_FROM[] = "0123456789+-−=()niabcdefghjklmoprstuvwxyzABDEGHIJKLMNOPRTUVWαβγδθφχ′";
static const char SUP_TO[]   = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾ⁿⁱᵃᵇᶜᵈᵉᶠᵍʰʲᵏˡᵐᵒᵖʳˢᵗᵘᵛʷˣʸᶻᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂᵅᵝᵞᵟᶿᵠᵡ′";
static const char SUB_FROM[] = "0123456789+-−=()aehijklmnoprstuvxβγρφχ";
static const char SUB_TO[]   = "₀₁₂₃₄₅₆₇₈₉₊₋₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓᵦᵧᵨᵩᵪ";
static u32 sup_f[96], sup_t[96], sub_f[64], sub_t[64]; static int n_sup, n_sub;
static int decode_list(const char* s, u32* out, int max) {
    const char* e = s; while (*e) e++;
    int n = 0; while (s < e && n < max) out[n++] = utf8_next(&s, e);
    return n;
}
static void init_scripts(void) {
    if (n_sup) return;
    int a = decode_list(SUP_FROM, sup_f, 96), b = decode_list(SUP_TO, sup_t, 96);
    n_sup = a == b ? a : 0;
    a = decode_list(SUB_FROM, sub_f, 64); b = decode_list(SUB_TO, sub_t, 64);
    n_sub = a == b ? a : 0;
    if (!n_sup) n_sup = -1;
}
static u32 script_of(u32 c, int sup) {
    const u32* f = sup ? sup_f : sub_f; const u32* t = sup ? sup_t : sub_t; int n = sup ? n_sup : n_sub;
    for (int i = 0; i < n; i++) if (f[i] == c) return t[i];
    return 0;
}
static int in_list(const char* const* l, int n, const char* s, int len) {
    for (int i = 0; i < n; i++) { int k = 0; while (k < len && l[i][k] == s[k]) k++; if (k == len && !l[i][k]) return 1; }
    return 0;
}
#define COUNT(a) (int)(sizeof(a) / sizeof((a)[0]))

typedef struct { const char* e; int depth; } MP;
static const char* m_seq(MP* m, const char* p, MOut* o, int until_brace);
static const char* m_atom(MP* m, const char* p, MOut* o);

static const char* skip_sp(const char* p, const char* e) { while (p < e && (*p == ' ' || *p == '\n' || *p == '\t')) p++; return p; }
// one argument: {group} or a single atom
static const char* m_arg(MP* m, const char* p, MOut* o) {
    p = skip_sp(p, m->e);
    if (p >= m->e) return p;
    if (*p == '{') return m_seq(m, p + 1, o, 1);
    return m_atom(m, p, o);
}
static const char* skip_group(const char* p, const char* e) {  // past a {...} (if one is there)
    p = skip_sp(p, e);
    if (p >= e || *p != '{') return p;
    int d = 0;
    for (; p < e; p++) { if (*p == '\\' && p + 1 < e) { p++; continue; } if (*p == '{') d++; else if (*p == '}' && --d == 0) return p + 1; }
    return p;
}
static int mo_simple(const MOut* a) {  // one "word": letters/digits only, so a⁄b reads right
    if (a->n == 0) return 0;
    for (int i = 0; i < a->n; i++) {
        u32 c = a->c[i] & ~IT;
        if (!(is_alpha(c) || is_digit(c) || (c >= 0x391 && c <= 0x3C9) || c == '.' || c == 0x2032)) {
            if (!(c >= 0x2070 && c <= 0x209F) && !(c >= 0x1D2C && c <= 0x1DBF) && !(c >= 0x2B0 && c <= 0x2FF)) return 0;
        }
    }
    return 1;
}
static int mo_atomic(const MOut* a) {  // needs no parentheses: one symbol or one number
    if (a->n == 1) return 1;
    for (int i = 0; i < a->n; i++) if (!is_digit(a->c[i]) && a->c[i] != '.') return 0;
    return a->n > 0;
}
static void mo_wrapped(MOut* o, const MOut* a) {
    if (mo_atomic(a)) { mo_cat(o, a); return; }
    mo_put(o, '('); mo_cat(o, a); mo_put(o, ')');
}
static void mo_trim(MOut* a) {
    int s = 0; while (s < a->n && a->c[s] == ' ') s++;
    int e = a->n; while (e > s && a->c[e - 1] == ' ') e--;
    if (s) for (int i = s; i < e; i++) a->c[i - s] = a->c[i];
    a->n = e - s;
}

static void m_script(MP* m, const char** pp, MOut* o, int sup) {
    MOut t; t.n = 0;
    *pp = m_arg(m, *pp + 1, &t);
    mo_trim(&t);
    if (!t.n) return;
    MOut r; r.n = 0; int ok = 1;
    for (int i = 0; i < t.n && ok; i++) {
        u32 c = t.c[i] & ~IT;
        if (c == ' ') continue;
        u32 s = script_of(c, sup);
        if (!s && sup && (c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0x2070 && c <= 0x207F) || (c >= 0x1D2C && c <= 0x1DBF))) s = c;
        if (!s && !sup && c >= 0x2080 && c <= 0x209C) s = c;
        if (s) mo_put(&r, s); else ok = 0;
    }
    if (ok && r.n) { mo_cat(o, &r); return; }
    mo_put(o, sup ? '^' : '_');
    if (t.n == 1) mo_cat(o, &t); else { mo_put(o, '('); mo_cat(o, &t); mo_put(o, ')'); }
}

static const char* m_command(MP* m, const char* p, MOut* o) {
    const char* e = m->e;
    p++;  // the backslash
    if (p >= e) return p;
    if (!is_alpha((u8)*p)) {  // one-character command
        char c = *p++;
        switch (c) {
        case ',': case ':': case ';': case '>': case ' ': mo_put(o, ' '); break;
        case '!': break;
        case '\\': mo_put(o, '\n'); break;
        case '|': mo_put(o, 0x2016); break;
        default: mo_put(o, (u8)c); break;  // \{ \} \_ \% \$ \# \&
        }
        return p;
    }
    const char* n = p; while (p < e && is_alpha((u8)*p)) p++;
    int len = (int)(p - n);
    for (int i = 0; i < COUNT(SYM); i++) {
        const char* s = SYM[i].name; int k = 0;
        while (k < len && s[k] == n[k]) k++;
        if (k == len && !s[k]) { mo_str(o, SYM[i].out); return p; }
    }
    if (in_list(DROP, COUNT(DROP), n, len)) {
        p = skip_sp(p, e);
        if (p < e && *p == '.') p++;  // \left. / \right.
        return p;
    }
    if (in_list(TEXTARG, COUNT(TEXTARG), n, len)) {
        p = skip_sp(p, e);
        if (p < e && *p == '{') {
            int d = 1; p++;
            while (p < e) {
                if (*p == '{') d++;
                else if (*p == '}' && --d == 0) { p++; break; }
                mo_put(o, utf8_next(&p, e));
                continue;
            }
        }
        return p;
    }
    if (in_list(UPRIGHT, COUNT(UPRIGHT), n, len)) {
        MOut t; t.n = 0; p = m_arg(m, p, &t);
        for (int i = 0; i < t.n; i++) t.c[i] &= ~IT;
        mo_cat(o, &t);
        return p;
    }
#define IS(s) (len == (int)sizeof(s) - 1 && !memcmp(n, s, len))
    if (IS("frac") || IS("dfrac") || IS("tfrac") || IS("cfrac")) {
        MOut a, b; a.n = b.n = 0;
        p = m_arg(m, p, &a); p = m_arg(m, p, &b); mo_trim(&a); mo_trim(&b);
        if (mo_simple(&a) && mo_simple(&b)) {
            // "b/a x" would read as b/(ax): bracket a short fraction that a factor follows
            const char* q = p; while (q < e && *q == ' ') q++;
            int glued = q < e && (is_alpha((u8)*q) || is_digit((u8)*q) || *q == '(');
            static int lookahead;  // bounded: a chain of fractions must not recurse without limit
            if (q + 1 < e && *q == '\\' && is_alpha((u8)q[1]) && lookahead < 2) {
                MOut t; t.n = 0; MP m2 = {e, 0};
                const char* q2 = q + 1; while (q2 < e && is_alpha((u8)*q2)) q2++;
                lookahead++; m_command(&m2, q, &t); lookahead--;
                u32 c0 = t.n ? t.c[0] & ~IT : 0;
                glued = (c0 >= 0x391 && c0 <= 0x3C9) || c0 == 0x221A || (q2 - q == 5 && !memcmp(q + 1, "frac", 4));
            }
            if (glued) mo_put(o, '(');
            mo_cat(o, &a); mo_put(o, 0x2044); mo_cat(o, &b);
            if (glued) mo_put(o, ')');
        }
        else { mo_wrapped(o, &a); mo_put(o, '/'); mo_wrapped(o, &b); }
        return p;
    }
    if (IS("sqrt")) {
        MOut idx, a; idx.n = a.n = 0;
        p = skip_sp(p, e);
        if (p < e && *p == '[') {
            p++;
            while (p < e && *p != ']') p = m_atom(m, p, &idx);
            if (p < e) p++;
            mo_trim(&idx);
        }
        p = m_arg(m, p, &a); mo_trim(&a);
        if (idx.n == 1 && idx.c[0] == '3') mo_put(o, 0x221B);
        else if (idx.n == 1 && idx.c[0] == '4') mo_put(o, 0x221C);
        else {
            for (int i = 0; i < idx.n; i++) { u32 s = script_of(idx.c[i] & ~IT, 1); mo_put(o, s ? s : idx.c[i]); }
            mo_put(o, 0x221A);
        }
        if (mo_atomic(&a) || (mo_simple(&a) && a.n <= 2)) mo_cat(o, &a); else { mo_put(o, '('); mo_cat(o, &a); mo_put(o, ')'); }
        return p;
    }
    if (IS("mathbb") || IS("Bbb") || IS("mathds")) {
        MOut t; t.n = 0; p = m_arg(m, p, &t);
        for (int i = 0; i < t.n; i++) {
            u32 c = t.c[i] & ~IT;
            u32 d = c == 'R' ? 0x211D : c == 'N' ? 0x2115 : c == 'Z' ? 0x2124 : c == 'Q' ? 0x211A : c == 'C' ? 0x2102
                  : c == 'P' ? 0x2119 : c == 'H' ? 0x210D : c;
            mo_put(o, d);
        }
        return p;
    }
    if (IS("begin") || IS("end")) {
        const char* q = skip_sp(p, e);
        int array = q + 6 < e && (!memcmp(q, "{array}", 7) || !memcmp(q, "{tabular", 8));
        p = skip_group(p, e);
        if (array && IS("begin")) p = skip_group(p, e);  // column spec
        if (IS("end")) mo_put(o, ' ');
        return p;
    }
    if (IS("overset") || IS("underset") || IS("stackrel") || IS("color") || IS("textcolor")) {
        p = skip_group(p, e);
        if (IS("color")) return p;
        return m_arg(m, p, o);
    }
    if (IS("binom") || IS("dbinom") || IS("tbinom") || IS("choose")) {
        MOut a, b; a.n = b.n = 0;
        p = m_arg(m, p, &a); p = m_arg(m, p, &b); mo_trim(&a); mo_trim(&b);
        mo_put(o, 'C'); mo_put(o, '('); mo_cat(o, &a); mo_put(o, ','); mo_cat(o, &b); mo_put(o, ')');
        return p;
    }
    if (IS("pmod")) {
        MOut a; a.n = 0; p = m_arg(m, p, &a);
        mo_str(o, " (mod "); mo_cat(o, &a); mo_put(o, ')');
        return p;
    }
#undef IS
    // unknown command (or a function name like \sin, \lim, \log): show the name upright
    for (int i = 0; i < len; i++) mo_put(o, (u8)n[i]);
    if (p < e && *p == ' ') mo_put(o, ' ');
    return p;
}

static const char* m_atom(MP* m, const char* p, MOut* o) {
    const char* e = m->e;
    char c = *p;
    if (++m->depth > 24) { m->depth--; return e; }  // pathological nesting: stop
    if (c == '{') p = m_seq(m, p + 1, o, 1);
    else if (c == '}') p++;
    else if (c == '\\') p = m_command(m, p, o);
    else if (c == '^' || c == '_') m_script(m, &p, o, c == '^');
    else if (c == '&' || c == '~') { mo_put(o, ' '); p++; }
    else if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        u32 l = o->n ? o->c[o->n - 1] : 0;
        if (l && l != ' ' && l != '\n' && l != '(' && l != '[') mo_put(o, ' ');
        p++;
    } else {
        u32 cp = utf8_next(&p, e);
        if (cp == '-') cp = 0x2212;
        else if (cp == '\'') cp = 0x2032;
        else if (is_alpha(cp)) cp |= IT;
        mo_put(o, cp);
    }
    m->depth--;
    return p;
}
static const char* m_seq(MP* m, const char* p, MOut* o, int until_brace) {
    while (p < m->e) {
        if (until_brace && *p == '}') return p + 1;
        p = m_atom(m, p, o);
    }
    return p;
}

// ---------------------------------------------------------------- formatter state
enum { M_NONE, M_DOLLAR, M_PAREN, M_DDOLLAR, M_BRACKET };
enum { B_MORE, B_TEXT, B_HEAD, B_BULLET, B_NUM, B_QUOTE, B_RULE, B_FENCE, B_FENCE_CLOSE, B_TABLE };

static struct {
    int base;
    int bold, italic, code, head, quote, fence, skip_line;
    int bol, line_vis, line_hidden;
    u32 pre[64]; int pre_n;
    u32 pend; int pend_n;
    u32 last;
    int math, math_bol, math_depth, close_pend;
    char mb[2048]; int mb_n;
    u32 utf; int utf_need;
    int row_sep, row_seen, active;  // streaming a |---| row
} md;

// table capture: rows are shown as they stream, then redrawn aligned when the table ends
#define TB_ROWS 48
#define TB_COLS 12
#define TB_CELLS 6144
static struct {
    int on, ok, nrows, row_open;
    ConMark start;
    u32 c[TB_CELLS]; int n;
    int ncell[TB_ROWS], cell[TB_ROWS][TB_COLS + 1], sep[TB_ROWS];
} tb;

static int color_now(int want) { return md.base == COL_DIM ? COL_DIM : want; }

static void emit(u32 cp, int color, int bg, int face) {
    con_style(color, bg, face);
    con_put(cp);
    md.line_vis++;
    md.last = cp;
    if (tb.on && tb.row_open && tb.ok) {
        int r = tb.nrows - 1, k = tb.ncell[r];
        if (tb.n < TB_CELLS && k >= 1) tb.c[tb.n++] = cp | (u32)color << 21 | (u32)bg << 25 | (u32)face << 27;
        else tb.ok = 0;
        tb.cell[r][k] = tb.n;
    }
}
static void deco(u32 cp, int color) {  // decoration: drawn, never part of a table cell
    con_style(color_now(color), BG_NONE, 0);
    con_put(cp);
    md.line_vis++;
}
static void out(u32 cp) {
    int color = md.fence || md.code ? COL_CODE : md.head ? COL_HEAD : md.bold ? COL_BOLD : md.quote ? COL_DIM : md.base;
    int bg = md.fence || md.code ? BG_CODE : BG_NONE;
    int face = (md.bold || md.head) && !md.code ? FACE_BOLD : md.italic && !md.code ? FACE_ITALIC : FACE_REG;
    emit(cp, color_now(color), bg, face);
}
static void out_str(const char* s, int color) {
    const char* e = s; while (*e) e++;
    while (s < e) emit(utf8_next(&s, e), color_now(color), BG_NONE, FACE_REG);
}

// ---- tables ----
static void tb_cell_open(void) {
    int r = tb.nrows - 1;
    if (tb.ncell[r] >= TB_COLS) { tb.ok = 0; return; }
    tb.ncell[r]++;
    tb.cell[r][tb.ncell[r]] = tb.n;
}
static void tb_row_begin(void) {
    if (!tb.on) { tb.on = 1; tb.ok = 1; tb.nrows = 0; tb.n = 0; tb.start = con_mark(); }
    if (tb.nrows >= TB_ROWS) { tb.ok = 0; tb.row_open = 0; return; }
    int r = tb.nrows++;
    tb.ncell[r] = 0; tb.cell[r][0] = tb.n; tb.sep[r] = 0; tb.row_open = 1;
    tb_cell_open();
}
static void tb_row_end(void) {
    if (!tb.row_open) return;
    tb.row_open = 0;
    int r = tb.nrows - 1;
    // drop the empty cell after the closing '|'
    int k = tb.ncell[r];
    int empty = 1;
    for (int i = tb.cell[r][k - 1]; i < tb.cell[r][k]; i++) if ((tb.c[i] & 0x1FFFFF) != ' ') empty = 0;
    if (k > 1 && empty) tb.ncell[r]--;
    // separator row: only - : and spaces
    int dash = 0, other = 0;
    for (int i = tb.cell[r][0]; i < tb.cell[r][tb.ncell[r]]; i++) {
        u32 c = tb.c[i] & 0x1FFFFF;
        if (c == '-' || c == 0x2500) dash = 1; else if (c != ':' && c != ' ') other = 1;
    }
    tb.sep[r] = dash && !other;
}
static void tb_trim(int r, int k, int* a, int* b) {
    *a = tb.cell[r][k]; *b = tb.cell[r][k + 1];
    while (*a < *b && (tb.c[*a] & 0x1FFFFF) == ' ') (*a)++;
    while (*b > *a && (tb.c[*b - 1] & 0x1FFFFF) == ' ') (*b)--;
}
static void tb_end(void) {
    if (!tb.on) return;
    tb_row_end();
    tb.on = 0;
    if (!tb.ok || !tb.nrows || !con_mark_visible(tb.start)) return;
    int ncols = 0, w[TB_COLS] = {0};
    for (int r = 0; r < tb.nrows; r++) {
        if (tb.ncell[r] > ncols) ncols = tb.ncell[r];
        if (tb.sep[r]) continue;
        for (int k = 0; k < tb.ncell[r]; k++) { int a, b; tb_trim(r, k, &a, &b); if (b - a > w[k]) w[k] = b - a; }
    }
    int total = 1;
    for (int k = 0; k < ncols; k++) total += w[k] + 3;
    if (total > con_cols()) return;  // wider than the screen: keep it as streamed
    con_rewind(tb.start);
    for (int r = 0; r < tb.nrows; r++) {
        if (tb.sep[r]) {
            con_style(color_now(COL_DIM), BG_NONE, 0);
            con_put(0x251C);
            for (int k = 0; k < ncols; k++) {
                for (int i = 0; i < w[k] + 2; i++) con_put(0x2500);
                con_put(k == ncols - 1 ? 0x2524 : 0x253C);
            }
        } else {
            for (int k = 0; k < ncols; k++) {
                con_style(color_now(COL_DIM), BG_NONE, 0); con_put(0x2502); con_put(' ');
                int a = 0, b = 0;
                if (k < tb.ncell[r]) tb_trim(r, k, &a, &b);
                for (int i = a; i < b; i++) {
                    u32 v = tb.c[i];
                    con_style((v >> 21) & 15, (v >> 25) & 3, (v >> 27) & 3);
                    con_put(v & 0x1FFFFF);
                }
                con_style(COL_TEXT, BG_NONE, 0);
                for (int i = b - a; i < w[k] + 1; i++) con_put(' ');
            }
            con_style(color_now(COL_DIM), BG_NONE, 0); con_put(0x2502);
        }
        con_put('\n');
    }
}

// ---- math ----
static void math_emit(const MOut* o, int display) {
    MOut t = *o;
    while (t.n && (t.c[t.n - 1] == '\n' || t.c[t.n - 1] == ' ')) t.n--;  // a trailing \\ ends the line anyway
    mo_trim(&t);
    if (!t.n) return;
    if (display && md.math_bol && md.line_vis == 0) { con_style(COL_TEXT, 0, 0); for (int i = 0; i < 4; i++) con_put(' '); }
    for (int i = 0; i < t.n; i++) {
        u32 c = t.c[i];
        if (c == '\n') {
            if (display) {
                md.line_vis = 0;
                con_put('\n');
                if (md.math_bol) { con_put(' '); con_put(' '); con_put(' '); con_put(' '); }
            } else { emit(';', color_now(COL_MATH), BG_NONE, 0); emit(' ', color_now(COL_MATH), BG_NONE, 0); }
            continue;
        }
        emit(c & ~IT, color_now(COL_MATH), BG_NONE, (c & IT) ? FACE_ITALIC : FACE_REG);
    }
}
static void math_convert(const char* s, int n, int display) {
    init_scripts();
    MOut o; o.n = 0;
    MP m = {s + n, 0};
    m_seq(&m, s, &o, 0);
    math_emit(&o, display);
}
static void md_cp(u32 cp);
static void math_raw(void) {  // not math after all: print what was held back
    static const char* const open[] = {"", "$", "\\(", "$$", "\\["};
    int kind = md.math; md.math = M_NONE;
    const char* p = open[kind];
    if (kind == M_PAREN || kind == M_BRACKET) p++;  // an escaped bracket is just the bracket
    while (*p) out((u8)*p++);
    char copy[256]; int n = md.mb_n < (int)sizeof copy ? md.mb_n : (int)sizeof copy;
    memcpy(copy, md.mb, n); md.mb_n = 0;
    const char* q = copy; const char* e = copy + n;  // the rest is ordinary text again
    while (q < e) md_cp(utf8_next(&q, e));
}
static void math_begin(int kind) {
    md.math = kind; md.mb_n = 0; md.math_depth = 0;
    md.math_bol = md.line_vis == 0;
    md.line_hidden = 1;
}
static void math_end(void) {
    int display = md.math >= M_DDOLLAR;
    md.math = M_NONE;
    math_convert(md.mb, md.mb_n, display);
    md.mb_n = 0;
}
static void math_cp(u32 cp) {
    int k = md.math;
    if (md.close_pend) {  // "$" then: a digit means it was a price ("$5 and $10"), else math ends
        md.close_pend = 0;
        if (is_digit(cp)) md.mb[md.mb_n++] = '$';
        else { math_end(); md_cp(cp); return; }
    }
    if (cp == '\n') {
        if (k == M_DOLLAR || k == M_PAREN) { math_raw(); md_cp('\n'); return; }
        if (md.math_depth > 0) cp = ' ';
        else {  // display math: show each finished source line
            int had = md.line_vis;
            math_convert(md.mb, md.mb_n, 1);
            md.mb_n = 0;
            if (md.line_vis) { con_put('\n'); md.line_vis = 0; }
            else if (had) { con_put('\n'); }
            md.math_bol = 1;
            return;
        }
    }
    char u[4]; int ul = utf8_enc(u, cp);
    if (md.mb_n + ul >= (int)sizeof md.mb - 1 || (k <= M_PAREN && md.mb_n > 200)) { math_raw(); md_cp(cp); return; }
    char prev = md.mb_n ? md.mb[md.mb_n - 1] : 0;
    int escaped = 0;
    for (int i = md.mb_n - 1; i >= 0 && md.mb[i] == '\\'; i--) escaped = !escaped;
    if (!escaped) {
        if (k == M_DOLLAR && cp == '$' && md.mb_n) { md.close_pend = 1; return; }
        if (k == M_DDOLLAR && cp == '$' && prev == '$') { md.mb_n--; math_end(); return; }
        if (cp == '{') md.math_depth++;
        if (cp == '}' && md.math_depth > 0) md.math_depth--;
    }
    if (escaped && prev == '\\') {
        if ((k == M_PAREN && cp == ')') || (k == M_BRACKET && cp == ']')) { md.mb_n--; math_end(); return; }
    }
    memcpy(md.mb + md.mb_n, u, ul); md.mb_n += ul;
}

// ---- inline ----
static void resolve_pending(u32 next) {
    u32 p = md.pend; md.pend = 0;
    if (p == '*') {
        if (md.pend_n >= 2) { md.bold = !md.bold; return; }
        if (md.italic && !is_space(md.last)) { md.italic = 0; return; }
        if (!md.italic && next && !is_space(next) && next != '\n') { md.italic = 1; return; }
        out('*');
    } else if (p == '\\') out('\\');
    else if (p == '$') out('$');
}
static void inl(u32 cp) {
    if (md.code) { if (cp == '`') md.code = 0; else out(cp); return; }
    if (md.pend == '*') {
        if (cp == '*' && md.pend_n == 1) { md.pend_n = 2; return; }
        resolve_pending(cp);
    } else if (md.pend == '\\') {
        md.pend = 0;
        if (cp == '(') { math_begin(M_PAREN); return; }
        if (cp == '[') { math_begin(M_BRACKET); return; }
        if (is_punct(cp)) { out(cp); return; }
        out('\\');
    } else if (md.pend == '$') {
        md.pend = 0;
        if (cp == '$') { math_begin(M_DDOLLAR); return; }
        math_begin(M_DOLLAR); math_cp(cp); return;
    }
    switch (cp) {
    case '*': md.pend = '*'; md.pend_n = 1; return;
    case '\\': md.pend = '\\'; return;
    case '$': md.pend = '$'; return;
    case '`': md.code = 1; return;
    case '|':
        if (tb.on && tb.row_open) {
            deco(md.row_sep ? 0x253C : 0x2502, COL_DIM);
            tb_cell_open();
            return;
        }
        break;
    }
    if (tb.on && tb.row_open && !md.row_seen && !is_space(cp)) { md.row_seen = 1; md.row_sep = cp == '-' || cp == ':'; }
    if (md.row_sep && (cp == '-' || cp == ':' || cp == ' ')) { emit(cp == ' ' ? ' ' : 0x2500, color_now(COL_DIM), BG_NONE, 0); return; }
    out(cp);
}

// ---- line starts ----
static int classify(int final, int* consumed, int* level) {
    int n = md.pre_n, i = 0; const u32* s = md.pre;
    *consumed = 0;
    while (i < n && is_space(s[i])) i++;
    if (n >= 60) return B_TEXT;
    if (i == n) return final ? B_TEXT : B_MORE;
    u32 c = s[i];
    if (md.fence) {
        int j = i; while (j < n && s[j] == '`') j++;
        if (j - i >= 3) {
            for (int k = j; k < n; k++) if (!is_space(s[k])) return B_TEXT;
            return final ? B_FENCE_CLOSE : B_MORE;
        }
        return j == n && !final ? B_MORE : B_TEXT;
    }
    if (c == '#') {
        int j = i; while (j < n && s[j] == '#') j++;
        if (j - i > 6) return B_TEXT;
        if (j == n) return final ? B_TEXT : B_MORE;
        if (s[j] == ' ') { *consumed = j + 1; *level = j - i; return B_HEAD; }
        return B_TEXT;
    }
    if (c == '-' || c == '*' || c == '_' || c == '+') {
        if (i + 1 == n) return final ? B_TEXT : B_MORE;
        if (s[i + 1] == ' ' && c != '_') {
            // "- - -" would be a rule, but a bullet is far more likely
            *consumed = i + 2; *level = i; return B_BULLET;
        }
        int cnt = 0;
        for (int k = i; k < n; k++) { if (s[k] == c) cnt++; else if (!is_space(s[k])) return B_TEXT; }
        if (c == '+') return B_TEXT;
        if (!final) return B_MORE;
        return cnt >= 3 ? B_RULE : B_TEXT;
    }
    if (is_digit(c)) {
        int j = i; while (j < n && is_digit(s[j]) && j - i < 4) j++;
        if (j == n) return final ? B_TEXT : B_MORE;
        if (s[j] == '.' || s[j] == ')') {
            if (j + 1 == n) return final ? B_TEXT : B_MORE;
            if (s[j + 1] == ' ') { *consumed = j + 2; *level = i; return B_NUM; }
        }
        return B_TEXT;
    }
    if (c == '>') {
        if (i + 1 == n && !final) return B_MORE;
        *consumed = i + 1 + (i + 1 < n && s[i + 1] == ' ');
        return B_QUOTE;
    }
    if (c == '`') {
        int j = i; while (j < n && s[j] == '`') j++;
        if (j - i >= 3) { *consumed = n; return B_FENCE; }
        return j == n && !final ? B_MORE : B_TEXT;
    }
    if (c == '|') { *consumed = i + 1; return B_TABLE; }
    return B_TEXT;
}

static void set_indent_prefix(int width, u32 first, int color) {
    u32 cells[16]; int n = width < 16 ? width : 16;
    for (int i = 0; i < n; i++) cells[i] = con_cell(' ', COL_TEXT, BG_NONE, 0);
    if (first && n) cells[0] = con_cell(first, color, BG_NONE, 0);
    con_wrap_prefix(cells, n);
}
static void code_pad(void) {
    u32 pad = con_cell(' ', COL_CODE, BG_CODE, 0);
    emit(' ', COL_CODE, BG_CODE, 0);
    con_wrap_prefix(&pad, 1);
}

// Decide what the buffered line start is, draw its decoration, replay the rest as text.
static void apply_block(int b, int consumed, int level) {
    u32 pre[64]; int n = md.pre_n;
    memcpy(pre, md.pre, sizeof pre);
    md.pre_n = 0; md.bol = 0;
    if (b != B_TABLE && tb.on) tb_end();
    int i = 0;
    switch (b) {
    case B_HEAD:
        md.head = level;
        while (i < consumed && is_space(pre[i])) out(pre[i++]);
        i = consumed;
        break;
    case B_BULLET:
        for (; i < level; i++) out(pre[i]);
        emit(0x2022, color_now(COL_ACCENT), BG_NONE, 0); emit(' ', COL_TEXT, BG_NONE, 0);
        set_indent_prefix(level + 2, 0, 0);
        i = consumed;
        break;
    case B_NUM:
        for (; i < level; i++) out(pre[i]);
        for (; i < consumed - 1; i++) emit(pre[i], color_now(COL_ACCENT), BG_NONE, 0);
        emit(' ', COL_TEXT, BG_NONE, 0);
        set_indent_prefix(consumed, 0, 0);
        i = consumed;
        break;
    case B_QUOTE:
        emit(0x2502, color_now(COL_DIM), BG_NONE, 0); emit(' ', COL_TEXT, BG_NONE, 0);
        set_indent_prefix(2, 0x2502, color_now(COL_DIM));
        md.quote = 1;
        i = consumed;
        break;
    case B_RULE: {
        int w = con_cols() - con_col();
        for (int k = 0; k < w; k++) emit(0x2500, color_now(COL_DIM), BG_NONE, 0);
        return;
    }
    case B_FENCE:
        md.fence = 1; md.skip_line = 1;
        return;
    case B_FENCE_CLOSE:
        code_pad();
        md.fence = 0;
        md.line_hidden = 2;  // padding line: fill it in code background at the newline
        return;
    case B_TABLE: {
        tb_row_begin();
        int j = consumed; while (j < n && is_space(pre[j])) j++;
        md.row_sep = j < n ? (pre[j] == '-' || pre[j] == ':') : 0;
        deco(md.row_sep ? 0x251C : 0x2502, COL_DIM);
        i = consumed;
        break;
    }
    default:
        if (md.fence) code_pad();
        break;
    }
    for (; i < n; i++) md_cp(pre[i]);
}

static void nl(void) {
    if (md.pend) resolve_pending('\n');
    if (md.bol && md.pre_n) {
        int consumed, level = 0, b = classify(1, &consumed, &level);
        apply_block(b, consumed, level);
        if (md.pend) resolve_pending('\n');
        if (md.math == M_DOLLAR || md.math == M_PAREN) math_raw();
    } else if (md.bol) {
        if (tb.on) tb_end();
        if (md.fence) code_pad();
    }
    md.skip_line = 0;
    if (tb.on && tb.row_open) tb_row_end();
    if (md.fence || md.line_hidden == 2) { con_style(COL_CODE, BG_CODE, 0); con_fill_eol(); }
    int suppress = md.line_vis == 0 && md.line_hidden == 1;
    md.bold = md.italic = md.code = md.head = md.quote = 0; md.row_sep = md.row_seen = 0;
    if (!suppress) con_put('\n');
    md.line_vis = 0; md.line_hidden = 0; md.bol = 1; md.last = ' ';
}

static void md_cp(u32 cp) {
    if (md.math) { math_cp(cp); return; }
    if (cp == '\n') { nl(); return; }
    if (cp == '\r') return;
    if (md.skip_line) return;
    if (md.bol) {
        md.pre[md.pre_n++] = cp;
        int consumed, level = 0, b = classify(0, &consumed, &level);
        if (b != B_MORE) apply_block(b, consumed, level);
        return;
    }
    if (md.fence) { out(cp); return; }
    inl(cp);
}

void md_putb(u8 b) {
    if (md.utf_need) {
        if ((b & 0xC0) == 0x80) { md.utf = md.utf << 6 | (b & 0x3F); if (--md.utf_need == 0) md_cp(md.utf); return; }
        md.utf_need = 0; md_cp(0xFFFD);
    }
    if (b < 0x80) md_cp(b);
    else if ((b & 0xE0) == 0xC0) { md.utf = b & 0x1F; md.utf_need = 1; }
    else if ((b & 0xF0) == 0xE0) { md.utf = b & 0x0F; md.utf_need = 2; }
    else if ((b & 0xF8) == 0xF0) { md.utf = b & 0x07; md.utf_need = 3; }
}

void md_finish(void) {
    if (!md.active) return;
    if (md.close_pend) { md.close_pend = 0; math_end(); }
    if (md.math >= M_DDOLLAR) math_end();
    else if (md.math) math_raw();
    if (md.pend) resolve_pending(0);
    if (!md.bol || md.pre_n) nl();
    if (tb.on) tb_end();
    md.fence = 0; md.skip_line = 0; md.utf_need = 0; md.active = 0;
    con_color(md.base);
}

void md_reset(int base) {
    md_finish();
    memset(&md, 0, sizeof md);
    md.base = base; md.bol = 1; md.last = ' '; md.active = 1;
    con_color(base);
    (void)first_cp;
}
