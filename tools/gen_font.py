#!/usr/bin/env python3
"""Generate src/font.h: anti-aliased monospaced bitmap fonts for the bare-metal UEFI app.

Usage:  python tools/gen_font.py [output_path]

All fonts live in tools/fonts/ (open licenses, see the license files there). Primary font is
Cascadia Mono. Codepoints it lacks come from DejaVu Sans Mono, Noto Sans Math, Noto Sans
Symbols 2 or DejaVu Sans, drawn at the same size on the same baseline (shrunk only when they overflow the
cell), so superscripts stay high and lowercase stays small. Box-drawing and
block/shade characters are drawn procedurally so they touch the cell edges and connect.
Bold and italic faces (Cascadia Mono Bold / Italic) cover ASCII and Latin-1 and are stored under
pseudo codepoints FACE_BOLD + cp and FACE_ITALIC + cp.

Every non-ASCII character in src/*.c is included automatically, so everything the app itself
prints (bullets, LaTeX symbols, table lines) has a glyph.

Output: font_rle_N holds the [FONT_NGLYPHS][h][w] coverage bytes (0 = background .. 255 = ink,
linear coverage), run-length encoded: a 0 byte is followed by a run length 1..255 of zeros,
every other byte is literal. console.c unpacks the size it uses at startup.
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
PRIMARY = os.path.join(FONTS_DIR, "CascadiaMono-Regular.ttf")
BOLD = os.path.join(FONTS_DIR, "CascadiaMono-Bold.ttf")
ITALIC = os.path.join(FONTS_DIR, "CascadiaMono-Italic.ttf")
# (path, index inside a .ttc), tried in order
FALLBACKS = [(os.path.join(FONTS_DIR, "DejaVuSansMono.ttf"), 0),
             (os.path.join(FONTS_DIR, "NotoSansMath-Regular.ttf"), 0),
             (os.path.join(FONTS_DIR, "NotoSansSymbols2-Regular.ttf"), 0),
             (os.path.join(FONTS_DIR, "DejaVuSans.ttf"), 0)]

CELLS = [(10, 20), (14, 28), (20, 40)]  # (w, h) for fonts[0..2]
FACE_BOLD, FACE_ITALIC = 0x110000, 0x120000
FACE_CPS = list(range(0x21, 0x7F)) + list(range(0xA1, 0x100))

EXTRA = [
    0x2018, 0x2019, 0x201A, 0x201C, 0x201D, 0x201E, 0x2013, 0x2014, 0x2026, 0x2022,
    0x20AC, 0x2122, 0x00D7, 0x2713, 0x2717, 0x25A0, 0x25A1, 0x25CF, 0x25CB,
    0x2588, 0x2591, 0x2592, 0x2032, 0x2033, 0x2016, 0x2044,
    0x27E8, 0x27E9, 0x2308, 0x2309, 0x230A, 0x230B, 0x2102, 0x2115, 0x211A, 0x211D, 0x2124,
    0x210F, 0x2113, 0x211C, 0x2111, 0x2135, 0x02B0, 0x02B2, 0x02B3, 0x02B7, 0x02B8, 0x02E1,
    0x02E2, 0x02E3, 0x1D43, 0x1D47, 0x1D48, 0x1D49, 0x1D4D, 0x1D4F, 0x1D50, 0x1D52, 0x1D56,
    0x1D57, 0x1D58, 0x1D5B, 0x1D9C, 0x1DA0, 0x1DBB, 0x1D2C, 0x1D2E, 0x1D30, 0x1D31, 0x1D33,
    0x1D34, 0x1D35, 0x1D36, 0x1D37, 0x1D38, 0x1D39, 0x1D3A, 0x1D3C, 0x1D3E, 0x1D3F, 0x1D40,
    0x1D41, 0x2C7D, 0x1D42, 0x1D45, 0x1D5D, 0x1D5E, 0x1D5F, 0x1DBF, 0x1D60, 0x1D61, 0x1D62,
    0x1D63, 0x1D64, 0x1D65, 0x1D66, 0x1D67, 0x1D68, 0x1D69, 0x1D6A, 0x2C7C,
]
RANGES = [
    (0x0100, 0x0180),  # Latin Extended-A
    (0x0391, 0x03AA), (0x03B1, 0x03CA), (0x03D1, 0x03D2), (0x03D5, 0x03D7), (0x03F5, 0x03F6),  # Greek
    (0x0400, 0x0460),  # Cyrillic
    (0x2070, 0x209D),  # super/subscripts
    (0x2190, 0x2200),  # arrows
    (0x2200, 0x2300),  # math operators
]
FALLBACK_CP = 0x25A1

# Drawn procedurally (edge-to-edge) instead of from the font.
PROCEDURAL = {0x2500, 0x2502, 0x250C, 0x2510, 0x2514, 0x2518, 0x251C, 0x2524, 0x252C, 0x2534, 0x253C,
              0x2588, 0x2591, 0x2592}

FALLBACK_SUBST = {}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def source_chars():
    """Every non-ASCII codepoint in src/*.c (the app's own UI and formatter output)."""
    out = set()
    src = os.path.join(ROOT, "src")
    for name in os.listdir(src):
        if name.endswith(".c"):
            with open(os.path.join(src, name), encoding="utf-8") as f:
                out |= {ord(ch) for ch in f.read() if 127 < ord(ch) < 0x10000}
    return out


UNASSIGNED = {0x03A2, 0x2072, 0x2073, 0x208F}
BASE_CPS = sorted((set(range(32, 127)) | set(range(160, 256)) | set(EXTRA) | PROCEDURAL
                   | {cp for a, b in RANGES for cp in range(a, b)} | source_chars()) - UNASSIGNED)
CODEPOINTS = BASE_CPS + [FACE_BOLD + cp for cp in FACE_CPS] + [FACE_ITALIC + cp for cp in FACE_CPS]


# ---------------------------------------------------------------- coverage detection
def make_coverage_checker(path, index=0):
    """Return f(cp) -> bool telling whether the font at `path` has a real glyph for cp."""
    try:
        from fontTools.ttLib import TTFont

        cmap = TTFont(path, lazy=True, fontNumber=index).getBestCmap() or {}
        return lambda cp: cp in cmap
    except ImportError:
        pass
    # Pillow-only heuristic: a missing glyph renders exactly like .notdef (probe with a
    # noncharacter that no font maps).
    font = ImageFont.truetype(path, 48, index=index)

    def render(ch):
        img = Image.new("L", (160, 160), 0)
        ImageDraw.Draw(img).text((40, 110), ch, font=font, fill=255, anchor="ls")
        return img.tobytes()

    notdef = render("\U0010FFFF")
    return lambda cp: cp == 0x20 or cp == 0xA0 or render(chr(cp)) != notdef


# ---------------------------------------------------------------- glyph rendering
PAD = 16  # extra canvas around the cell so clipping can be detected


def render_on_canvas(w, h, ch, font, x, baseline):
    """Render ch with its baseline-left origin at (x, baseline) in cell coordinates.
    Returns (cell_image, clipped_ink_sum)."""
    canvas = Image.new("L", (w + 2 * PAD, h + 2 * PAD), 0)
    ImageDraw.Draw(canvas).text((PAD + x, PAD + baseline), ch, font=font, fill=255, anchor="ls")
    total = sum(canvas.getdata())
    cell = canvas.crop((PAD, PAD, PAD + w, PAD + h))
    clipped = total - sum(cell.getdata())
    return cell, clipped


def render_nudged(w, h, ch, font, x0, baseline):
    """Glyphs overhanging the advance (e.g. Æ, —, italics): nudge up to 1px sideways."""
    cell, clipped = render_on_canvas(w, h, ch, font, x0, baseline)
    if clipped:
        for dx in (0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0):
            c2, k2 = render_on_canvas(w, h, ch, font, x0 + dx, baseline)
            if k2 < clipped:
                cell, clipped = c2, k2
    return cell, clipped


def procedural(cp, w, h):
    img = Image.new("L", (w, h), 0)
    px = img.load()
    t = max(1, round(h / 20))          # light line thickness: 1,1,2 px
    cx0 = (w - t) // 2                  # vertical stroke columns [cx0, cx0+t)
    cy0 = (h - t) // 2                  # horizontal stroke rows  [cy0, cy0+t)

    def fill(x0, y0, x1, y1, v=255):
        for y in range(max(0, y0), min(h, y1)):
            for x in range(max(0, x0), min(w, x1)):
                px[x, y] = v

    H = lambda x0, x1: fill(x0, cy0, x1, cy0 + t)
    V = lambda y0, y1: fill(cx0, y0, cx0 + t, y1)
    if cp == 0x2500:   # ─
        H(0, w)
    elif cp == 0x2502:  # │
        V(0, h)
    elif cp == 0x250C:  # ┌
        H(cx0, w); V(cy0, h)
    elif cp == 0x2510:  # ┐
        H(0, cx0 + t); V(cy0, h)
    elif cp == 0x2514:  # └
        H(cx0, w); V(0, cy0 + t)
    elif cp == 0x2518:  # ┘
        H(0, cx0 + t); V(0, cy0 + t)
    elif cp == 0x251C:  # ├
        V(0, h); H(cx0, w)
    elif cp == 0x2524:  # ┤
        V(0, h); H(0, cx0 + t)
    elif cp == 0x252C:  # ┬
        H(0, w); V(cy0, h)
    elif cp == 0x2534:  # ┴
        H(0, w); V(0, cy0 + t)
    elif cp == 0x253C:  # ┼
        H(0, w); V(0, h)
    elif cp == 0x2588:  # █
        fill(0, 0, w, h)
    elif cp == 0x2591:  # ░ 25% dot pattern (tiles across cells: w, h are even)
        for y in range(0, h, 2):
            for x in range(0, w, 2):
                px[x, y] = 255
    elif cp == 0x2592:  # ▒ 50% checkerboard
        for y in range(h):
            for x in range(w):
                if (x + y) % 2 == 0:
                    px[x, y] = 255
    else:
        raise ValueError(hex(cp))
    return img


def ink_box(font, ch):
    # Measure real ink from a render (getbbox() reports layout boxes for some fonts).
    n = 4 * int(font.size) + 8
    img = Image.new("L", (n, n), 0)
    ImageDraw.Draw(img).text((n // 4, n // 2), ch, font=font, fill=255, anchor="ls")
    box = img.getbbox()
    if not box:
        return None
    l, t, r, b = box
    return l - n // 4, t - n // 2, r - n // 4, b - n // 2


def render_fitted(cp, w, h, fb, size, baseline, cap):
    """Draw cp from font fb=(path, index) at the primary size on the primary baseline, centered
    horizontally. Only if its ink does not fit the cell is it scaled down, then centered on the
    cap-height band."""
    path, index = fb
    ch = chr(FALLBACK_SUBST.get(cp, cp))
    font = ImageFont.truetype(path, size, index=index)
    box = ink_box(font, ch)
    if box is None:
        return Image.new("L", (w, h), 0)
    l, t, r, b = box
    margin = max(1, round(w * 0.05))
    if r - l <= w - 2 * margin and baseline + t >= 0 and baseline + b <= h:
        cell, _ = render_on_canvas(w, h, ch, font, round((w - (r - l)) / 2 - l), baseline)
        return cell
    scale = min((w - 2 * margin) / (r - l), (h - 2) / (b - t), 1.0)
    font = ImageFont.truetype(path, size * scale, index=index)
    l, t, r, b = ink_box(font, ch)
    x = round((w - (r - l)) / 2 - l)
    y = round((baseline - cap / 2) - (t + b) / 2)
    y = min(max(y, -t), h - b)
    cell, _ = render_on_canvas(w, h, ch, font, x, y)
    return cell


def build_size(w, h, has_primary, fallback_for, warnings):
    """Render all glyphs for one cell size. Returns bytes of length N*h*w."""
    # Point size from the primary font's advance width.
    ref = ImageFont.truetype(PRIMARY, 1000)
    adv_per_px = ref.getlength("M") / 1000.0
    size = w / adv_per_px

    # Fit the cell height on Latin text only; tall symbols get shrunk one by one.
    normal = [chr(cp) for cp in BASE_CPS
              if cp < 0x250 and has_primary(cp) and cp not in (0x20, 0xA0)]
    while True:
        font = ImageFont.truetype(PRIMARY, size)
        boxes = [font.getbbox(ch, anchor="ls") for ch in normal]
        top = min(b[1] for b in boxes)      # negative: above baseline
        bottom = max(b[3] for b in boxes)   # positive: below baseline
        if bottom - top <= h:
            break
        size *= 0.98
    # Center the union ink box vertically, snap baseline to a whole pixel.
    baseline = round((h - (bottom - top)) / 2 - top)
    baseline = min(max(baseline, -top), h - bottom)
    adv = font.getlength("M")
    x0 = (w - adv) / 2
    cap = -font.getbbox("H", anchor="ls")[1]
    info = dict(size=size, adv=adv, baseline=baseline, top=top, bottom=bottom, cap=cap)
    faces = {FACE_BOLD: ImageFont.truetype(BOLD, size), FACE_ITALIC: ImageFont.truetype(ITALIC, size)}

    out = bytearray()
    for cp in CODEPOINTS:
        if cp >= FACE_BOLD:
            base = cp & ~0xFFFF
            cell, _ = render_nudged(w, h, chr(cp - base), faces[base], x0, baseline)
        elif cp in PROCEDURAL:
            cell = procedural(cp, w, h)
        elif cp in (0x20, 0xA0) or 0x2000 <= cp <= 0x200F:
            cell = Image.new("L", (w, h), 0)
        elif has_primary(cp):
            cell, clipped = render_nudged(w, h, chr(cp), font, x0, baseline)
            if clipped > 255 * 3:  # big operator etc.: shrink it into the cell
                cell = render_fitted(cp, w, h, (PRIMARY, 0), size, baseline, cap)
        else:
            fb = fallback_for(cp)
            if fb is None:
                warnings.append(f"{w}x{h}: U+{cp:04X} missing everywhere, left blank")
                cell = Image.new("L", (w, h), 0)
            else:
                cell = render_fitted(cp, w, h, fb, size, baseline, cap)
        out += cell.tobytes()
    return bytes(out), info


# ---------------------------------------------------------------- C emission
def rle(data):
    out = bytearray()
    i = 0
    while i < len(data):
        if data[i] == 0:
            j = i
            while j < len(data) and data[j] == 0 and j - i < 255:
                j += 1
            out += bytes((0, j - i))
            i = j
        else:
            out.append(data[i])
            i += 1
    return bytes(out)


def c_string_lines(data, width=110):
    """Bytes as C string literal lines: printable ASCII as-is, the rest as 3-digit octal."""
    lines, cur = [], []
    n = 0
    for v in data:
        ch = chr(v)
        piece = ch if (32 <= v < 127 and ch not in '"\\?') else "\\%03o" % v
        cur.append(piece)
        n += len(piece)
        if n >= width:
            lines.append('    "' + "".join(cur) + '"')
            cur, n = [], 0
    if cur:
        lines.append('    "' + "".join(cur) + '"')
    return "\n".join(lines)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "font.h")

    has_primary = make_coverage_checker(PRIMARY)
    fb_checkers = [(fb, make_coverage_checker(*fb)) for fb in FALLBACKS if os.path.exists(fb[0])]

    def fallback_for(cp):
        for fb, has in fb_checkers:
            if has(FALLBACK_SUBST.get(cp, cp)):
                return fb
        return None

    warnings = []
    fallback_used = {}
    for cp in BASE_CPS:
        if cp in PROCEDURAL or has_primary(cp) or cp in (0x20, 0xA0) or 0x2000 <= cp <= 0x200F:
            continue
        fb = fallback_for(cp)
        fallback_used.setdefault(os.path.basename(fb[0]) if fb else "NONE", []).append(cp)

    arrays = []
    for i, (w, h) in enumerate(CELLS):
        data, info = build_size(w, h, has_primary, fallback_for, warnings)
        assert len(data) == len(CODEPOINTS) * w * h
        packed = rle(data)
        arrays.append((i, w, h, packed, info))
        print(f"cell {w}x{h}: size {info['size']:.2f}px advance {info['adv']:.2f}px "
              f"baseline {info['baseline']} ink {info['top']}..{info['bottom']} cap {info['cap']} "
              f"{len(data)} -> {len(packed)} bytes")

    n = len(CODEPOINTS)
    s = []
    s.append("// Generated by tools/gen_font.py -- do not edit by hand.")
    s.append("// Glyphs rendered from the fonts in tools/fonts/; their licenses (OFL 1.1, Bitstream Vera) are there too.")
    s.append(f"// Primary: {os.path.basename(PRIMARY)} (+ Bold/Italic faces); fallback glyphs: "
             + "; ".join(f"{f}: {len(c)}" for f, c in fallback_used.items()) + ".")
    s.append("// Glyph data: [FONT_NGLYPHS][h][w] coverage bytes, RLE (0 byte + zero-run length).")
    s.append("#pragma once")
    s.append('#include "common.h"')
    s.append("")
    s.append(f"#define FONT_NGLYPHS {n}")
    s.append(f"#define FONT_FACE_BOLD 0x{FACE_BOLD:x}")
    s.append(f"#define FONT_FACE_ITALIC 0x{FACE_ITALIC:x}")
    s.append("static const u32 font_codepoints[FONT_NGLYPHS] = {")
    for i in range(0, n, 12):
        s.append("    " + ",".join(f"0x{cp:04x}" for cp in CODEPOINTS[i:i + 12]) + ",")
    s.append("};")
    s.append("")
    s.append("typedef struct { int w, h; const u8* rle; usize rle_len; } Font;")
    s.append("")
    for i, w, h, packed, _ in arrays:
        s.append(f"// {w}x{h} cells")
        s.append(f"static const u8 font_rle_{i}[{len(packed) + 1}] =")
        s.append(c_string_lines(packed) + ";")
        s.append("")
    s.append("static const Font fonts[3] = {"
             + ",".join(f"{{{w},{h},font_rle_{i},{len(p)}}}" for i, w, h, p, _ in arrays) + "};")
    s.append(f"#define FONT_FALLBACK_INDEX {CODEPOINTS.index(FALLBACK_CP)}  // U+{FALLBACK_CP:04X}")
    s.append("")

    with open(out_path, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(s))

    print(f"wrote {out_path}: {n} glyphs, {os.path.getsize(out_path)} bytes")
    for f, cps in fallback_used.items():
        print(f"fallback {f}: " + " ".join(f"{cp:04X}" for cp in cps))
    for wmsg in warnings:
        print("WARNING:", wmsg)


if __name__ == "__main__":
    main()
