"""Render ANSI truecolour art back to a PNG, so the art can actually be looked at.

Shipping terminal art sight-unseen is guesswork; this replays the escape codes
onto an image using a monospace font, which is close enough to judge by.
"""
import re
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SCRATCH = Path(__file__).parent
BG = (18, 20, 27)
CELL_W, CELL_H = 9, 18

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\consola.ttf",
    r"C:\Windows\Fonts\CascadiaMono.ttf",
    r"C:\Windows\Fonts\lucon.ttf",
]

TOKEN = re.compile(r"\x1b\[([0-9;]*)m")


def load_font(size=15):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def parse(line):
    """-> [(char, fg, bg)] with fg/bg as RGB tuples or None."""
    cells, fg, bg, i = [], None, None, 0
    for m in TOKEN.finditer(line):
        for ch in line[i:m.start()]:
            cells.append((ch, fg, bg))
        codes = [c for c in m.group(1).split(";") if c != ""]
        j = 0
        if not codes:
            fg = bg = None
        while j < len(codes):
            c = codes[j]
            if c == "0":
                fg = bg = None
                j += 1
            elif c == "38" and codes[j + 1:j + 2] == ["2"]:
                fg = tuple(int(x) for x in codes[j + 2:j + 5])
                j += 5
            elif c == "48" and codes[j + 1:j + 2] == ["2"]:
                bg = tuple(int(x) for x in codes[j + 2:j + 5])
                j += 5
            else:
                j += 1
        i = m.end()
    for ch in line[i:]:
        cells.append((ch, fg, bg))
    return cells


def render(paths, out_path, gap_rows=1):
    font = load_font()
    blocks = [p.read_text(encoding="utf-8").split("\n") for p in paths]
    grids = [[parse(l) for l in b] for b in blocks]

    cols = max(max((len(r) for r in g), default=0) for g in grids)
    rows = sum(len(g) for g in grids) + gap_rows * (len(grids) - 1)

    img = Image.new("RGB", (cols * CELL_W + 20, rows * CELL_H + 20), BG)
    d = ImageDraw.Draw(img)

    y = 10
    for gi, grid in enumerate(grids):
        for cells in grid:
            x = 10
            for ch, fg, bg in cells:
                if bg:
                    d.rectangle([x, y, x + CELL_W, y + CELL_H], fill=bg)
                if ch.strip():
                    d.text((x, y), ch, font=font, fill=fg or (200, 200, 200))
                x += CELL_W
            y += CELL_H
        y += gap_rows * CELL_H

    img.save(out_path)
    print(f"{out_path}  ({cols} cols x {rows} rows -> {img.size[0]}x{img.size[1]} px)")


if __name__ == "__main__":
    assets = (Path(sys.argv[1]) if len(sys.argv) > 1 else
              Path(__file__).resolve().parents[2]
              / "superradiant_assistant" / "assets")
    render([assets / "mit_dome.ansi", assets / "mit_logo.ansi"],
           SCRATCH / "art_preview.png")
