"""Convert the two MIT images into ANSI art assets for the splash screen.

Both use half-block cells: one glyph carries two vertical pixels, foreground for
the top and background for the bottom, which doubles the vertical resolution.

The obvious alternative -- a density ramp of ' .:-=+*#%@' coloured per cell --
was tried first and abandoned for the photograph. The sky is the brightest thing
in the frame, so luminance hands it the heaviest glyphs and the colonnade
disappears underneath a wall of '@'. Weighting by local detail instead only
amplified the webp compression blocks. At 40 columns a photograph has too little
room to survive being redrawn in punctuation; drawn as pixels it reads instantly.
"""
from pathlib import Path
from PIL import Image, ImageEnhance

DESKTOP = Path.home() / "Desktop"
OUT = Path(str(Path(__file__).resolve().parents[2])
           r"\superradiant_assistant\assets")
OUT.mkdir(parents=True, exist_ok=True)

RESET = "\x1b[0m"
UPPER = "\u2580"
LOWER = "\u2584"


def fg(r, g, b):
    return f"\x1b[38;2;{r};{g};{b}m"


def bg(r, g, b):
    return f"\x1b[48;2;{r};{g};{b}m"


def _prepare(path, cols, crop, contrast, sat, mode):
    im = Image.open(path).convert(mode)
    if crop:
        w0, h0 = im.size
        l, t, r, b = crop
        im = im.crop((int(l * w0), int(t * h0), int(r * w0), int(b * h0)))
    if contrast != 1.0:
        im = ImageEnhance.Contrast(im).enhance(contrast)
    if sat != 1.0:
        im = ImageEnhance.Color(im).enhance(sat)
    w, h = im.size
    rows = max(1, round(cols * (h / w) * 0.5))
    return im.resize((cols, rows * 2), Image.LANCZOS), rows


def photo(path, cols, crop=None, contrast=1.25, sat=1.45):
    """Opaque image: every cell gets both a foreground and a background."""
    im, rows = _prepare(path, cols, crop, contrast, sat, "RGB")
    px = im.load()
    lines = []
    for row in range(rows):
        out = []
        for x in range(cols):
            t = px[x, row * 2]
            b = px[x, row * 2 + 1]
            out.append(f"{fg(*t)}{bg(*b)}{UPPER}")
        out.append(RESET)
        lines.append("".join(out))
    return "\n".join(lines)


def _is_background(r, g, b, a, alpha_cut=110, white_cut=0.72) -> bool:
    """True for anything that is not brand ink.

    The logo sits on a white field rather than a transparent one, so alpha alone
    does not separate it, and the antialiased edge pixels between ink and white
    render as a pale pink halo around every bar. A luminance cut removes both.
    """
    if a < alpha_cut:
        return True
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0 > white_cut


def logo(path, cols, crop=None):
    """Mark on a field: background cells stay transparent, so no colour box."""
    im, rows = _prepare(path, cols, crop, 1.0, 1.0, "RGBA")
    px = im.load()
    lines = []
    for row in range(rows):
        out = []
        for x in range(cols):
            tr, tg, tb, ta = px[x, row * 2]
            br, bg_, bb, ba = px[x, row * 2 + 1]
            top = not _is_background(tr, tg, tb, ta)
            bot = not _is_background(br, bg_, bb, ba)
            if not top and not bot:
                out.append(f"{RESET} ")
            elif top and not bot:
                out.append(f"{RESET}{fg(tr, tg, tb)}{UPPER}")
            elif bot and not top:
                out.append(f"{RESET}{fg(br, bg_, bb)}{LOWER}")
            else:
                out.append(f"{fg(tr, tg, tb)}{bg(br, bg_, bb)}{UPPER}")
        out.append(RESET)
        lines.append("".join(out))
    return trim_blank(lines)


def trim_blank(lines):
    """Drop leading and trailing rows that carry no ink."""
    import re
    def blank(s):
        return not re.sub(r"\x1b\[[0-9;]*m", "", s).strip()
    while lines and blank(lines[0]):
        lines.pop(0)
    while lines and blank(lines[-1]):
        lines.pop()
    return "\n".join(lines)


def report(name, art):
    import re
    rows = art.split("\n")
    w = max(len(re.sub(r"\x1b\[[0-9;]*m", "", r)) for r in rows)
    print(f"  {name:<16} {len(rows):>3} rows x {w:>3} cols   {len(art):>6} bytes")


dome_art = photo(DESKTOP / "MIT.webp", cols=40, crop=(0.22, 0.06, 0.80, 0.92))
logo_art = logo(DESKTOP / "MIT2.png", cols=26, crop=(0.0, 0.0, 0.88, 1.0))

(OUT / "mit_dome.ansi").write_text(dome_art, encoding="utf-8")
(OUT / "mit_logo.ansi").write_text(logo_art, encoding="utf-8")

print(f"wrote to {OUT}")
report("mit_dome.ansi", dome_art)
report("mit_logo.ansi", logo_art)
