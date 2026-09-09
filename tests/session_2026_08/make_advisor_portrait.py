"""Render the advisor's photograph into `assets/advisor.ansi`.

    python tests/session_2026_08/make_advisor_portrait.py <image> [columns]

Same half-block cells as `make_art.py`: one glyph carries two vertical pixels,
foreground for the top and background for the bottom, so a cell is square-ish and
the vertical resolution is doubled. At the 18 columns the reply header uses that
is 18x36 pixels of face, which is enough to be recognised and not enough to be
mistaken for anything else.

Two things are done here that the MIT art did not need:

* **A head crop.** A headshot is mostly background. Framing the upper-middle of
  the frame -- where a portrait's face sits -- spends those 36 pixel rows on the
  face instead of on the shoulders and the wall.
* **A slight brightness lift.** A face rendered at this size against a dark
  terminal reads as a silhouette otherwise; the eyes and the mouth are the only
  features with room to survive, and they need the contrast.

Run once. The `.ansi` file is the asset; the source image is not needed at
runtime, is not copied into the repo, and should not be committed.
"""
import sys
from pathlib import Path

from PIL import Image, ImageEnhance

OUT = Path(__file__).resolve().parents[2] / "superradiant_assistant" / "assets"

RESET = "\x1b[0m"
UPPER = "\u2580"

#: Fraction of the source frame kept: (left, top, right, bottom).
#:
#: Measured, not guessed. On the source headshot the subject is NOT centred: skin
#: tone occupies cols 0.39-0.76 and the bright hair starts at row 0.12, while the
#: densest non-background rows are the shoulders at the bottom. A centred crop
#: framed the shirt and cut the chin. This window puts the face in the middle of
#: 18x24 pixels and still keeps the hair and a little of the collar, which is what
#: makes it read as a person rather than as a smudge.
HEAD_CROP = (0.28, 0.05, 0.82, 0.92)


def fg(r, g, b):
    return f"\x1b[38;2;{r};{g};{b}m"


def bg(r, g, b):
    return f"\x1b[48;2;{r};{g};{b}m"


def portrait(path: Path, cols: int = 18, crop=HEAD_CROP,
             contrast: float = 1.18, sat: float = 1.12,
             brightness: float = 1.10) -> str:
    im = Image.open(path).convert("RGB")
    if crop:
        w0, h0 = im.size
        left, top, right, bottom = crop
        im = im.crop((int(left * w0), int(top * h0),
                      int(right * w0), int(bottom * h0)))
    im = ImageEnhance.Brightness(im).enhance(brightness)
    im = ImageEnhance.Contrast(im).enhance(contrast)
    im = ImageEnhance.Color(im).enhance(sat)

    w, h = im.size
    rows = max(1, round(cols * (h / w) * 0.5))
    im = im.resize((cols, rows * 2), Image.LANCZOS)
    px = im.load()

    lines = []
    for row in range(rows):
        cells = [f"{fg(*px[x, row * 2])}{bg(*px[x, row * 2 + 1])}{UPPER}"
                 for x in range(cols)]
        lines.append("".join(cells) + RESET)
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[2].strip())
        print("\n  e.g. make_advisor_portrait.py "
              '"~/Pictures/advisor.jpg"')
        return 2
    src = Path(sys.argv[1].strip().strip('"').strip("'"))
    if not src.exists():
        print(f"no such image: {src}")
        return 1
    cols = int(sys.argv[2]) if len(sys.argv) > 2 else 18

    art = portrait(src, cols=cols)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "advisor.ansi"
    dest.write_text(art, encoding="utf-8")
    rows = len(art.split("\n"))
    print(f"wrote {dest}  ({cols} columns x {rows} rows, from {src.name})")
    print("\npreview:\n")
    print(art)
    print("\nIf the crop cut the face badly, edit HEAD_CROP at the top of this "
          "file and run it again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
