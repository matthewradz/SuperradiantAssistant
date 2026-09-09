"""Render the night-dome photo for the lab-select screen.

The first attempt cropped a ten-row strip of the colonnade. It was legible as
*columns* and completely unrecognisable as MIT: the dome is the whole point, and
the dome needs height. Row count is forced by the crop --
`rows = cols * (h/w) * 0.5` -- so a recognisable building cannot be ten rows
wide-screen. It goes beside the boxes instead of above them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_art import photo
from preview_ansi import render

SRC = Path.home() / "Desktop" / "Slice_24_04_16_USNews.jpg"
OUT = Path(str(Path(__file__).resolve().parents[2])
           r"\superradiant_assistant\assets")
HERE = Path(__file__).parent

if not SRC.exists():
    raise SystemExit(f"not found: {SRC}")

# Chosen after looking at four crops side by side. 44 columns is what the boot
# screen already uses for its art, and it leaves the lab boxes enough width to
# read. Wider (60 columns) shows more of the wings and is the nicer picture, but
# it squeezes the boxes to 37 columns, where every fact row truncates.
ART = ("mit_night_dome.ansi", 44, (0.24, 0.04, 0.76, 0.86))

name, cols, crop = ART
art = photo(SRC, cols, crop=crop, contrast=1.3, sat=1.45)
(OUT / name).write_text(art, encoding="utf-8")
rows = art.count("\n") + 1
render([OUT / name], HERE / "lab_art.png")
print(f"  {name:<22} {cols} cols x {rows:>2} rows")

# The ten-row colonnade strip was unrecognisable as MIT; remove it so nothing
# loads it by accident.
old = OUT / "mit_night_band.ansi"
if old.exists():
    old.unlink()
    print(f"  removed {old.name} (too short to read as a building)")
