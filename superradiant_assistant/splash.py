"""The boot screen: MIT art, the team's name, and a menu you can arrow through.

The old startup was a wall of `key : value` lines that scrolled past before
anyone read it, and the two settings that actually decide what a session does --
whether shots reach hardware, and whether the agent may write code -- were
buried in it as text. They were also only settable as command-line flags, which
is how several sessions got run without `--live` and quietly replayed history
instead of taking data.

So the same facts are laid out as a screen, and the two that matter are
togglable before the session starts. Nothing here talks to the model or the
hardware; it returns a dict of choices and gets out of the way.

The art is pre-rendered into `assets/*.ansi` from the source images, so this
module needs no image library at runtime and the files can be swapped by hand.
"""
from __future__ import annotations
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

ASSETS = Path(__file__).resolve().parent / "assets"

CARDINAL = (163, 31, 52)          # MIT red, as used in the logo art
#: The wordmark is line art, not solid fill: at one glyph per stroke, true
#: cardinal on a dark terminal is too dark to read. These are the brand colours
#: lifted toward the background so the strokes carry.
INK_RED = (206, 66, 90)
INK_GREY = (172, 174, 178)
SILVER = (138, 139, 140)
ACCENT = (226, 168, 83)
TEXT = (208, 213, 224)
DIM = (108, 114, 130)
GREEN = (126, 211, 133)
AMBER = (222, 170, 90)
#: The advisor's frame. Every other reply in this program is cardinal, so the
#: colour alone says "this is the consultant, not the agent that acts" before a
#: word of it has been read.
ADVISOR_BLUE = (96, 150, 216)

#: The envelope: paper as the background, seal red for every line drawn on it.
#: Not quite #fff — pure white on a dark terminal glares, and the flap creases
#: stop reading as creases.
PAPER = (243, 242, 238)
SEAL_RED = (198, 46, 60)

#: Plain '>' rather than '❯'. Consolas -- still the default in conhost -- has no
#: glyph for U+276F and draws a replacement box, which is worse than an ASCII
#: arrow on the one element the eye is supposed to track.
CURSOR = ">"

#: The last option on the Lab row. Cycling onto it and pressing enter starts the
#: creation flow instead of selecting an existing apparatus.
NEW_APPARATUS = "+ new lab"

#: Clears the visible screen only. The scrollback stays, which matters for
#: anything drawn mid-session.
CLEAR = "\x1b[2J\x1b[H"
#: Also drops the scrollback. Only correct at startup, when there is nothing
#: worth keeping behind us -- using it to leave the team screen threw away the
#: loading screen and the whole conversation above it.
CLEAR_ALL = "\x1b[2J\x1b[3J\x1b[H"

#: Alternate screen buffer. A full-screen overlay drawn inside it is discarded
#: on exit and the terminal restores exactly what was there before, so /team
#: does not cost the operator their transcript. Terminals that do not support
#: it ignore both codes, which leaves the previous behaviour rather than a
#: broken one.
ALT_ON = "\x1b[?1049h"
ALT_OFF = "\x1b[?1049l"

HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
RESET = "\x1b[0m"


def c(rgb) -> str:
    return f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def bold(s: str) -> str:
    return f"\x1b[1m{s}\x1b[22m"


# --------------------------------------------------------------------------
# terminal plumbing
# --------------------------------------------------------------------------

def enable_ansi() -> bool:
    """Turn on VT processing. Returns whether colour is usable at all."""
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)          # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def read_key() -> str:
    """One keypress as a name: 'up' 'down' 'left' 'right' 'enter' 'esc' or a char."""
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):                  # arrow / function prefix
            return {b"H": "up", b"P": "down", b"K": "left", b"M": "right",
                    b"I": "pgup", b"Q": "pgdn", b"G": "home",
                    b"O": "end"}.get(msvcrt.getch(), "")
        if ch in (b"\r", b"\n"):
            return "enter"
        if ch == b"\x1b":
            return "esc"
        if ch == b"\x03":
            raise KeyboardInterrupt
        try:
            return ch.decode("utf-8", "ignore").lower()
        except Exception:
            return ""

    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            nxt = sys.stdin.read(2)
            return {"[A": "up", "[B": "down", "[D": "left",
                    "[C": "right"}.get(nxt, "esc")
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _plain_len(s: str) -> int:
    import re
    return len(re.sub(r"\x1b\[[0-9;]*m", "", s))


def load_art(name: str) -> List[str]:
    try:
        return (ASSETS / name).read_text(encoding="utf-8").split("\n")
    except OSError:
        return []


# --------------------------------------------------------------------------
# the wordmark
# --------------------------------------------------------------------------

#: figlet "Standard", hand-transcribed for the eleven letters this needs. A
#: dependency on pyfiglet for one fixed string is not worth carrying onto a lab
#: machine that may be offline.
_GLYPHS = {
    "A": ["    _    ", "   / \\   ", "  / _ \\  ", " / ___ \\ ", "/_/   \\_\\"],
    "B": [" ____  ", "| __ ) ", "|  _ \\ ", "| |_) |", "|____/ "],
    "C": ["  ____ ", " / ___|", "| |    ", "| |___ ", " \\____|"],
    "E": [" _____ ", "| ____|", "|  _|  ", "| |___ ", "|_____|"],
    "G": ["  ____ ", " / ___|", "| |  _ ", "| |_| |", " \\____|"],
    "I": [" ___ ", "|_ _|", " | | ", " | | ", "|___|"],
    "L": [" _     ", "| |    ", "| |    ", "| |___ ", "|_____|"],
    "N": [" _   _ ", "| \\ | |", "|  \\| |", "| |\\  |", "|_| \\_|"],
    "P": [" ____  ", "|  _ \\ ", "| |_) |", "|  __/ ", "|_|    "],
    "R": [" ____  ", "|  _ \\ ", "| |_) |", "|  _ < ", "|_| \\_\\"],
    "S": [" ____  ", "/ ___| ", "\\___ \\ ", " ___) |", "|____/ "],
    "T": [" _____ ", "|_   _|", "  | |  ", "  | |  ", "  |_|  "],
}


def wordmark(text: str, colour=INK_RED) -> List[str]:
    rows = ["", "", "", "", ""]
    for ch in text.upper():
        glyph = _GLYPHS.get(ch)
        if glyph is None:
            glyph = ["  "] * 5
        for i in range(5):
            rows[i] += glyph[i]
    return [f"{c(colour)}{r}{RESET}" for r in rows]


# --------------------------------------------------------------------------
# menu model
# --------------------------------------------------------------------------

@dataclass
class Item:
    key: str
    label: str
    hint: str
    kind: str = "action"                    # "action" | "toggle" | "cycle"
    value: object = None
    options: List = field(default_factory=list)

    def display_value(self) -> str:
        if self.kind == "toggle":
            return "ON" if self.value else "OFF"
        if self.kind == "cycle":
            return str(self.value)
        return ""

    def advance(self, step: int = 1) -> None:
        if self.kind == "toggle":
            self.value = not self.value
        elif self.kind == "cycle" and self.options:
            i = (self.options.index(self.value) + step) % len(self.options)
            self.value = self.options[i]


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _side_by_side(left: List[str], right: List[str], gap: int = 3) -> List[str]:
    width = max((_plain_len(l) for l in left), default=0)
    out, n = [], max(len(left), len(right))
    for i in range(n):
        l = left[i] if i < len(left) else ""
        r = right[i] if i < len(right) else ""
        pad = " " * (width - _plain_len(l) + gap)
        out.append(f"{l}{RESET}{pad}{r}")
    return out


def _nav_line(items: List[Item], cursor: int, room: int = 10 ** 6) -> List[str]:
    """The menu, wrapped onto as many rows as it needs.

    One row was enough until the Lab item arrived with values like `+ new lab`;
    after that the line ran 20 columns past the frame and the last entries --
    including Quit -- were off screen. Truncating is not an option for a menu:
    every entry has to stay reachable and visible.
    """
    parts = []
    for i, it in enumerate(items):
        val = it.display_value()
        if i == cursor:
            label = bold(f"{c(ACCENT)}{CURSOR} {it.label}")
            if val:
                label += f" {c(GREEN if val in ('ON',) else AMBER)}[{val}]"
        else:
            label = f"{c(DIM)}{it.label}"
            if val:
                label += f" [{val}]"
        parts.append(label + RESET)

    sep = f"{c(DIM)}  ·  {RESET}"
    rows, cur, cur_len = [], "", 0
    for p in parts:
        add = _plain_len(p) + (5 if cur else 0)
        if cur and cur_len + add > room:
            rows.append(cur)
            cur, cur_len = p, _plain_len(p)
            continue
        cur = f"{cur}{sep}{p}" if cur else p
        cur_len += add
    if cur:
        rows.append(cur)
    return rows


def compose(items: List[Item], cursor: int, info: Dict[str, str],
            width: int) -> List[str]:
    wide = width >= 104
    dome = load_art("mit_dome.ansi") if wide else []
    logo = load_art("mit_logo.ansi")

    right: List[str] = []
    right += logo
    right.append("")
    if width >= 104:
        right += wordmark("LABSCRIPT")
        right += wordmark("AGENT", INK_GREY)
    else:
        right.append(bold(f"{c(INK_RED)}LABSCRIPT {c(INK_GREY)}AGENT{RESET}"))
    right.append("")

    live = info.get("live") == "on"
    dot = c(GREEN) if live else c(AMBER)
    state = ("LIVE — shots queue on real hardware" if live
             else "OFFLINE — replay only, nothing reaches hardware")
    right.append(f"{dot}● {bold(state)}{RESET}")
    right.append("")
    right.append(f"{c(TEXT)}An autonomous experiment agent for the labscript "
                 f"suite. It plans,{RESET}")
    right.append(f"{c(TEXT)}writes its own sequences and analyses, runs them, "
                 f"and reads the{RESET}")
    right.append(f"{c(TEXT)}data back before it says anything.{RESET}")
    right.append("")
    # What is left after the art. These rows carry variable-length content: a lab
    # with eight globals rendered a 346-column line and pushed the whole frame
    # off screen.
    art_cols = max((_plain_len(l) for l in dome), default=0)
    room = max(30, width - (art_cols + 3 if dome else 2))

    right.extend(_nav_line(items, cursor, room))
    right.append(_truncate(f"{c(DIM)}  {items[cursor].hint}{RESET}", room))
    right.append("")
    # Nothing apparatus-specific here. The limits, the memory sizes and the
    # notebook count belong to one lab, and no lab has been chosen yet at this
    # point -- so they were showing the default apparatus's, read as though they
    # were the session's. Per-lab facts are on the lab screen, which is where the
    # choice is actually made, and the limits are printed again under the loading
    # bar once the lab is open.
    for line in info.get("lines", []):
        right.append(_truncate(line, room))

    body = _side_by_side(dome, right) if dome else right
    return body


def _status_bar(width: int, info: Dict[str, str]) -> List[str]:
    rule = f"{c(DIM)}{'─' * max(20, width - 2)}{RESET}"
    left = (f"{c(DIM)}↔ select   {c(TEXT)}enter{c(DIM)} open   "
            f"{c(TEXT)}space{c(DIM)} toggle   {c(TEXT)}q{c(DIM)} quit{RESET}")
    right = f"{c(DIM)}{info.get('model', '')}   {info.get('clock', '')}{RESET}"
    pad = max(1, width - 2 - _plain_len(left) - _plain_len(right))
    return [rule, f" {left}{' ' * pad}{right}"]


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def show(defaults: Dict[str, object], info_lines: Callable[[], List[str]],
         model: str = "") -> Optional[Dict[str, object]]:
    """Draw the boot screen and let the operator choose. None means quit.

    Falls back to a single printed frame when there is no terminal to drive --
    a piped or redirected run must not block waiting for a keypress that can
    never arrive.
    """
    coloured = enable_ansi()
    from superradiant_assistant.config import APPARATUS
    items = [
        Item("start", "Start", "choose a lab, then begin the session"),
        Item("live", "Live", "queue shots on real hardware; off replays history",
             kind="toggle", value=bool(defaults.get("live"))),
        Item("creative", "Creative",
             "let the agent write its own shots and analyses",
             kind="toggle", value=bool(defaults.get("creative"))),
        Item("thinking", "Thinking",
             "how hard the model deliberates before each tool call",
             kind="cycle", value=defaults.get("thinking", "low"),
             options=["minimal", "low", "medium", "high"]),
        # No Memory entry here. The memory screen is per apparatus -- MEMORY.md,
        # INSTRUCTIONS.md and the notebook all belong to one lab -- and no lab has
        # been chosen at this point, so it could only ever show the default one's,
        # which is the same confusion the limits/memory/notebook rows caused. It
        # is reachable as `/memory` once a session is open, and the lab screen
        # shows each lab's memory size while you pick.
        Item("quit", "Quit", "exit without starting"),
    ]

    def choices() -> Dict[str, object]:
        return {it.key: it.value for it in items if it.kind != "action"}

    if not coloured or not sys.stdin.isatty():
        print(_flat_header(items, model))
        return choices()

    cursor = 0
    # The one place dropping the scrollback is right: the app is taking over a
    # terminal it was just launched in, and the loading panel should start at
    # the top of a clean screen.
    sys.stdout.write(CLEAR_ALL + HIDE_CURSOR)
    try:
        while True:
            from datetime import datetime
            width = shutil.get_terminal_size((100, 30)).columns
            # By key, not by index: inserting a row above this once made the
            # status bar report the apparatus name as the live/offline state.
            by_key = {it.key: it for it in items}
            info = {
                "live": "on" if by_key["live"].value else "off",
                "model": model,
                "clock": datetime.now().strftime("%H:%M"),
                "lines": info_lines(str(defaults.get("apparatus") or APPARATUS)),
            }
            frame = compose(items, cursor, info, width)
            frame += _status_bar(width, info)
            sys.stdout.write(CLEAR + "\n".join(frame) + "\n")
            sys.stdout.flush()

            key = read_key()
            if key in ("right", "down", "\t"):
                cursor = (cursor + 1) % len(items)
            elif key in ("left", "up"):
                cursor = (cursor - 1) % len(items)
            elif key in (" ",):
                items[cursor].advance()
            elif key == "enter":
                it = items[cursor]
                if it.key == "start":
                    sys.stdout.write(CLEAR)
                    return choices()
                if it.key == "quit":
                    return None
                it.advance()
            elif key in ("q", "esc"):
                return None
    except KeyboardInterrupt:
        return None
    finally:
        sys.stdout.write(SHOW_CURSOR + RESET)
        sys.stdout.flush()


# --------------------------------------------------------------------------
# lab screen
# --------------------------------------------------------------------------

def _ago(when) -> str:
    """`2026-08-11 15:43  (today)` — the date, and how long ago in words."""
    from datetime import datetime
    if not when:
        return "never"
    d = (datetime.now().date() - when.date()).days
    rel = ("today" if d == 0 else "yesterday" if d == 1
           else f"{d} days ago" if d < 30 else f"{d // 30} month(s) ago")
    return f"{when:%Y-%m-%d %H:%M}  ({rel})"


def _lab_facts(name: str) -> List[tuple]:
    """Label/value rows for one lab, gathered without committing to it."""
    from datetime import datetime
    from superradiant_assistant import config as C
    from superradiant_assistant.memory.store import MemoryStore
    from superradiant_assistant.safety import load_global_specs

    restore = C.APPARATUS
    if name != restore:
        C.select_apparatus(name)
    try:
        specs = load_global_specs()
        store = MemoryStore()
        own = [p for p in (store.memory_file, store.instructions_file) if p.exists()]
        pages = [store.episode_path(d) for d in store.episode_days()]

        # "Last opened" is the newest thing the agent itself wrote for this lab.
        # Shot files are reported separately: data can arrive from runmanager
        # without the agent being involved at all.
        touched = [p.stat().st_mtime for p in own + pages if p.exists()]
        last_agent = datetime.fromtimestamp(max(touched)) if touched else None

        data = C.CONFIG.historical_data_root
        shots = sorted(data.glob("*.h5"), key=lambda p: p.stat().st_mtime) \
            if data.exists() else []
        last_shot = (datetime.fromtimestamp(shots[-1].stat().st_mtime)
                     if shots else None)

        rows = [("last opened", _ago(last_agent)),
                ("last shot", _ago(last_shot))]
        rows.append(("data", f"{len(shots)} file(s) in {data.name}"
                     if data.exists() else
                     f"{data} — missing"))
        # Two names, not three: the third was always the one clipped, and a
        # clipped identifier is worse than an honest count.
        rows.append(("globals",
                     f"{len(specs)}   " + ", ".join(list(specs)[:2])
                     + (f", +{len(specs) - 2} more" if len(specs) > 2 else "")
                     if specs else
                     "none declared — no sweep can be range-checked"))
        rows.append(("memory",
                     f"{sum(p.stat().st_size for p in own) / 1000:.1f} kB of its "
                     f"own, {len(pages)} day(s) of notebook" if own else
                     "a fresh branch — shares labscript know-how only"))
        return rows
    finally:
        if C.APPARATUS != restore:
            C.select_apparatus(restore)


def lab_screen(current: str = "") -> Optional[str]:
    """Pick which lab to open. Returns a name, NEW_APPARATUS, or None to go back.

    Its own screen rather than a row on the boot menu: the choice decides the
    globals, the safe ranges, the data root and the memory branch, and each lab
    needs a few lines of its own to be told apart. Cycling a one-line value could
    not show any of that.
    """
    from superradiant_assistant.config import available_apparatus, APPARATUS

    labs = available_apparatus()
    rows = labs + [NEW_APPARATUS]
    cursor = rows.index(current) if current in rows else (
        rows.index(APPARATUS) if APPARATUS in rows else 0)

    if not enable_ansi() or not sys.stdin.isatty():
        return current or APPARATUS

    facts: Dict[str, List[tuple]] = {}
    art = load_art("mit_night_dome.ansi")
    art_w = max((_plain_len(l) for l in art), default=0)
    sys.stdout.write(ALT_ON + HIDE_CURSOR)
    try:
        while True:
            size = shutil.get_terminal_size((100, 30))
            total = min(size.columns - 2, 108)
            # Beside the boxes, not above them. A recognisable building needs
            # height -- the crop that fitted in ten rows read as "some columns"
            # and not as MIT -- and twenty-three rows of banner plus the boxes
            # would not fit a terminal at all. Dropped entirely when the window
            # is too narrow to give the boxes room.
            show_art = bool(art) and total >= art_w + 52
            width = total - (art_w + 3) if show_art else total

            frame: List[str] = []
            right: List[str] = [
                f" {bold(c(INK_RED) + 'CHOOSE A LAB' + RESET)}",
                f" {c(DIM)}each lab keeps its own globals, data and memory{RESET}",
                f" {c(DIM)}shared: how labscript behaves, and who you are{RESET}",
                "",
            ]

            for i, name in enumerate(rows):
                picked = i == cursor
                colour = ACCENT if picked else DIM

                # `_box` clips a long row without saying so, which read as
                # broken text: `sine_freque`, `y_bias_field_lo`. Every row is
                # truncated with an ellipsis before it gets there.
                room = width - 4

                if name == NEW_APPARATUS:
                    title = f"{CURSOR} new lab" if picked else "  new lab"
                    tone = c(TEXT if picked else DIM)
                    inner = [
                        f"{tone}A name, a data folder, and the globals every "
                        f"sweep is checked against.{RESET}",
                        f"{c(DIM)}Starts a fresh memory branch; inherits the "
                        f"shared labscript know-how.{RESET}",
                    ]
                    right += _box(title, [_truncate(l, room, "…") for l in inner],
                                  width, GREEN if picked else DIM)
                    continue

                if name not in facts:
                    facts[name] = _lab_facts(name)
                title = (f"{CURSOR} {name}" if picked else f"  {name}")
                if name == APPARATUS:
                    title += "  ·  current"
                label_w = max(len(k) for k, _ in facts[name])
                inner = [
                    _truncate(
                        f"{c(DIM)}{k:<{label_w}}  "
                        f"{c(TEXT if picked else DIM)}{v}{RESET}", room, "…")
                    for k, v in facts[name]
                ]
                right += _box(title, inner, width, colour)

            right.append("")
            right.append(f" {c(DIM)}↑↓ move   {c(TEXT)}enter{c(DIM)} open   "
                         f"{c(TEXT)}q{c(DIM)} back{RESET}")

            frame = _side_by_side(art, right) if show_art else right
            sys.stdout.write(CLEAR + "\n".join(frame) + "\n")
            sys.stdout.flush()

            key = read_key()
            if key in ("down", "j", "\t"):
                cursor = (cursor + 1) % len(rows)
            elif key in ("up", "k"):
                cursor = (cursor - 1) % len(rows)
            elif key == "enter":
                return rows[cursor]
            elif key in ("q", "esc"):
                return None
    except KeyboardInterrupt:
        return None
    finally:
        sys.stdout.write(SHOW_CURSOR + RESET + ALT_OFF)
        sys.stdout.flush()


# --------------------------------------------------------------------------
# team screen
# --------------------------------------------------------------------------

THINKING_LEVELS = ["minimal", "low", "medium", "high"]

ROLE_BLURB = {
    "lead": "coordinates; may consult the others",
    "planner": "decides what to measure; cannot reach hardware",
    "coder": "writes the code and runs the sweeps",
    "advisor": "physicist; diagnoses results, changes nothing",
}

#: Tools grouped by what they do to the apparatus, most dangerous last. A flat
#: alphabetical list of twenty-five names says nothing about which of them can
#: fire a shot.
_TOOL_GROUPS = [
    ("plan", ["set_plan", "update_plan"]),
    ("read", ["list_scripts", "inspect_shot", "read_lab_file", "list_lab_files",
              "search_lab_knowledge", "load_skill", "read_shot_results",
              "analyze_results", "get_runmanager_globals",
              "list_lab_history", "read_notebook", "read_report"]),
    ("lyse", ["set_lyse_routines", "get_lyse_routines"]),
    ("web", ["search_web", "fetch_web_page"]),
    ("mailbox", ["send_note", "read_notes"]),
    ("hardware", ["load_sequence", "set_runmanager_global", "engage_shot",
                  "run_sweep", "run_optimization"]),
    ("writes code", ["propose_global", "write_shot", "write_analysis",
                     "save_experiment_skill", "write_report"]),
    ("delegate", ["ask_planner", "ask_coder", "ask_advisor"]),
]

_GROUP_COLOUR = {
    "plan": DIM, "read": TEXT, "lyse": TEXT, "web": TEXT,
    "mailbox": ACCENT,
    "hardware": AMBER, "writes code": AMBER, "delegate": ACCENT,
}


def _truncate(s: str, limit: int, mark: str = "") -> str:
    """Cut a coloured string to `limit` visible columns, keeping the escapes.

    `mark` is appended when something was actually cut — pass "…" wherever the
    reader could mistake a clipped line for the whole value. `sine_freque` reads
    as a typo; `sine_freq…` reads as "there is more".
    """
    if _plain_len(s) <= limit:
        return s
    import re
    limit = max(0, limit - len(mark))
    out, seen = [], 0
    for piece in re.split(r"(\x1b\[[0-9;]*m)", s):
        if piece.startswith("\x1b"):
            out.append(piece)
            continue
        room = limit - seen
        if room <= 0:
            break
        out.append(piece[:room])
        seen += len(piece[:room])
    return "".join(out) + mark + RESET


#: Colours for the bracketed source tags in progress output, so a scrolling
#: transcript can be skimmed by who is speaking rather than read line by line.
TAG_COLOURS = {
    "sweep": ACCENT,
    "live": GREEN,
    "memory": DIM,
    "sequence": TEXT,
    "config": DIM,
    "LLM": AMBER,
}


def tag(name: str) -> str:
    """`[sweep]` in that source's colour."""
    return f"{c(TAG_COLOURS.get(name, TEXT))}[{name}]{RESET}"


def status_line(text: str) -> None:
    """Overwrite one line in place, for progress that repeats.

    A poll loop that prints 'N analysed...' once per second turned a 26-shot
    sweep into 26 near-identical lines, burying the queue and the result either
    side of it. Only the current count is worth a line.
    """
    if sys.stdout.isatty():
        sys.stdout.write("\r\x1b[2K  " + text)
        sys.stdout.flush()
    else:
        print("  " + text, flush=True)


def end_status_line() -> None:
    """Finish a status line so the next print starts cleanly."""
    if sys.stdout.isatty():
        sys.stdout.write("\r\x1b[2K")
        sys.stdout.flush()


def _table_block(rows: List[str], width: int) -> Optional[List[str]]:
    """Render consecutive Markdown pipe-table lines as an aligned table.

    Left alone, a table arrives as `| a | b |` text that the wrapper folds at
    arbitrary points and a `|---|---|` separator row that means nothing to a
    reader. Aligning the columns is the whole value of a table.
    """
    cells = []
    for r in rows:
        body = r.strip().strip("|")
        parts = [p.strip() for p in body.split("|")]
        if set("".join(parts)) <= set("-: "):        # the separator row
            continue
        cells.append(parts)
    if len(cells) < 2:
        return None

    ncol = max(len(r) for r in cells)
    cells = [r + [""] * (ncol - len(r)) for r in cells]
    widths = [max(len(r[i]) for r in cells) for i in range(ncol)]

    # Shrink the widest columns until the table fits.
    while sum(widths) + 3 * ncol + 1 > width and max(widths) > 6:
        widths[widths.index(max(widths))] -= 1

    def render(row, colour):
        out = []
        for i, cell in enumerate(row):
            txt = cell if len(cell) <= widths[i] else cell[: widths[i] - 1] + "…"
            out.append(f"{txt:<{widths[i]}}")
        return f"{c(DIM)}│{RESET} " + f" {c(DIM)}│{RESET} ".join(
            f"{c(colour)}{x}{RESET}" for x in out) + f" {c(DIM)}│{RESET}"

    rule = f"{c(DIM)}├─" + "─┼─".join("─" * w for w in widths) + f"─┤{RESET}"
    top = f"{c(DIM)}┌─" + "─┬─".join("─" * w for w in widths) + f"─┐{RESET}"
    bot = f"{c(DIM)}└─" + "─┴─".join("─" * w for w in widths) + f"─┘{RESET}"

    out = [top, render(cells[0], ACCENT), rule]
    out += [render(r, TEXT) for r in cells[1:]]
    out.append(bot)
    return out


def _inline(s: str, base: str) -> str:
    """Turn inline Markdown into terminal styling.

    The model writes Markdown because that is what it writes everywhere; a
    terminal renders none of it, so `**not connected**` arrives with the
    asterisks intact and the emphasis lost. Each span is closed by returning to
    `base`, or the rest of the line inherits the span's colour.
    """
    import re
    out = re.sub(r"\*\*(.+?)\*\*",
                 lambda m: f"\x1b[1m{c(TEXT)}{m.group(1)}\x1b[22m{base}", s)
    out = re.sub(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])",
                 lambda m: f"{c(TEXT)}{m.group(1)}{base}", out)
    out = re.sub(r"`([^`]+)`",
                 lambda m: f"{c(ACCENT)}{m.group(1)}{base}", out)
    # LaTeX delimiters are noise here; the expression inside is still readable.
    out = re.sub(r"\$([^$\n]+?)\$", lambda m: f"{c(ACCENT)}{m.group(1)}{base}", out)
    return out


def render_markdown(text: str, width: int) -> List[str]:
    """Markdown to terminal rows, wrapped to `width`.

    Wrapping happens on the plain text and styling is applied afterwards:
    folding a line that already contains escape sequences counts them as
    characters and breaks it far too early.
    """
    rows: List[str] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        # Gather a run of pipe-table lines and render them as one table.
        if stripped.startswith("|") and stripped.count("|") >= 2:
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            table = _table_block(block, width)
            rows.extend(table if table else
                        [f"{c(TEXT)}{l.strip()}{RESET}" for l in block])
            continue
        i += 1

        if not stripped:
            rows.append("")
            continue
        if set(stripped) <= set("-=_") and len(stripped) >= 3:
            rows.append(f"{c(DIM)}{'─' * max(4, width - 2)}{RESET}")
            continue

        if stripped.startswith("#"):
            depth = len(stripped) - len(stripped.lstrip("#"))
            body = stripped[depth:].strip()
            # `###` is the level the model reaches for most, so it has to be
            # one of the visible ones or a reply's only structure is invisible.
            for line in _wrap(body, width):
                rows.append(bold(c(ACCENT if depth <= 3 else TEXT) + line) + RESET)
            continue

        bullet, indent = "", " " * (len(raw) - len(raw.lstrip()))
        body = stripped
        if stripped[:2] in ("- ", "* ", "+ "):
            bullet, body = f"{c(DIM)}·{RESET} ", stripped[2:]
        else:
            import re
            m = re.match(r"^(\d+[.)])\s+(.*)$", stripped)
            if m:
                bullet, body = f"{c(ACCENT)}{m.group(1)}{RESET} ", m.group(2)

        pad = len(indent) + (2 if bullet else 0)
        wrapped = _wrap(body, max(8, width - pad))
        # NOT `i` -- the outer loop owns that, and reusing it here reset the
        # line index after every paragraph and looped forever.
        for k, line in enumerate(wrapped):
            lead = indent + (bullet if k == 0 else " " * 2)
            rows.append(f"{lead}{c(TEXT)}{_inline(line, c(TEXT))}{RESET}")
    return rows


def reply_panel(text: str, title: str = "assistant", width: int = 0,
                colour=None) -> str:
    """The agent's answer, rendered and framed so it is not just more scroll."""
    width = width or min(shutil.get_terminal_size((100, 30)).columns - 2, 100)
    rows = render_markdown(text.strip(), width - 4)
    while rows and not rows[0].strip():
        rows.pop(0)
    while rows and not rows[-1].strip():
        rows.pop()
    return "\n".join(_box(title, rows or ["(no reply)"], width,
                          colour or CARDINAL))


#: A small envelope glyph, one panel row tall on either side of it. Static -- no
#: animation, no cursor movement -- so it renders identically to a pipe, a log
#: file, or a real terminal. `_box` draws the one frame around both the glyph and
#: the status text, with "email" as its title, so the two read as one delivery
#: rather than a picture next to an unrelated line.
_ENVELOPE_GLYPH = ("┌────┐",
                    "│\\  /│",
                    "│ \\/ │",
                    "└────┘")


def envelope_sent(to: str, chars: int = 0) -> None:
    """Announce a note leaving the mailbox: one frame around glyph and status."""
    status = (f"Mail has been sent to {to}"
              + (f" ({chars:,} chars)" if chars else ""))
    rows = list(_ENVELOPE_GLYPH)
    mid = len(rows) // 2
    rows[mid] = f"{rows[mid]}  {c(SEAL_RED)}{status}{RESET}"
    width = max(_plain_len(r) for r in rows) + 4
    print("\n".join(_box("email", rows, width, SEAL_RED)))


def advisor_panel(text: str, model: str = "", thinking: str = "",
                   width: int = 0) -> str:
    """The advisor's answer: a blue frame with its portrait in the header.

    Summoning the advisor is a different act from talking to the lead -- a
    different mind, on a different model, that is allowed to disagree with you --
    and a transcript where every reply is framed the same cardinal red does not
    say so. The colour and the face are there to be recognised from across the
    scroll, before any of it is read.

    The portrait is optional. With no `assets/advisor.ansi` the header is just the
    identity lines, which is the correct look on a machine where the file was
    never rendered.
    """
    width = width or min(shutil.get_terminal_size((100, 30)).columns - 2, 100)
    art = [ln for ln in load_art("advisor.ansi") if ln.strip()]

    ident = [f"{bold(c(ADVISOR_BLUE) + 'Advisor')}{RESET}",
             f"{c(DIM)}{ROLE_BLURB.get('advisor', '')}{RESET}"]
    if model:
        ident.append(f"{c(DIM)}{model}"
                     + (f" · thinking {thinking}" if thinking else "")
                     + RESET)
    ident.append(f"{c(DIM)}consulted; it reads and reasons and changes "
                 f"nothing{RESET}")

    rows: List[str] = []
    if art:
        # The identity block sits vertically centred against the face rather than
        # at its top: aligned to the top it reads as a caption that ran short.
        pad_top = max(0, (len(art) - len(ident)) // 2)
        beside = [""] * pad_top + ident
        gutter = " " * 3
        for i, line in enumerate(art):
            right = beside[i] if i < len(beside) else ""
            rows.append(f"{line}{gutter}{right}")
        rows.append("")
    else:
        rows.extend(ident)
        rows.append("")

    body = render_markdown(text.strip(), width - 4)
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    rows.extend(body or ["(no reply)"])
    return "\n".join(_box("advisor", rows, width, ADVISOR_BLUE))


def framed(title: str, body: str, width: int = 0, colour=None) -> str:
    """A titled frame around arbitrary text, for use outside this module.

    The plan and the confirmation prompt were delimited by rows of `=` and were
    indistinguishable from each other and from tool output in a scrolling
    transcript. A frame with a title says what you are looking at without
    reading it.
    """
    width = width or min(shutil.get_terminal_size((100, 30)).columns - 2, 100)
    # A list of lines is the obvious thing to pass and it used to crash here.
    if not isinstance(body, str):
        body = "\n".join(str(b) for b in body)
    rows: List[str] = []
    for line in body.split("\n"):
        rows.extend(_wrap(line, width - 4) if len(line) > width - 4 else [line])
    return "\n".join(_box(title, rows, width, colour))


def _box(title: str, rows: List[str], width: int, colour=None) -> List[str]:
    """A framed panel exactly `width` columns wide.

    The frame is drawn from visible-column counts only. Measuring the header
    with the escape sequences still in it is what made an intended 116-column
    box render at 172 and wrap.
    """
    inner = max(24, width - 2)                 # columns between the corners
    edge = c(colour or DIM)
    head = c(colour) if colour else c(TEXT)
    lead = f"─ {title} "
    out = [f"{edge}┌{lead}{'─' * max(0, inner - len(lead))}┐{RESET}"
           .replace(f"─ {title} ", f"─ {head}{title}{edge} ", 1)]
    for r in rows:
        r = _truncate(r, inner - 2)
        pad = " " * max(0, inner - 2 - _plain_len(r))
        out.append(f"{edge}│{RESET} {r}{pad} {edge}│{RESET}")
    out.append(f"{edge}└{'─' * inner}┘{RESET}")
    return out


def _group_tools(names: List[str], width: int) -> List[str]:
    """Tool names laid out in labelled, aligned columns."""
    left = set(names)
    rows: List[str] = []
    per_row = max(2, (width - 18) // 22)
    for group, members in _TOOL_GROUPS:
        present = [m for m in members if m in left]
        if not present:
            continue
        left -= set(present)
        colour = _GROUP_COLOUR.get(group, TEXT)
        for i in range(0, len(present), per_row):
            chunk = present[i:i + per_row]
            label = group if i == 0 else ""
            cells = "".join(f"{n:<22}" for n in chunk).rstrip()
            rows.append(f"{c(DIM)}{label:<13}{c(colour)}{cells}{RESET}")
    for i in range(0, len(sorted(left)), per_row):
        chunk = sorted(left)[i:i + per_row]
        cells = "".join(f"{n:<22}" for n in chunk).rstrip()
        rows.append(f"{c(DIM)}{'other':<13}{c(TEXT)}{cells}{RESET}")
    return rows


def _level_picker(level: str, selected: bool) -> str:
    i = THINKING_LEVELS.index(level) if level in THINKING_LEVELS else 1
    if selected:
        arrows = c(ACCENT) + "<" , c(ACCENT) + ">"
        body = bold(f"{c(ACCENT)}{level:^7}")
    else:
        arrows = c(DIM) + " ", c(DIM) + " "
        body = f"{c(DIM)}{level:^7}"
    scale = "".join(
        (c(ACCENT) if selected else c(DIM)) + ("=" if n <= i else "·")
        for n in range(len(THINKING_LEVELS)))
    return f"{arrows[0]}{body}{arrows[1]}{RESET} {scale}{RESET}"


def model_problem(name: str) -> str:
    """Why this model name cannot be used, or "" if it can.

    A name is only checked for which provider it claims, not against a list of
    releases -- a list would reject the next model the day it ships. What it
    does catch is the typo, and a typo is expensive here: the switch succeeds,
    and then every message fails with a 404 until someone notices.
    """
    name = (name or "").strip()
    if not name:
        return "no model name given"
    if "claude" in name.lower() or "gemini" in name.lower():
        return ""
    return (f"'{name}' names neither provider — a Gemini model has 'gemini' in "
            f"its name, a Claude model has 'claude'")


#: Wide enough for the longest name either provider currently ships, and fixed
#: so the role blurbs beside it stay in a column while a name is being typed.
_MODEL_COL = 18


def _model_field(model: str, selected: bool, editing: Optional[str]) -> str:
    """The model column: a value, or the text box while it is being typed."""
    if editing is not None:
        pad = " " * max(0, _MODEL_COL - len(editing) - 3)
        return f"{c(ACCENT)}[{bold(editing)}{c(ACCENT)}█]{RESET}{pad}"
    shown = model if len(model) <= _MODEL_COL else model[:_MODEL_COL - 1] + "…"
    shown = shown.ljust(_MODEL_COL)
    if selected:
        return f"{bold(c(ACCENT) + shown)}{RESET}"
    return f"{c(DIM)}{shown}{RESET}"


def team_screen(team, registry, extra_tools=None):
    """Show the team, and set each agent's thinking level and model separately.

    Returns ({agent: level}, {agent: model}) for whatever the operator left it
    on. Changes are applied by the caller, because rebuilding a chat session is
    the caller's business, not this module's.

    The model is per agent because the agents do different jobs: the coder
    writes the code that reaches hardware, the answer agent formats text, and
    there is no reason those have to be the same model. `/model` moves the whole
    team at once; this is where one of them is moved on its own.
    """
    names = list(team)
    extra_tools = extra_tools or {}
    levels = {n: getattr(team[n], "thinking_level", "low") for n in names}
    models = {n: getattr(team[n], "model", "") for n in names}
    coloured = enable_ansi()

    if not coloured or not sys.stdin.isatty():
        for n in names:
            tools = registry.names_for(n) + extra_tools.get(n, [])
            print(f"  {n:<9}{len(tools)} tools  thinking {levels[n]}  "
                  f"model {models[n]}")
            print(f"    {', '.join(tools)}")
        return levels, models

    cursor = 0
    editing: Optional[str] = None      # the buffer while a model is being typed
    complaint = ""                     # why the last entry was not accepted
    sys.stdout.write(ALT_ON + HIDE_CURSOR)
    try:
        while True:
            width = min(shutil.get_terminal_size((100, 30)).columns - 2, 118)
            rows = []
            for i, n in enumerate(names):
                tools = registry.names_for(n) + extra_tools.get(n, [])
                mark = f"{c(ACCENT)}{CURSOR}" if i == cursor else " "
                name = (bold(f"{c(ACCENT)}{n:<9}") if i == cursor
                        else f"{c(TEXT)}{n:<9}")
                field = _model_field(models[n], i == cursor,
                                     editing if i == cursor else None)
                rows.append(
                    f"{mark} {name}{RESET}{c(DIM)}{len(tools):>3} tools   "
                    f"{RESET}{_level_picker(levels[n], i == cursor)}"
                    f"   {field}   {c(DIM)}{ROLE_BLURB.get(n, '')}{RESET}")

            sel = names[cursor]
            sel_tools = registry.names_for(sel) + extra_tools.get(sel, [])
            frame = _box("Team", rows, width)
            frame += _box(f"{sel} · {len(sel_tools)} tools",
                          _group_tools(sel_tools, width), width)
            frame.append("")
            if editing is not None:
                frame.append(f"  {c(ACCENT)}typing {sel}'s model{RESET}"
                             f"{c(DIM)}    {c(TEXT)}enter{c(DIM)} apply    "
                             f"{c(TEXT)}esc{c(DIM)} cancel{RESET}")
            else:
                frame.append(f"  {c(DIM)}↑↓ agent    ←→ thinking effort    "
                             f"{c(TEXT)}m{c(DIM)} this agent's model    "
                             f"{c(TEXT)}enter{c(DIM)} apply and go back    "
                             f"{c(TEXT)}q{c(DIM)} back{RESET}")
            if complaint:
                frame.append(f"  {c(AMBER)}{complaint}{RESET}")

            sys.stdout.write(CLEAR + "\n".join(frame) + "\n")
            sys.stdout.flush()

            key = read_key()

            if editing is not None:
                # While typing, every key belongs to the text box -- otherwise
                # the 'm' in 'gemini' would open a second one.
                if key == "enter":
                    problem = model_problem(editing)
                    if problem:
                        complaint = problem
                    else:
                        models[sel], editing, complaint = editing.strip(), None, ""
                elif key == "esc":
                    editing, complaint = None, ""
                elif key in ("\x08", "\x7f"):
                    editing = editing[:-1]
                elif len(key) == 1 and key.isprintable():
                    editing += key
                continue

            if key in ("down", "j", "\t"):
                cursor = (cursor + 1) % len(names)
            elif key in ("up", "k"):
                cursor = (cursor - 1) % len(names)
            elif key in ("right", "l"):
                i = THINKING_LEVELS.index(levels[names[cursor]])
                levels[names[cursor]] = THINKING_LEVELS[
                    min(i + 1, len(THINKING_LEVELS) - 1)]
            elif key in ("left", "h"):
                i = THINKING_LEVELS.index(levels[names[cursor]])
                levels[names[cursor]] = THINKING_LEVELS[max(i - 1, 0)]
            elif key == "m":
                # Prefilled with what it is on now: most changes are one version
                # apart, and it also shows the shape of a name that works.
                editing, complaint = models[names[cursor]], ""
            elif key in ("enter", "q", "esc"):
                return levels, models
    except KeyboardInterrupt:
        return levels, models
    finally:
        # Leaving the alternate buffer restores the screen underneath: the
        # loading panel, and everything said since.
        sys.stdout.write(SHOW_CURSOR + RESET + ALT_OFF)
        sys.stdout.flush()


# --------------------------------------------------------------------------
# memory screen
# --------------------------------------------------------------------------

def _style_markdown(line: str) -> str:
    """Light styling for the memory files, which are hand-written Markdown.

    Not a renderer -- the point is to make headings and emphasis findable while
    scrolling, without reflowing text the operator may have written by hand.
    """
    import re
    stripped = line.lstrip()
    indent = " " * (len(line) - len(stripped))

    if stripped.startswith("#"):
        depth = len(stripped) - len(stripped.lstrip("#"))
        body = stripped[depth:].strip()
        colour = ACCENT if depth <= 2 else TEXT
        return f"{indent}{bold(c(colour) + body)}{RESET}"
    if stripped.startswith(("---", "===")):
        return f"{c(DIM)}{indent}{stripped}{RESET}"

    out = stripped
    if out.startswith(("- ", "* ")):
        out = f"{c(DIM)}·{RESET} {out[2:]}"
    out = re.sub(r"\*\*(.+?)\*\*", lambda m: bold(c(TEXT) + m.group(1)) + c(DIM),
                 out)
    out = re.sub(r"`([^`]+)`", lambda m: c(ACCENT) + m.group(1) + c(DIM), out)
    return f"{c(DIM)}{indent}{out}{RESET}"


def _wrap(text: str, width: int) -> List[str]:
    """Split on newlines, then fold anything wider than the panel."""
    out: List[str] = []
    for raw in text.split("\n"):
        if len(raw) <= width:
            out.append(raw)
            continue
        indent = " " * (len(raw) - len(raw.lstrip()))
        line = ""
        for word in raw.split():
            candidate = f"{line} {word}".strip()
            if len(indent + candidate) > width and line:
                out.append(indent + line)
                line = word
            else:
                line = candidate
        out.append(indent + line)
    return out


def memory_screen() -> None:
    """What the agent carries between sessions, as a screen rather than a dump.

    The three layers are separate pages because they answer different
    questions -- what the apparatus is, who the operator is, what happened today
    -- and because printing seven thousand characters into the scrollback made
    all three unreadable at once.
    """
    from superradiant_assistant.memory import MEMORY

    def stamp(p) -> str:
        if not p.exists():
            return "not written yet"
        from datetime import datetime
        size = p.stat().st_size
        when = datetime.fromtimestamp(p.stat().st_mtime)
        return f"{size / 1000:.1f} kB   {when:%Y-%m-%d %H:%M}"

    episode = MEMORY.episode_path()
    # Every day that has a page, so the notebook can be read back and not only
    # written. It opens on today.
    days = MEMORY.episode_days() or [episode.stem]
    day_i = days.index(episode.stem) if episode.stem in days else len(days) - 1

    from superradiant_assistant.memory.store import CONFIG_APPARATUS
    lab = CONFIG_APPARATUS()

    def build_layers(i: int):
        d = days[i]
        # Ordered by how far each travels: the first two follow the code into any
        # lab, the next two belong to this apparatus alone, the last is today.
        return [
            ("labscript · shared", MEMORY.read_labscript(),
             "how labscript, BLACS and lyse behave — true in every lab"),
            ("Operator · shared", MEMORY.read_user(),
             "who runs this and how they work — the same person everywhere"),
            (f"Apparatus · {lab}", MEMORY.read_memory(),
             "this apparatus only: hardware, wiring, what has been measured"),
            (f"Instructions · {lab}", MEMORY.read_instructions(),
             "standing rules the operator gave for this apparatus"),
            (f"Notebook · {d}.md", MEMORY.read_episode(d),
             f"the day's page: summary, next steps, log"
             f"   ({i + 1} of {len(days)} days)"),
        ]

    layers = build_layers(day_i)
    injected = len(MEMORY.build_context_block())

    hist = MEMORY.history_file
    turns = 0
    if hist.exists():
        try:
            turns = sum(1 for _ in hist.open(encoding="utf-8"))
        except OSError:
            turns = -1

    header = [
        f"{c(DIM)}{'lab':<10}{c(ACCENT)}{lab}{c(DIM)}   own memory at "
        f"{c(TEXT)}{MEMORY.root}{RESET}",
        f"{c(DIM)}{'shared':<10}{c(TEXT)}{MEMORY.shared_root}{RESET}",
        f"{c(DIM)}{'':<10}{c(TEXT)}LABSCRIPT.md   {stamp(MEMORY.labscript_file)}"
        f"{c(DIM)}   shared{RESET}",
        f"{c(DIM)}{'':<10}{c(TEXT)}OPERATOR.md    {stamp(MEMORY.operator_file)}"
        f"{c(DIM)}   shared{RESET}",
        f"{c(DIM)}{'':<10}{c(TEXT)}MEMORY.md      {stamp(MEMORY.memory_file)}"
        f"{c(DIM)}   {lab} only{RESET}",
        f"{c(DIM)}{'':<10}{c(TEXT)}INSTRUCTIONS.md {stamp(MEMORY.instructions_file)}"
        f"{c(DIM)}   {lab} only{RESET}",
        f"{c(DIM)}{'notebook':<10}{c(TEXT)}{episode.name}  {stamp(episode)}{RESET}",
        f"{c(DIM)}{'log':<10}{c(TEXT)}history.jsonl  {stamp(hist)}   "
        f"{turns} turns{c(DIM)}   raw, never injected{RESET}",
        (f"{c(DIM)}{'injected':<10}{c(GREEN)}{injected:,} characters{c(DIM)} go into "
         f"every system prompt{RESET}" if injected else
         f"{c(DIM)}{'injected':<10}{c(AMBER)}nothing — the agent starts each "
         f"session blank{RESET}"),
    ]

    if not enable_ansi() or not sys.stdin.isatty():
        print(f"\n  Memory root: {MEMORY.root}")
        block = MEMORY.build_context_block()
        print(f"\n{block}" if block else "  (memory is empty)")
        return

    page, top = 0, 0
    sys.stdout.write(ALT_ON + HIDE_CURSOR)
    try:
        while True:
            size = shutil.get_terminal_size((100, 30))
            width = min(size.columns - 2, 118)
            body_rows = max(6, size.lines - len(header) - 9)

            title, text, blurb = layers[page]
            lines = _wrap(text, width - 6) if text.strip() else ["(empty)"]
            top = max(0, min(top, max(0, len(lines) - body_rows)))
            shown = [_style_markdown(l) for l in lines[top:top + body_rows]]

            tabs = []
            for i, (t, txt, _) in enumerate(layers):
                name = t.split(" · ")[0]
                mark = f"{bold(c(ACCENT) + CURSOR + ' ' + name)}" if i == page \
                    else f"{c(DIM) if txt.strip() else c(DIM)}  {name}"
                tabs.append(mark + RESET)
            pos = (f"line {top + 1}-{min(top + body_rows, len(lines))} "
                   f"of {len(lines)}")

            frame = _box("Memory", header, width)
            frame += _box(title,
                          [f"{c(DIM)}·  {RESET}".join([]) or
                           f"{c(DIM)}{blurb}{RESET}", ""] + shown +
                          ["", f"{c(DIM)}{pos:>{width - 6}}{RESET}"], width)
            frame.append("")
            frame.append(f"  {'   '.join(tabs)}")
            keys = f"{c(DIM)}↑↓ scroll   ←→ layer"
            # The notebook is the last layer; day paging only applies there.
            if page == len(layers) - 1 and len(days) > 1:
                keys += (f"   {c(TEXT)}[ ]{c(DIM)} day "
                         f"{c(ACCENT)}{days[day_i]}{c(DIM)}")
            frame.append(f"  {keys}   {c(TEXT)}q{c(DIM)} back{RESET}")

            sys.stdout.write(CLEAR + "\n".join(frame) + "\n")
            sys.stdout.flush()

            key = read_key()
            if key in ("down", "j"):
                top += 1
            elif key in ("up", "k"):
                top -= 1
            elif key == "pgdn":
                top += body_rows
            elif key == "pgup":
                top -= body_rows
            elif key == "home":
                top = 0
            elif key == "end":
                top = len(lines)
            elif key in ("right", "l", "\t"):
                page, top = (page + 1) % len(layers), 0
            elif key in ("left", "h"):
                page, top = (page - 1) % len(layers), 0
            elif key in ("[", "]"):
                # Paging days is deliberately not bound to the arrows: those
                # move between layers, and one key doing two things depending
                # on which page you are on is how you press it by mistake.
                day_i = max(0, min(len(days) - 1,
                                   day_i + (1 if key == "]" else -1)))
                layers, top = build_layers(day_i), 0
            elif key in ("q", "esc", "enter"):
                return
    except KeyboardInterrupt:
        return
    finally:
        sys.stdout.write(SHOW_CURSOR + RESET + ALT_OFF)
        sys.stdout.flush()


# --------------------------------------------------------------------------
# skills screen
# --------------------------------------------------------------------------

def _skill_steps(body: str) -> List[str]:
    """The section headings of a SKILL.md, as its table of contents.

    A procedure's headings say what it covers without dumping the procedure --
    which is the whole reason skills are loaded on demand rather than preloaded.
    """
    out = []
    for line in body.split("\n"):
        s = line.strip()
        if s.startswith("## "):
            out.append(s[3:].strip())
    return out


def skills_screen() -> None:
    """What the agent can load, and what each one is for.

    Skills are documents, not scripts: `load_skill` puts the body of one into the
    model's context and the model then follows it. So this screen is a reading
    list, and there is deliberately nothing on it that runs anything.
    """
    from superradiant_assistant.skill_loader import SKILLS

    names = SKILLS.names()
    here = SKILLS.CURRENT_APPARATUS

    def label(n: str) -> str:
        app = SKILLS.apparatus_of(n)
        if not app:
            return ""
        return app if app == here else f"{app}, NOT this bench"

    if not names:
        print(f"\n  no skills installed — looked in {SKILLS.skills_dir}")
        return

    if not enable_ansi() or not sys.stdin.isatty():
        print(f"\n  {len(names)} skills in {SKILLS.skills_dir}\n")
        for n in names:
            tag = label(n)
            print(f"  {n}{f'  [{tag}]' if tag else ''}")
            print(f"    {SKILLS.skills[n]['meta'].get('description', '(none)')}")
        return

    cursor = 0
    sys.stdout.write(ALT_ON + HIDE_CURSOR)
    try:
        while True:
            size = shutil.get_terminal_size((100, 30))
            width = min(size.columns - 2, 118)

            header = [
                f"{c(DIM)}{'skills':<10}{c(TEXT)}{SKILLS.skills_dir}{RESET}",
                f"{c(DIM)}{'loaded by':<10}{c(TEXT)}load_skill{c(DIM)} — available "
                f"to every agent. The body goes into the model's context; nothing "
                f"here executes.{RESET}",
                f"{c(DIM)}{'this lab':<10}{c(ACCENT)}{here}{c(DIM)} — a skill for "
                f"another apparatus is marked, not hidden: the method is often "
                f"still what you want.{RESET}",
            ]

            rows = []
            for i, n in enumerate(names):
                tag = label(n)
                mark = f"{c(ACCENT)}{CURSOR}" if i == cursor else " "
                name = (bold(f"{c(ACCENT)}{n:<28}") if i == cursor
                        else f"{c(TEXT)}{n:<28}")
                colour = AMBER if tag and tag != here else DIM
                rows.append(f"{mark} {name}{RESET}{c(colour)}"
                            f"{('[' + tag + ']') if tag else '':<29}{RESET}"
                            f"{c(DIM)}{len(_skill_steps(SKILLS.skills[n]['body']))}"
                            f" sections{RESET}")

            sel = names[cursor]
            meta = SKILLS.skills[sel]["meta"]
            body = SKILLS.skills[sel]["body"]
            detail = _wrap(meta.get("description", "(no description)"), width - 6)
            detail = [f"{c(TEXT)}{l}{RESET}" for l in detail]
            steps = _skill_steps(body)
            if steps:
                detail.append("")
                detail.append(f"{c(DIM)}covers{RESET}")
                for s in steps:
                    detail.append(f"{c(DIM)}  ·  {c(TEXT)}{s}{RESET}")
            tags = meta.get("tags", "")
            if tags:
                detail.append("")
                detail.append(f"{c(DIM)}tags    {tags}{RESET}")
            # Relative to the directory already named in the header: the absolute
            # path is long enough to be truncated, and it is the same prefix
            # every time.
            try:
                rel = Path(SKILLS.skills[sel]['path']).relative_to(SKILLS.skills_dir)
            except ValueError:
                rel = Path(SKILLS.skills[sel]['path'])
            detail.append(f"{c(DIM)}file    {rel}{RESET}")
            detail.append(f"{c(DIM)}length  {len(body.splitlines())} lines, "
                          f"{len(body):,} characters — this much enters the "
                          f"context when it is loaded{RESET}")
            if label(sel) and label(sel) != here:
                detail.append("")
                detail.append(f"{c(AMBER)}Its sequences, globals and metrics do "
                              f"not exist on this bench. Read it for method.{RESET}")

            frame = _box("Skills", header, width)
            frame += _box(f"{len(names)} available", rows, width)
            frame += _box(sel, detail, width)
            frame.append("")
            # The hint does not name the selected skill: it would jump on every
            # keypress, and at 80 columns the longest name overflowed the line.
            frame.append(f"  {c(DIM)}↑↓ skill    {c(TEXT)}q{c(DIM)} back    "
                         f"to use one, ask for it by name: "
                         f"{c(TEXT)}load X and run it{RESET}")

            sys.stdout.write(CLEAR + "\n".join(frame) + "\n")
            sys.stdout.flush()

            key = read_key()
            if key in ("down", "j", "\t"):
                cursor = (cursor + 1) % len(names)
            elif key in ("up", "k"):
                cursor = (cursor - 1) % len(names)
            elif key in ("q", "esc", "enter"):
                return
    except KeyboardInterrupt:
        return
    finally:
        sys.stdout.write(SHOW_CURSOR + RESET + ALT_OFF)
        sys.stdout.flush()


# --------------------------------------------------------------------------
# loading screen
# --------------------------------------------------------------------------

#: Block Elements, so the bar's leading edge can move in eighth-cell steps
#: rather than jumping a whole character at a time. All of these exist in
#: Consolas; braille spinners do not, and draw as replacement boxes.
_EIGHTHS = " ▏▎▍▌▋▊▉"
_FULL = "█"
_TRACK = "░"


class Loader:
    """A progress screen for the work between 'Start' and the first prompt.

    The steps are real: building the tool registry reads recent shot files, and
    reaching the labscript suite spawns a subprocess and waits on a socket that
    may never answer. That last one takes seconds when the GUIs are up and the
    better part of a minute when they are not, which is exactly the wait that
    used to look like a hang.

    So the bar tracks completed steps and the spinner animates independently --
    a bar that invents its own progress is worse than no bar, because it stops
    being evidence of anything.
    """

    def __init__(self, steps: List[str], width: int = 0, enabled: bool = True):
        self.labels = list(steps)
        self.details: Dict[str, str] = {}
        self.state: Dict[str, str] = {s: "pending" for s in steps}
        self.current: Optional[str] = None
        self.enabled = enabled and sys.stdout.isatty()
        self.width = width or shutil.get_terminal_size((100, 30)).columns
        self._logo = load_art("mit_logo.ansi")
        self._lines_drawn = 0
        self._phase = 0
        self._stop = None
        self._thread = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self):
        if self.enabled:
            sys.stdout.write(CLEAR + HIDE_CURSOR)
            self._spin_start()
        return self

    def __exit__(self, *exc):
        self._spin_stop()
        if self.enabled:
            self._draw()
            sys.stdout.write(SHOW_CURSOR + RESET + "\n")
            sys.stdout.flush()
        return False

    def _spin_start(self):
        import threading
        self._stop = threading.Event()

        def tick():
            while not self._stop.wait(0.08):
                self._phase += 1
                self._draw()

        self._thread = threading.Thread(target=tick, daemon=True)
        self._thread.start()

    def _spin_stop(self):
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)

    # -- reporting ---------------------------------------------------------

    def step(self, label: str) -> None:
        if self.current and self.state.get(self.current) == "active":
            self.state[self.current] = "done"
        self.current = label
        self.state[label] = "active"
        self._draw()

    def ok(self, detail: str = "") -> None:
        if self.current:
            self.state[self.current] = "done"
            if detail:
                self.details[self.current] = detail
        self._draw()

    def fail(self, detail: str = "") -> None:
        if self.current:
            self.state[self.current] = "failed"
            self.details[self.current] = detail or "failed"
        self._draw()

    def skip(self, detail: str = "") -> None:
        if self.current:
            self.state[self.current] = "skipped"
            self.details[self.current] = detail or "not needed"
        self._draw()

    # -- drawing -----------------------------------------------------------

    def _bar(self, done: int, total: int, cells: int = 34) -> str:
        frac = done / total if total else 0.0
        exact = frac * cells
        whole = int(exact)
        # The eighth-block leading edge is where the motion lives while a slow
        # step runs; without it the bar sits frozen and reads as a hang.
        if whole >= cells:
            head, rest = "", 0
        else:
            drift = (self._phase % 8) if self.current else int((exact % 1) * 8)
            head, rest = _EIGHTHS[drift], 1
        filled = _FULL * whole + head
        track = _TRACK * max(0, cells - whole - rest)
        pct = f"{done}/{total}"
        return f"{c(CARDINAL)}{filled}{c(DIM)}{track}{RESET}  {c(TEXT)}{pct}{RESET}"

    def _rows(self) -> List[str]:
        marks = {
            "pending": (f"{c(DIM)}  ·  ", DIM),
            "active": (f"{c(ACCENT)}  {CURSOR}  ", ACCENT),
            "done": (f"{c(GREEN)}  +  ", TEXT),
            "failed": (f"{c(INK_RED)}  !  ", INK_RED),
            "skipped": (f"{c(DIM)}  -  ", DIM),
        }
        # Indent the mark to the same column as the step list below it.
        out = [f"  {row}" for row in self._logo]
        out.append("")
        out.append(f"  {bold(c(TEXT) + 'Starting Labscript Agent')}{RESET}")
        out.append("")
        for label in self.labels:
            state = self.state[label]
            mark, colour = marks[state]
            detail = self.details.get(label, "")
            if state == "active" and not detail:
                detail = "." * (1 + self._phase % 3)
            out.append(f"{mark}{c(colour)}{label:<20}{RESET}"
                       f"{c(DIM)}{detail}{RESET}")
        out.append("")
        done = sum(1 for s in self.state.values()
                   if s in ("done", "skipped", "failed"))
        out.append(f"  {self._bar(done, len(self.labels))}")
        return out

    def _draw(self) -> None:
        if not self.enabled:
            return
        rows = self._rows()
        buf = []
        if self._lines_drawn:
            buf.append(f"\x1b[{self._lines_drawn}F")
        for r in rows:
            buf.append("\x1b[2K" + r + "\n")
        self._lines_drawn = len(rows)
        sys.stdout.write("".join(buf))
        sys.stdout.flush()


def _flat_header(items: List[Item], model: str) -> str:
    """No terminal to drive: state the same facts in one block and move on."""
    vals = "  ".join(f"{it.label}={it.display_value()}"
                     for it in items if it.kind != "action")
    return (f"{'=' * 70}\n  Labscript Agent Team\n"
            f"  model : {model}\n  {vals}\n{'=' * 70}")
