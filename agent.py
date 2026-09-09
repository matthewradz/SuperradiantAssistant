"""Interactive entry point for the labscript agent team.

Run:
    python agent.py                     # goal mode on
    python agent.py --goal-mode off     # one shot per optimize request
    python agent.py --live              # queue shots on real hardware

Commands inside the session:
    /goal on|off     toggle goal mode
    /team            show the team and each member's tools
    /memory          show what is currently in memory
    exit             quit
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from superradiant_assistant.config import CONFIG
from superradiant_assistant.hooks import default_gate
from superradiant_assistant.llm.client import (
    DEFAULT_MODEL, CLAUDE_DEFAULT_MODEL, is_claude,
)
from superradiant_assistant.llm.cost_tracker import CostTracker
from superradiant_assistant.memory import MEMORY
from superradiant_assistant.orchestrator.agents import build_team
from superradiant_assistant.skill_loader import SKILLS
from superradiant_assistant.tools import build_registry, configure_session


def _show_team(team, registry) -> None:
    """The team screen: who has which tools, and how hard each one thinks.

    Thinking effort is per agent because the agents do different jobs. The coder
    writes code that reaches hardware and is worth deliberation; the lead mostly
    picks the next tool, which is not, and that choice is what turned a
    twenty-call turn into 462 seconds.
    """
    from superradiant_assistant import splash as S

    # No `extra_tools` any more: the delegation tools are in the registry, so
    # `names_for("lead")` already lists them.
    chosen, models = S.team_screen(team, registry)
    changed = [f"{n} -> {lvl}" for n, lvl in chosen.items()
               if team[n].set_thinking(lvl)]
    if changed:
        print(f"  {S.c(S.ACCENT)}thinking effort updated: "
              f"{', '.join(changed)}{S.RESET}")
        print(f"  {S.c(S.DIM)}(each agent keeps its conversation; only the "
              f"deliberation budget changed){S.RESET}")

    # Applied one at a time, so the team can be mixed: the coder on Claude and
    # everyone else on Gemini is a valid state, and each keeps its own context.
    moved = [team[n].switch_model(m) for n, m in models.items()
             if m and m != team[n].model]
    if moved:
        print(f"  {S.c(S.ACCENT)}model changed{S.RESET}")
        for line in moved:
            print(f"    {line}")


def _print_effective_limits() -> None:
    """The ranges this process will actually enforce, under the loading bar.

    Printed after a lab is open, not on the boot menu: these belong to one
    apparatus, and on the boot menu no apparatus has been chosen yet.

    config.json is read once at import, so a running session keeps whatever the
    file said when it started. Editing the file mid-session silently changes
    nothing — printing the live values here is what makes that visible.
    """
    import shutil
    from superradiant_assistant import splash as S
    from superradiant_assistant.safety import load_global_specs

    label, indent = "limits", 22        # the column the loader's details start in
    specs = load_global_specs()
    if not specs:
        print(f"  {S.c(S.AMBER)}  !  {label:<20}{S.RESET}"
              f"{S.c(S.DIM)}none declared in config.json — no sweep can be "
              f"range-checked{S.RESET}")
        return

    room = max(40, min(shutil.get_terminal_size((100, 30)).columns - 2, 118) - indent)
    parts = [f"{n} {s.describe_range()}" if s.has_range else f"{n} (unbounded)"
             for n, s in specs.items()]

    # Wrapped, not truncated: a lab with eight globals ran the line 346 columns
    # wide once, and a limit that is off screen is not a limit anyone read.
    rows, line = [], ""
    for p in parts:
        candidate = f"{line}, {p}" if line else p
        if len(candidate) > room and line:
            rows.append(line)
            line = p
        else:
            line = candidate
    rows.append(line)

    print(f"  {S.c(S.GREEN)}  +  {S.c(S.TEXT)}{label:<20}{S.RESET}"
          f"{S.c(S.TEXT)}{rows[0]}{S.RESET}")
    for extra in rows[1:]:
        print(f"{' ' * (indent + 3)}{S.c(S.TEXT)}{extra}{S.RESET}")
    print(f"{' ' * (indent + 3)}{S.c(S.DIM)}every sweep is checked against these; "
          f"read at startup, so edit config.json and RESTART to change{S.RESET}")


def _show_skills() -> None:
    """The skills screen. Reached from /skill."""
    from superradiant_assistant import splash as S
    S.skills_screen()


def _show_memory() -> None:
    """The memory screen. Reached from the boot menu and from /memory alike."""
    from superradiant_assistant import splash as S
    S.memory_screen()


def _read_prompt(args, tracker) -> str:
    """The input area: a status strip, a rule, and the prompt.

    The status sits ABOVE the rule rather than below the typing line, which is
    where a mock-up would put it. Below means redrawing around a live `input()`,
    and a wrapped long line would then tear the box apart. Above, it is always
    correct and always visible while typing -- which is the point, because the
    two things it shows are whether shots reach hardware and what that is
    costing.
    """
    from superradiant_assistant import splash as S

    width = min(shutil.get_terminal_size((100, 30)).columns - 2, 118)
    live = S.c(S.GREEN) + "LIVE" if args.live else S.c(S.AMBER) + "offline"
    bits = [f"{live}{S.RESET}"]
    if args.creative:
        bits.append(f"{S.c(S.DIM)}creative{S.RESET}")
    bits.append(f"{S.c(S.DIM)}thinking {args.thinking}{S.RESET}")
    left = f"{S.c(S.DIM)} · {S.RESET}".join(bits)

    spent = tracker.total_tokens
    right = f"{S.c(S.DIM)}{args.model}"
    if spent:
        right += f"   {spent / 1000:.1f}k tokens"
    right += S.RESET

    pad = max(1, width - S._plain_len(left) - S._plain_len(right))
    print(f"\n  {left}{' ' * pad}{right}")
    print(f"  {S.c(S.DIM)}{'─' * width}{S.RESET}")
    return input(f"  {S.c(S.ACCENT)}› {S.RESET}").strip()


def _splash_info_lines(apparatus: str = ""):
    """What the boot screen shows about the session, as coloured rows.

    Deliberately nothing apparatus-specific. The limits, the memory sizes and the
    notebook count all belong to one lab, and the lab is chosen on the screen
    *after* this one -- so they were being read off the default apparatus and
    taken for the session's. Those facts are on the lab screen, per lab, and the
    limits are printed again under the loading bar once a lab is open.
    """
    return []


def _new_apparatus_flow():
    """Ask for what a new lab needs, write its config, return its name.

    Only what cannot be discovered is asked for. The globals and their safe
    ranges are the part that matters: they are what `run_sweep` refuses against,
    so a lab set up without them has no limits at all.
    """
    from superradiant_assistant import splash as S
    from superradiant_assistant.config import (apparatus_config_path,
                                               available_apparatus)
    import json

    print(S.framed("New lab", [
        "A lab is a set of globals, a data folder, and its own memory branch.",
        "",
        "It shares with every other lab:",
        "  · how labscript, BLACS and lyse behave (LABSCRIPT.md)",
        "  · the operator profile (OPERATOR.md)",
        "",
        "It gets its own:",
        "  · MEMORY.md, INSTRUCTIONS.md, notebook pages, history",
        "",
        "Leave a field empty to cancel.",
    ]))

    name = input("  name (one word, e.g. cesium): ").strip().lower()
    if not name or not name.replace("_", "").replace("-", "").isalnum():
        print("  cancelled")
        return None
    if name in available_apparatus():
        print(f"  '{name}' already exists — pick it on the Lab row instead")
        return None

    data = input("  data folder (where shot .h5 files are written): ").strip()
    if not data:
        print("  cancelled")
        return None

    print("  globals, one per line as  name  min  max  unit   (blank line ends)")
    print("  these are the limits every sweep is checked against")
    globals_ = []
    while True:
        line = input("    ").strip()
        if not line:
            break
        bits = line.split()
        if len(bits) < 3:
            print("      need at least: name min max")
            continue
        try:
            lo, hi = float(bits[1]), float(bits[2])
        except ValueError:
            print("      min and max must be numbers")
            continue
        g = {"name": bits[0], "min": lo, "max": hi}
        if len(bits) > 3:
            g["unit"] = bits[3]
        globals_.append(g)

    cfg = {
        "_comment": f"Apparatus '{name}'. Created by the new-lab flow.",
        "MEMORY_DATA_PATH": data,
        "globals": globals_,
        "sequences": [],
        "analysis_scripts": [],
    }
    path = apparatus_config_path(name)
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {path.name}")

    from superradiant_assistant import config as C
    C.select_apparatus(name)
    from superradiant_assistant.memory.store import MemoryStore
    store = MemoryStore()
    store.root.mkdir(parents=True, exist_ok=True)
    store.ensure_episode()
    print(f"  memory branch at {store.root}")
    print(f"  shared memory    {store.shared_root}")
    if not globals_:
        print("  NOTE: no globals declared, so no sweep can be range-checked. "
              f"Add them to {path.name} before running anything on hardware.")
    input("\n  enter to continue ")
    return name


def _run_splash(args):
    """Boot screen. Returns the chosen settings, or None if the operator quit."""
    from superradiant_assistant import config as C
    if args.no_splash:
        if args.apparatus:
            C.select_apparatus(args.apparatus)
        return {"live": args.live, "creative": args.creative,
                "thinking": args.thinking, "apparatus": C.APPARATUS}
    from superradiant_assistant import splash as S
    pref = args.apparatus or C.APPARATUS
    if args.apparatus:
        C.select_apparatus(args.apparatus)
    while True:
        chosen = S.show(
            {"live": args.live, "creative": args.creative,
             "thinking": args.thinking, "apparatus": pref},
            _splash_info_lines, model=args.model)
        if chosen is None:
            return None

        # Start opens the lab screen. Which lab is a bigger decision than any
        # toggle on the boot menu -- it picks the globals, the safe ranges, the
        # data root and the memory branch -- so it gets asked on its own screen,
        # and `q` there comes back here rather than starting anything.
        while True:
            picked = S.lab_screen(pref)
            if picked is None:
                break
            if picked == S.NEW_APPARATUS:
                made = _new_apparatus_flow()
                pref = made or pref
                continue
            C.select_apparatus(picked)
            pref = picked
            return {**chosen, "apparatus": picked}


def _flush_memory(team, args, tracker) -> None:
    """Distil this session into memory before the process exits.

    Compaction used to be the only thing that ever wrote MEMORY.md, and it only
    ran mid-session once a threshold was crossed -- which never happened. So
    everything a session learned died with it, every time.

    Only sessions that did something are distilled: an API call to summarise two
    lines of chat is wasted, and the note it writes dilutes the real ones.
    """
    if getattr(args, "no_memory_flush", False):
        return
    from superradiant_assistant.creative.evidence import get_evidence
    from superradiant_assistant.llm.client import make_client

    evidence = get_evidence()
    if not evidence.did_substantive_work:
        print("\n[memory] nothing substantive this session — nothing to remember")
        return

    print(f"\n[memory] distilling this session ({evidence.work_summary()}) ...")
    # The advisor is left out on purpose. Its session can hold the text of a
    # fetched web page, and the curator turns a session into MEMORY.md -- so
    # including it would let an arbitrary page write the store that every future
    # session is started with. Its conclusions reach memory the way any teammate's
    # do: through the lead's reply, and through `remember` if the lead judges the
    # fact durable.
    distil = [a for name, a in team.items() if name != "advisor"]
    try:
        wrote = MEMORY.compact_on_exit(
            distil,
            make_client(args.model, cost_tracker=tracker),
            evidence=evidence,
        )
    except Exception as e:
        print(f"[memory] could not write memory: {type(e).__name__}: {e}")
        return
    if wrote:
        print(f"[memory] updated {MEMORY.memory_file.name}, "
              f"{MEMORY.episode_path().name} in {MEMORY.root}")
    else:
        print("[memory] nothing was written")

    # Rewritten, not appended, and drawn from the whole day's log rather than
    # this session's turns -- so exiting six times in a day leaves the same page
    # as exiting once, and a two-minute last session does not become "the day".
    print("[memory] writing today's summary and next steps ...")
    try:
        if MEMORY.summarise_day(make_client(args.model, cost_tracker=tracker)):
            print(f"[memory] {MEMORY.episode_path().name} summary updated")
        else:
            print("[memory] no log to summarise today")
    except Exception as e:
        print(f"[memory] day summary failed: {type(e).__name__}: {e}")


def _write_weekly(args, tracker, command: str) -> None:
    """`/week [N]` — one page covering the last N days of notebook pages.

    Built from the daily pages, not from raw history: writing a summary every
    day is what makes the week cheap to assemble instead of a re-read of six
    hundred turns.
    """
    from superradiant_assistant.llm.client import make_client
    from superradiant_assistant import splash as S

    parts = command.split()
    try:
        n = int(parts[1]) if len(parts) > 1 else 7
    except ValueError:
        print("  usage: /week [number of days, default 7]")
        return

    days = MEMORY.episode_days()[-max(1, n):]
    if not days:
        print("  no notebook pages yet")
        return
    print(f"  summarising {len(days)} day(s): {days[0]} to {days[-1]} ...")
    out = MEMORY.weekly_summary(
        make_client(args.model, cost_tracker=tracker), days=days)
    if out is None:
        print("  nothing was written")
        return
    print(f"  wrote {out}")
    print("\n" + S.reply_panel(out.read_text(encoding="utf-8")))


def _start_up(args):
    """Build everything the session needs, reporting each step as it happens.

    Returns (team, registry, tracker), or (None, None, None) if a step that the
    session cannot do without failed. The steps are real work, not a decorated
    sleep: reaching the labscript suite spawns a subprocess and waits on a
    socket, which is where the wait actually is.
    """
    from superradiant_assistant import splash as S

    steps = ["skills", "configuration", "tool registry", "labscript suite",
             "memory", "agent team"]
    tracker = registry = team = None

    with S.Loader(steps, enabled=not args.no_splash) as ld:
        ld.step("skills")
        names = SKILLS.names()
        ld.ok(f"{len(names)} loaded")

        ld.step("configuration")
        from superradiant_assistant.safety import load_global_specs
        specs = load_global_specs()
        ld.ok(f"{len(specs)} globals, {len(CONFIG.sequences)} sequences")

        ld.step("tool registry")
        tracker = CostTracker(max_dollars=args.max_dollars,
                              max_tokens=args.max_tokens)
        registry = build_registry(gate=default_gate(auto_approve=args.yes),
                                  creative=args.creative)
        configure_session(goal_mode=args.goal_mode == "on", live=args.live,
                          creative=args.creative)
        ld.ok(f"{len(registry.names_for('lead'))} tools for the lead"
              + ("  (creative writes enabled)" if args.creative else ""))

        ld.step("labscript suite")
        if not args.live:
            ld.skip("offline — replaying history, nothing reaches hardware")
        else:
            from superradiant_assistant.labscript_api import (
                LabscriptAPI, LabscriptUnavailable,
            )
            from superradiant_assistant.orchestrator.live_executor import LiveExecutor
            try:
                api = LabscriptAPI(output_folder=str(CONFIG.historical_data_root))
                configure_session(
                    executor=LiveExecutor(CONFIG.historical_data_root,
                                          api.runmanager))
                ld.ok("runmanager reached")
            except LabscriptUnavailable as e:
                ld.fail("runmanager did not answer")
                print(f"\n[live] {e}")
                return None, None, None

        ld.step("memory")
        # Opening the agent starts today's notebook page, whether or not the
        # session goes on to do anything. A missing page and an uneventful page
        # mean different things to whoever reads back the week.
        page = MEMORY.ensure_episode()
        block = MEMORY.build_context_block()
        ld.ok(f"{len(block):,} characters carried in, notebook {page.name}"
              if block else f"empty — notebook {page.name} started")

        ld.step("agent team")
        team = build_team(registry, model=args.model, cost_tracker=tracker,
                          thinking=args.thinking)
        ld.ok(f"{len(team)} agents, thinking {args.thinking}")

    # Below the bar, once the lab is open and the values are this lab's. The
    # boot menu cannot show them: it runs before the lab is chosen.
    if not args.no_splash:
        _print_effective_limits()

    return team, registry, tracker


def main() -> None:
    ap = argparse.ArgumentParser(description="Labscript agent team")
    ap.add_argument("prompt", nargs="?", default=None,
                    help="One-shot request; omit for an interactive session")
    ap.add_argument("--goal-mode", choices=["on", "off"], default="on",
                    help="on (default): iterate until the threshold is met. "
                         "off: one shot per optimize request, then report")
    ap.add_argument("--live", action="store_true",
                    help="Queue shots on real hardware (requires the labscript GUIs)")
    ap.add_argument("--creative", action="store_true",
                    help="Let the agent write its own shots, analyses and parameters. "
                         "Every write needs your confirmation; --yes cannot skip it.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Any Gemini or Claude model, e.g. {DEFAULT_MODEL} or "
                         f"{CLAUDE_DEFAULT_MODEL}. A name containing 'claude' "
                         f"routes to Anthropic and needs ANTHROPIC_API_KEY; the "
                         f"tools, the skills and the memory are the same either "
                         f"way. Knowledge search embeds with Gemini regardless.")
    ap.add_argument("--thinking", default="low",
                    choices=["minimal", "low", "medium", "high", "default"],
                    help="How hard the model deliberates before each tool call. "
                         "'low' (default) suits routine work; the planner and "
                         "coder always run one level higher. 'default' hands the "
                         "choice to the model, which is what made a 20-call turn "
                         "take 462 s.")
    ap.add_argument("--max-dollars", type=float, default=100.0)
    ap.add_argument("--max-tokens", type=int, default=1_000_000)
    ap.add_argument("--yes", action="store_true",
                    help="Skip confirmation prompts. Only for offline/replay runs")
    ap.add_argument("--no-memory-flush", action="store_true",
                    help="Do not distil this session into MEMORY.md on exit. "
                         "The flush costs one extra API call, and only runs at "
                         "all if the session ran shots or wrote scripts.")
    ap.add_argument("--apparatus", default=None,
                    help="which lab to open: a name with a config.<name>.json "
                         "beside config.json. Decides the globals, the data "
                         "root and the memory branch.")
    ap.add_argument("--no-splash", action="store_true",
                    help="Skip the boot screen and take the flags as given.")
    args = ap.parse_args()

    if is_claude(args.model):
        if not CONFIG.anthropic_api_key:
            print(f"ERROR: --model {args.model} is an Anthropic model but there "
                  f"is no ANTHROPIC_API_KEY. Put it in .env (see .env.example).")
            return
        if not CONFIG.gemini_api_key:
            print("NOTE: no GEMINI_API_KEY — search_knowledge embeds with Gemini "
                  "and will fail; everything else works.")
    elif not CONFIG.gemini_api_key:
        print("ERROR: no GEMINI_API_KEY. Put it in .env (see .env.example).")
        return

    # The boot screen runs before anything is built, because live and creative
    # decide what gets built. Setting them here rather than only on the command
    # line is the point: sessions were repeatedly started without --live and
    # quietly replayed history instead of taking data.
    chosen = _run_splash(args)
    if chosen is None:
        print("\n[exiting]")
        return
    args.live = bool(chosen.get("live", args.live))
    args.creative = bool(chosen.get("creative", args.creative))
    args.thinking = str(chosen.get("thinking", args.thinking))
    # `_run_splash` has already applied the apparatus; carried here so the
    # session line and the memory flush name the right one.
    from superradiant_assistant import config as _C
    args.apparatus = _C.APPARATUS

    team, registry, tracker = _start_up(args)
    if team is None:
        return
    lead = team["lead"]

    # The boot screen already showed the art and the settings; this is the one
    # line the operator still needs once it is cleared away.
    from superradiant_assistant import splash as S
    mode = ("LIVE — shots queue on hardware" if args.live
            else "OFFLINE — replay only")
    print(f"{S.c(S.GREEN if args.live else S.AMBER)}  {mode}{S.RESET}"
          f"{S.c(S.ACCENT)}   lab {args.apparatus}{S.RESET}"
          f"{S.c(S.DIM)}   model {args.model}   thinking {args.thinking}"
          f"   goal {args.goal_mode}"
          f"{'   creative ON' if args.creative else ''}{S.RESET}")
    print(f"{S.c(S.DIM)}  /goal on|off   /model <name>   /team   /skill   "
          f"/advisor <question>   /memory   /week [n]   exit{S.RESET}")

    if args.prompt:
        from superradiant_assistant import splash as S
        from superradiant_assistant.orchestrator.agents import language_directive
        print("\n" + S.reply_panel(
            lead.send(args.prompt, language_directive(args.prompt))))
        _flush_memory(team, args, tracker)
        print("\nUsage:", tracker.summary())
        return

    while True:
        try:
            user_input = _read_prompt(args, tracker)
        except (EOFError, KeyboardInterrupt):
            print("\n[exiting]")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break

        if user_input.startswith("/goal"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1] in ("on", "off"):
                configure_session(goal_mode=parts[1] == "on")
                print(f"  goal mode {parts[1]}")
            else:
                print("  usage: /goal on | /goal off")
            continue
        if user_input.startswith("/model"):
            parts = user_input.split()
            if len(parts) != 2:
                print(f"  usage: /model <name>   (now: {args.model})")
                print(f"  e.g. /model {CLAUDE_DEFAULT_MODEL} | /model {DEFAULT_MODEL}")
                continue
            from superradiant_assistant.splash import model_problem
            want = parts[1]
            problem = model_problem(want)
            if problem:
                # Without this the switch succeeds and every message after it
                # 404s, with the context still sitting in the dead session.
                print(f"  {problem}. Staying on {args.model}.")
                continue
            try:
                # The whole team moves, not just the lead: a coder left on the
                # other provider would answer the lead out of a context the
                # lead no longer has. /team is where one agent is moved alone.
                #
                # Except the advisor. It is pinned to the strongest model because
                # that is the entire point of the role, and dragging it down with
                # a routine `/model` switch would quietly make its diagnoses worse
                # -- the one failure that does not announce itself.
                for name, member in team.items():
                    if name == "advisor":
                        continue
                    print("  " + member.switch_model(want))
                adv = team.get("advisor")
                if adv is not None and adv.model != want:
                    print(f"  advisor stays on {adv.model} — /team to move it")
                args.model = want
            except Exception as e:
                print(f"  [error] {type(e).__name__}: {e}")
            continue
        if user_input.startswith("/advisor"):
            question = user_input[len("/advisor"):].strip()
            if not question:
                print("  usage: /advisor <what you want explained>")
                print("  e.g. /advisor the transmission peak fell from 0.81 V on "
                      "8/12 to 0.43 V today and the optics have not changed")
                print(f"  ({team['advisor'].model}, thinking "
                      f"{team['advisor'].thinking_level}; it reads shots, the "
                      f"notebook, past reports, the knowledge base and the web)")
                continue
            try:
                from superradiant_assistant import splash as S
                from superradiant_assistant.orchestrator import mailbox
                from superradiant_assistant.orchestrator.agents import (
                    language_directive,
                )
                adv = team["advisor"]
                reply = adv.send(question,
                                 language_directive(question)
                                 + mailbox.directive("advisor"))
                # Its own frame, blue, with its face in the header. Only on this
                # path: when the LEAD consults it through `ask_advisor` the reply
                # is an input to the lead's answer, not an answer to the operator,
                # and framing it as one would put two verdicts on screen.
                print("\n" + S.advisor_panel(reply, model=adv.model,
                                              thinking=adv.thinking_level))
                # Not auto-posted. Whether the lead needs to hear this is the
                # advisor's judgment call, made with `send_note` like any other
                # tool -- see its role prompt. `/advisor` still bypasses the lead
                # for the conversation itself; only what the advisor chooses to
                # write down reaches it.
                #
                # `send_note` fired mid-turn, before this panel existed. The
                # envelope is queued, not printed, until now -- so it always
                # lands below the blue frame, never above it.
                from superradiant_assistant.tools import team_tools
                team_tools.flush_mail_animations()
            except Exception as e:
                print(f"\n[error] {type(e).__name__}: {e}")
            continue
        if user_input == "/team":
            _show_team(team, registry)
            continue
        if user_input in ("/skill", "/skills"):
            _show_skills()
            continue
        if user_input == "/memory":
            _show_memory()
            continue
        if user_input.startswith("/week"):
            _write_weekly(args, tracker, user_input)
            continue

        try:
            from superradiant_assistant import splash as S
            from superradiant_assistant.orchestrator import mailbox
            from superradiant_assistant.orchestrator.agents import language_directive
            # A model does not call `read_notes` on the off chance something
            # arrived, so the inbox has to announce itself. Per-turn, like the
            # language directive: shown to the model, not recorded as part of
            # what the operator asked for.
            print("\n" + S.reply_panel(
                lead.send(user_input, language_directive(user_input)
                          + mailbox.directive("lead"))))
            from superradiant_assistant.tools import team_tools
            team_tools.flush_mail_animations()
        except Exception as e:
            print(f"\n[error] {type(e).__name__}: {e}")

    _flush_memory(team, args, tracker)
    print("\nUsage:", tracker.summary())


if __name__ == "__main__":
    main()
