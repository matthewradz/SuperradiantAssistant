"""Hook manager plus the tool-call gate.

Two layers live here:
- HookManager: the original named-callback registry used by run_loop
  (before_loop / before_shot / after_shot / ...).
- The tool-call gate: before_tool_call / after_tool_call, which every tool
  dispatch passes through. Confirmation prompts and the audit log are enforced
  here rather than inside tool bodies, so a model cannot talk its way past them
  — the rules are never shown to the model.
"""
from __future__ import annotations
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Any, Optional


class HookManager:
    def __init__(self) -> None:
        self._hooks: Dict[str, List[Callable[..., Any]]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self._hooks.setdefault(name, []).append(fn)

    def run(self, name: str, **kwargs: Any) -> List[Any]:
        return [fn(**kwargs) for fn in self._hooks.get(name, [])]

    def any_true(self, name: str, **kwargs: Any) -> bool:
        return any(bool(r) for r in self.run(name, **kwargs))

    def all_true(self, name: str, **kwargs: Any) -> bool:
        results = self.run(name, **kwargs)
        if not results:
            return True   # no hooks registered → don't block
        return all(bool(r) for r in results)


@dataclass
class HookDecision:
    action: str                      # "allow" | "deny"
    reason: str = ""
    updated_input: Optional[Dict[str, Any]] = None

    @property
    def denied(self) -> bool:
        return self.action == "deny"


#: What a cancelled call hands back to the model.
#:
#: "cancelled by operator" on its own is an invitation to improvise, and it was
#: taken: a declined `write_analysis` was followed by the agent picking a
#: hand-written routine it had not been asked about, swapping lyse's loaded
#: routine list to it, and marking the plan step done. The operator said "do not
#: write this", not "find something else and reconfigure my analysis".
CANCELLED = (
    "cancelled by the operator. This is a decision, not an obstacle.\n"
    "- Do NOT retry this call, under this or any other name.\n"
    "- Do NOT substitute a different existing file, and do NOT change any "
    "configuration (lyse routines, globals, the loaded sequence) as a way "
    "around it.\n"
    "- Do NOT mark the step done or report the work as complete.\n"
    "Finish only the steps that do not depend on this one. Then STOP, say in "
    "one or two sentences what you were about to do and why, and ask the "
    "operator what they want instead. Asking is the correct end of this turn."
)

#: Same reasoning for the plain y/N path.
DECLINED = (
    "declined by the operator. Do not retry it, do not route around it, and do "
    "not mark anything done. Say what you were about to do and ask what they "
    "would prefer."
)


@dataclass
class ChoiceOption:
    """One answer the operator can type at a confirmation prompt.

    `apply` rewrites the tool's arguments (returning a new dict); an option with
    `apply=None` refuses the call instead, handing `deny_reason` back to the
    model so it knows what to do next rather than just seeing "declined".
    """
    key: str
    label: str
    detail: str = ""
    apply: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    deny_reason: str = ""


@dataclass
class ChoicePrompt:
    """A multi-way confirmation, in place of the plain y/N.

    Used where "yes" is genuinely ambiguous: the model asking to write a script
    might mean add one, replace one, or should not be writing at all, and only
    the operator can say which.
    """
    question: str
    options: List[ChoiceOption]
    context: str = ""                # shown above the options


@dataclass
class ToolGate:
    """Confirmation + audit gate wrapped around every tool dispatch."""
    audit_path: Path
    session_id: str
    require_confirm: frozenset = field(default_factory=frozenset)
    always_confirm: frozenset = field(default_factory=frozenset)
    audit: frozenset = field(default_factory=frozenset)
    auto_approve: bool = False       # tests / unattended runs

    def before_tool_call(self, name: str, args: Dict[str, Any],
                          preview: str = "",
                          choices: Optional[ChoicePrompt] = None) -> HookDecision:
        # `always_confirm` has to mean always, including under --yes. Folding both
        # tiers into one auto_approve check made the two names identical in
        # practice, so an unattended run could write a parameter or a generated
        # shot with nobody reading it.
        if name in self.always_confirm:
            needs_confirm = True
        elif name in self.require_confirm:
            needs_confirm = not self.auto_approve
        else:
            needs_confirm = False
        if not needs_confirm:
            return HookDecision(action="allow")

        # Progress ticks from other threads would otherwise scroll the question
        # off the screen while it is waiting to be answered.
        try:
            from superradiant_assistant.orchestrator.agents import suppress_heartbeat
        except Exception:
            from contextlib import nullcontext as suppress_heartbeat  # type: ignore

        from superradiant_assistant import splash as S

        with suppress_heartbeat():
            body = (preview if preview
                    else "\n".join(f"{k} = {v}" for k, v in args.items()))
            print()
            print(S.framed(f"confirm · {name}", body, colour=S.AMBER))

            if not sys.stdin.isatty():
                # Fail closed: an unattended run must not silently touch hardware.
                return HookDecision(
                    action="deny",
                    reason=f"{name} requires confirmation but stdin is not interactive",
                )

            if choices is not None:
                return self._ask_choice(args, choices)

            try:
                ans = input(f"  {S.c(S.AMBER)}Proceed?{S.RESET} "
                            f"{S.c(S.DIM)}[y/N]{S.RESET} ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = ""
        if ans not in ("y", "yes"):
            return HookDecision(action="deny", reason=DECLINED)
        return HookDecision(action="allow")

    def _ask_choice(self, args: Dict[str, Any],
                    prompt: ChoicePrompt) -> HookDecision:
        """A multi-way question, answered by moving a cursor rather than typing.

        Typing a digit meant reading four lines, choosing, and hoping the key
        pressed was the one intended -- and a bare 'n' at a prompt that has just
        described an overwrite reads as "no" to anyone who has ever seen [y/N].
        Moving a highlighted line and pressing enter cannot be misread.
        """
        from superradiant_assistant import splash as S

        if prompt.context:
            print()
            print(S.framed("what already exists", prompt.context, colour=S.DIM))

        options = list(prompt.options)
        options.append(ChoiceOption(
            key="c", label="cancel", detail="do nothing; ask the operator",
            apply=None, deny_reason=CANCELLED))

        if not sys.stdout.isatty():
            return HookDecision(action="deny", reason=CANCELLED)

        import shutil

        cursor, drawn = 0, 0
        print()
        while True:
            # Every row is cut to the terminal width. CSI-F moves by PHYSICAL
            # lines, so a single row long enough to wrap makes the redraw move
            # up fewer lines than it drew -- the block walks down the screen and
            # leaves the tail of the wrapped line behind it.
            width = max(30, shutil.get_terminal_size((100, 30)).columns - 1)
            out = [f"  {S.bold(S.c(S.TEXT) + prompt.question)}{S.RESET}"]
            for i, opt in enumerate(options):
                if i == cursor:
                    row = (f"  {S.c(S.ACCENT)}{S.CURSOR} "
                           f"{S.bold(opt.label):<20}{S.RESET}"
                           f"{S.c(S.TEXT)}{opt.detail}{S.RESET}")
                else:
                    row = (f"    {S.c(S.DIM)}{opt.label:<20}{opt.detail}"
                           f"{S.RESET}")
                out.append(S._truncate(row, width))
            out.append(f"  {S.c(S.DIM)}↑↓ choose   "
                       f"{S.c(S.TEXT)}enter{S.c(S.DIM)} confirm{S.RESET}")

            if drawn:
                sys.stdout.write(f"\x1b[{drawn}F")
            sys.stdout.write("".join("\x1b[2K" + l + "\n" for l in out))
            sys.stdout.flush()
            drawn = len(out)

            try:
                key = S.read_key()
            except (EOFError, KeyboardInterrupt):
                return HookDecision(action="deny", reason=CANCELLED)

            if key in ("down", "j", "\t", "right"):
                cursor = (cursor + 1) % len(options)
            elif key in ("up", "k", "left"):
                cursor = (cursor - 1) % len(options)
            elif key in ("q", "esc", "n", ""):
                # 'n' still means no. The operator has just read a paragraph
                # about an overwrite; whatever the UI is, that reflex should
                # not commit anything.
                cursor = len(options) - 1
                key = "enter"
            elif key.isdigit() and 1 <= int(key) <= len(prompt.options):
                cursor = int(key) - 1
                key = "enter"

            if key == "enter":
                chosen = options[cursor]
                print(f"  {S.c(S.ACCENT)}-> {chosen.label}{S.RESET}")
                if chosen.apply is None:
                    return HookDecision(
                        action="deny",
                        reason=chosen.deny_reason or chosen.label)
                return HookDecision(action="allow",
                                    updated_input=chosen.apply(dict(args)))

    def after_tool_call(self, name: str, args: Dict[str, Any],
                         result: Any, error: Optional[str] = None) -> None:
        if name not in self.audit:
            return
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "session_id": self.session_id,
            "tool": name,
            "args": _jsonable(args),
            "result": _jsonable(result),
            "error": error,
        }
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            print(f"  [audit] could not write audit record: {e}")


def _jsonable(v: Any) -> Any:
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)


def default_gate(auto_approve: bool = False, session_id: Optional[str] = None) -> ToolGate:
    """Build the gate with the risk tiers from docs/tool-schema.md §3."""
    from superradiant_assistant.config import CONFIG

    # Creative-mode writes are `always_confirm`, never `require_confirm`: the
    # operator has to read generated code before it runs, and --yes must not be
    # able to wave that through the way it can for a routine sweep.
    creative_writes = frozenset({
        "propose_global", "write_shot", "write_analysis",
        "save_experiment_skill", "write_report",
    })

    audit_dir = Path(CONFIG.historical_data_root).parent / "agent_logs"
    return ToolGate(
        audit_path=audit_dir / "tool_audit.jsonl",
        session_id=session_id or datetime.now().strftime("%Y%m%d-%H%M%S"),
        require_confirm=frozenset({"run_optimization", "run_sweep"}),
        # `remember` writes a file that is injected into every future system
        # prompt, so the operator reads the wording before it becomes standing
        # instruction. Cheap to confirm, expensive to get subtly wrong.
        always_confirm=(frozenset({"set_runmanager_global", "remember"})
                        | creative_writes),
        audit=frozenset({
            "analyze_results", "get_runmanager_globals", "run_optimization",
            "run_sweep", "set_runmanager_global", "remember",
        }) | creative_writes,
        auto_approve=auto_approve or os.environ.get("AGENT_AUTO_APPROVE") == "1",
    )

