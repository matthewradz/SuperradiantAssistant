"""Static checks on machine-generated shot and analysis scripts.

Creative mode deliberately removes the property that the model cannot write or
execute code, so something has to take its place. This module is that
replacement: generated code is parsed and rejected before it is ever written to
disk if it reaches outside the sandbox the operator agreed to.

It is a whitelist, not a blacklist — an unknown import or an unrecognised device
call is refused rather than allowed. Blacklists lose this game.

Note what this does NOT do: it is a static check, not a sandbox. It stops the
obvious escapes and every accidental one, but the operator confirmation step is
still the real boundary. Never present it as making review unnecessary.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass, field
from typing import List, Set

# Modules a generated script may import.
SHOT_IMPORTS: Set[str] = {
    "numpy", "np", "math",
    "labscript",
    "labscriptlib.Cesium.connection_table",
}

ANALYSIS_IMPORTS: Set[str] = {
    "numpy", "np", "math", "h5py",
    "matplotlib", "matplotlib.pyplot", "plt",
    "scipy", "scipy.signal", "scipy.optimize",
    "lyse",
    "analysislib.Cesium.analysis_utils.data_classes",
}

# Never allowed anywhere: these are the actual escape hatches.
FORBIDDEN_NAMES: Set[str] = {
    "eval", "exec", "compile", "__import__", "open", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "breakpoint", "memoryview",
}

FORBIDDEN_MODULES: Set[str] = {
    "os", "sys", "subprocess", "shutil", "socket", "requests", "urllib",
    "importlib", "pickle", "ctypes", "pathlib", "tempfile", "glob",
    "multiprocessing", "threading", "pty", "signal", "builtins",
}

# Device objects a shot may name. These come from ConnectionTable(), not from an
# import, so a plain undefined-name check would flag them.
DEVICE_NAMES: Set[str] = {"RigolDG1022", "scope1", "dummy_clock"}

# Shot-time device methods a generated sequence may call.
#
# An earlier design routed these through a wrapper module to range-check them.
# That wrapper had to live inside labscriptlib, where importing it turned the
# package into a namespace-package import that runmanager's compiler could not
# survive, and it made generated shots structurally different from the ones that
# already worked here. Values are checked before they reach runmanager instead.
DEVICE_METHOD_ALLOWLIST: Set[str] = {
    "program_sine",
    "program_harmonics",
    "program_arbitrary_waveform",
    "program_arbitrary_waveform_settings",
    "acquire",
}


@dataclass
class GuardReport:
    ok: bool
    violations: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    device_calls: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok:
            return "code guard: passed"
        lines = ["code guard: REFUSED"]
        lines += [f"  - {v}" for v in self.violations]
        return "\n".join(lines)


def _root_module(name: str) -> str:
    return (name or "").split(".")[0]


# A rejection has to say what the right answer is. Telling the model only that
# an import is "not on the whitelist" taught it to delete the import instead of
# fixing the path — which passed the guard and produced a NameError at runtime.
_SUGGESTIONS = {
    "safe_devices": "Use `from labscriptlib.Cesium import safe_devices`.",
    "connection_table": ("Use `from labscriptlib.Cesium.connection_table import "
                          "ConnectionTable`."),
    "data_classes": ("Use `from analysislib.Cesium.analysis_utils.data_classes "
                      "import Shot`."),
}


def _suggest(target: str, kind: str) -> str:
    leaf = (target or "").split(".")[-1]
    if leaf in _SUGGESTIONS:
        return _SUGGESTIONS[leaf]
    allowed = SHOT_IMPORTS if kind == "shot" else ANALYSIS_IMPORTS
    return f"Allowed imports for a {kind}: {', '.join(sorted(allowed))}."


def _structural_problems(tree: ast.AST, kind: str) -> List[str]:
    """Faults that are not about safety but guarantee the script cannot work.

    Catching these here keeps the operator from being asked to approve code that
    was never going to run, and stops the model burning turns on a script whose
    only feedback would have been a runtime traceback.
    """
    problems: List[str] = []

    imported: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                imported.add(a.asname or a.name)

    used: Set[str] = {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    assigned: Set[str] = {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }

    if "safe_devices" in used and "safe_devices" not in imported:
        problems.append(
            "uses `safe_devices` but never imports it — add "
            "`from labscriptlib.Cesium import safe_devices`")

    if kind == "multishot":
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        calls = {n.func.id for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        attr_calls = {n.func.attr for n in ast.walk(tree)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}

        if "data" not in calls:
            problems.append(
                "a multishot routine works from the whole sequence: call `data()` "
                "to get the dataframe of every shot's results")
        if "path" in names or "Run" in names:
            problems.append(
                "`path` and `Run` are single-shot concepts — a multishot routine "
                "runs once over all shots and has no single shot to open")
        if "show" in attr_calls:
            problems.append(
                "remove plt.show() — lyse displays the figure itself, and calling "
                "show() from a routine blocks its analysis thread")

    if kind in ("analysis", "multishot"):
        src_calls = [n for n in ast.walk(tree)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]

        # `lyse.routine_storage` is scratch space shared between shots; it has no
        # save_result. A routine using it runs, saves nothing, and reports no
        # error until the results are missing much later.
        for call in src_calls:
            owner = call.func.value
            attr = call.func.attr
            owner_name = (owner.attr if isinstance(owner, ast.Attribute)
                          else owner.id if isinstance(owner, ast.Name) else "")
            if owner_name == "routine_storage" and attr.startswith("save_result"):
                problems.append(
                    "`routine_storage.save_result(...)` does not exist — "
                    "routine_storage is scratch space that persists between shots. "
                    "Use `run = Run(path)` then `run.save_result(name, value)`")

        # Only a per-shot routine has to produce a metric. A multishot routine
        # legitimately just draws the curve across a sweep.
        if kind == "analysis":
            saves = [c for c in src_calls
                     if c.func.attr in ("save_result", "save_results")]
            if not saves:
                problems.append(
                    "a single-shot routine must call run.save_result(name, value) at "
                    "least once — a plot alone produces no metric to read back")

        # lyse exec's the file directly, so anything guarded by __main__ never runs.
        for node in ast.walk(tree):
            if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                    and isinstance(node.test.left, ast.Name)
                    and node.test.left.id == "__name__"):
                problems.append(
                    "remove the `if __name__ == '__main__':` block — lyse exec's the "
                    "routine directly, so that code never runs")

    if kind == "shot":
        calls = {n.func.id for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        if "ConnectionTable" not in calls:
            problems.append(
                "a shot must call ConnectionTable() before touching any device, "
                "otherwise names like RigolDG1022 do not exist")
        if "start" not in calls or "stop" not in calls:
            problems.append("a shot must call start() and stop(t) from labscript")
        # Device names come from ConnectionTable(), not from an import, so they
        # look undefined to a normal linter — check them explicitly.
        for dev in sorted(DEVICE_NAMES & used):
            if "ConnectionTable" not in calls and dev not in assigned:
                problems.append(f"'{dev}' is used but ConnectionTable() was never called")

    return problems


def check(source: str, kind: str = "shot") -> GuardReport:
    """Parse `source` and report why it may not be written, if so.

    `kind` is "shot" or "analysis"; they get different import whitelists
    because an analysis script legitimately needs h5py and matplotlib while a
    shot does not.
    """
    allowed = SHOT_IMPORTS if kind == "shot" else ANALYSIS_IMPORTS
    report = GuardReport(ok=True)

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return GuardReport(ok=False, violations=[f"syntax error on line {e.lineno}: {e.msg}"])

    for node in ast.walk(tree):
        # --- imports ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                report.imports.append(alias.name)
                root = _root_module(alias.name)
                if root in FORBIDDEN_MODULES:
                    report.violations.append(
                        f"line {node.lineno}: import of '{alias.name}' is never allowed")
                elif alias.name not in allowed and root not in allowed:
                    report.violations.append(
                        f"line {node.lineno}: import of '{alias.name}' is not on the "
                        f"{kind} whitelist. {_suggest(alias.name, kind)}")

        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            report.imports.append(mod)
            root = _root_module(mod)
            # `from pkg.sub import thing` must be judged on `pkg.sub.thing`, not
            # `pkg.sub` alone — otherwise whitelisting one submodule either
            # rejects the legitimate form or opens the whole package.
            targets = [f"{mod}.{a.name}" if mod else a.name for a in node.names]
            if root in FORBIDDEN_MODULES:
                report.violations.append(
                    f"line {node.lineno}: import from '{mod}' is never allowed")
            elif mod in allowed or root in allowed:
                pass
            else:
                for a, target in zip(node.names, targets):
                    if target not in allowed:
                        report.violations.append(
                            f"line {node.lineno}: import of '{target}' is not on the "
                            f"{kind} whitelist. {_suggest(target, kind)}")

        # --- dangerous builtins ---
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            report.violations.append(
                f"line {node.lineno}: use of '{node.id}' is not allowed")

        # --- dunder access: the usual way out of a restricted namespace ---
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                report.violations.append(
                    f"line {node.lineno}: attribute '{node.attr}' is not allowed")

        # --- direct device method calls ---
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if isinstance(owner, ast.Name) and owner.id in DEVICE_NAMES:
                call = f"{owner.id}.{node.func.attr}"
                report.device_calls.append(call)
                if node.func.attr not in DEVICE_METHOD_ALLOWLIST:
                    report.violations.append(
                        f"line {node.lineno}: '{call}(...)' calls the device directly "
                        f"and bypasses the range checks. Use "
                        f"safe_devices.{node.func.attr}({owner.id}, ...) instead."
                    )

    report.violations.extend(
        f"structural: {p}" for p in _structural_problems(tree, kind)
    )
    report.ok = not report.violations
    return report


def check_shot(source: str) -> GuardReport:
    return check(source, kind="shot")


def check_analysis(source: str, multishot: bool = False) -> GuardReport:
    return check(source, kind="multishot" if multishot else "analysis")
