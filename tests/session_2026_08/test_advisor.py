"""The advisor: a physicist that reads everything and can change nothing.

Three things here are load-bearing and none of them is obvious from reading the
code:

  * every tool it can reach is read-only, and the whitelist -- not the schema --
    is what stops it reaching the others;
  * `fetch_web_page` has no confirmation prompt in front of it, so its URL guard
    is the entire boundary, including across a redirect;
  * two of the things the operator asked for live only in prompt text (reach for
    the RAG on your own initiative; judge results against the goal), so a future
    edit that drops them would degrade the advice silently rather than break.
"""
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["AGENT_APPARATUS"] = "cesium"

from superradiant_assistant.hooks import default_gate
from superradiant_assistant.llm.cost_tracker import CostTracker
from superradiant_assistant.memory import MEMORY
from superradiant_assistant.memory.store import CONTEXT_LAYERS
from superradiant_assistant.orchestrator.agents import (
    ADVISOR_MODEL, advisor_model, build_team, _MEMORY_LAYERS,
)
from superradiant_assistant.tools import build_registry
from superradiant_assistant.tools import advisor_tools as A
from superradiant_assistant.tools.advisor_tools import build_advisor_tool_specs

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


reg = build_registry(gate=default_gate(), creative=True)
team = build_team(reg, model="gemini-2.5-flash", cost_tracker=CostTracker())

print("\n=== 1. the fourth team member ===")
check(set(team) == {"lead", "planner", "coder", "advisor"},
      f"the team is four roles ({sorted(team)})")
adv = team["advisor"]
check(adv.model == ADVISOR_MODEL,
      f"the advisor is pinned to {ADVISOR_MODEL}, not the team's model "
      f"({adv.model})")
check(adv.thinking_level == "high",
      f"and thinks hardest ({adv.thinking_level})")

# The pin has to yield on a machine that cannot reach that provider, or the first
# summon raises from inside the session with the question already asked.
key = os.environ.pop("ANTHROPIC_API_KEY", None)
try:
    check(advisor_model("gemini-2.5-flash") == "gemini-2.5-flash",
          "with no ANTHROPIC_API_KEY it falls back to the team's model")
finally:
    if key is not None:
        os.environ["ANTHROPIC_API_KEY"] = key
check(advisor_model("gemini-2.5-flash") == ADVISOR_MODEL
      or not os.environ.get("ANTHROPIC_API_KEY"),
      "and takes the pinned model back when the key is there")

print("\n=== 2. it can read everything and write nothing ===")
tools = set(reg.names_for("advisor"))
for name in ("read_shot_results", "read_lab_file", "list_scripts",
             "list_lab_files", "analyze_results", "get_runmanager_globals",
             "get_lyse_routines", "load_skill"):
    check(name in tools, f"reads the bench: {name}")
# Deliberately absent. It inspected two shots on 2026-08-18 for numbers that were
# already named metrics: 2,092 chars and a round trip each, against 429 chars and
# one call for both shots. What it uniquely offers is layout, and the advisor
# writes no code.
check("inspect_shot" not in tools,
      "no inspect_shot: read_shot_results returns the same numbers by name, "
      "for many shots, in one call")
check("inspect_shot" in set(reg.names_for("coder")),
      "the coder keeps it -- writing an analysis needs the trace names and columns")
check("search_lab_knowledge" in tools,
      "has the RAG -- the knowledge base is half of why this role exists")
for name in ("list_lab_history", "read_notebook", "read_report", "search_lab_notes"):
    check(name in tools, f"reads what we wrote before: {name}")
for name in ("search_web", "fetch_web_page"):
    check(name in tools, f"reads outside: {name}")

# Everything that changes the apparatus, the code, the lyse config, the plan or
# memory. `remember` included: the advisor proposes durable facts in its reply and
# the lead records them, so a hypothesis cannot be filed as an established one.
forbidden = ["set_runmanager_global", "load_sequence", "engage_shot", "run_sweep",
             "run_optimization", "set_lyse_routines", "remember", "set_plan",
             "update_plan", "propose_global", "write_shot", "write_analysis",
             "write_report", "save_experiment_skill"]
leaked = [n for n in forbidden if n in tools]
check(not leaked, f"cannot reach any writer or hardware tool (leaked: {leaked})")
check(len(tools) == 17, f"17 tools in total, incl. the mailbox ({len(tools)})")

print("\n=== 3. the whitelist refuses, not merely the schema ===")
# A tool absent from the declarations is a hint; this is the boundary. It has to
# hold even if something in the code calls dispatch directly.
for name, args in (("set_runmanager_global",
                    {"name": "waveplate_angle", "value": 42, "reason": "x"}),
                   ("engage_shot", {"reason": "x"}),
                   ("remember", {"fact": "x", "kind": "apparatus"})):
    out = reg.dispatch("advisor", name, args)
    check(out.startswith("error: agent 'advisor' is not permitted"),
          f"dispatch(advisor, {name!r}) is refused: {out[:60]}")

print("\n=== 4. the advisor's own tools are its own ===")
for role in ("lead", "planner", "coder"):
    others = set(reg.names_for(role))
    bled = [n for n in ("list_lab_history", "read_notebook", "read_report",
                        "search_lab_notes", "search_web", "fetch_web_page")
            if n in others]
    check(not bled, f"{role} did not inherit the advisor's tools ({bled})")

print("\n=== 5. the memory it carries ===")
layers = _MEMORY_LAYERS["advisor"]
sysmsg = adv.build_system_instruction()
check("objective" in layers, "carries the objective: it judges results against the goal")
check(layers[0] == "objective", "and the goal leads the block, before any number")
for want in ("labscript", "memory", "instructions", "notebook"):
    check(want in layers, f"carries {want}")
check("operator" not in layers,
      "does not carry the operator profile -- the one agent whose job is not to "
      "be agreeable")
check("Standing instructions" in sysmsg,
      "still carries the operator's standing instructions")
check("Operator profile" not in sysmsg, "and nothing from the profile leaked in")
check("objective" in CONTEXT_LAYERS, "'objective' is a real layer name")
check(MEMORY.build_context_block() == MEMORY.build_context_block(CONTEXT_LAYERS),
      "the default block still equals every layer, with the new one in it")
check(sum(1 for r in ("lead", "planner", "coder")
          if "objective" in (_MEMORY_LAYERS.get(r) or CONTEXT_LAYERS)) == 1,
      "the lead gets it by falling through to all layers; planner and coder do not")

print("\n=== 6. nothing can write the goal ===")
# The handler directly, not through dispatch: the gate would decline the write for
# want of a confirmation and the kind would never be validated at all.
from superradiant_assistant.tools.lab_tools import remember as _remember
out = _remember(fact="the goal is x", kind="objective")
check("kind must be" in out,
      f"remember(kind='objective') is rejected: {out[:70]}")
check(not hasattr(MEMORY, "write_objective"),
      "the store has no writer for OBJECTIVE.md at all")
check(MEMORY.objective_file.name == "OBJECTIVE.md"
      and MEMORY.objective_file.parent == MEMORY.root,
      f"it lives beside MEMORY.md ({MEMORY.objective_file})")

print("\n=== 7. the prompt says the two things only the prompt can say ===")
low = sysmsg.lower()
check("search_lab_knowledge" in sysmsg,
      "names the RAG tool, so it reaches for it instead of waiting to be handed "
      "evidence")
check("before reasoning from general knowledge" in low,
      "and says to search BEFORE reasoning from general knowledge")
check("inferring" in low and "guessed objective" in low,
      "says to label advice built on a guessed objective")
check("flatter" in low or "agreeable" in low,
      "tells it not to flatter anyone")
check("untrusted" in low or "data, not instruction" in low,
      "tells it a fetched page is data, not instruction")
check("instrument before the physics" in low,
      "tells it to suspect the instrument first -- the filter-sweep outliers were "
      "a scope window, not physics")
# 2026-08-18: summoned directly, it diagnosed a real result, put the whole
# finding in `send_note` to the lead, and replied to the OPERATOR with nothing
# but "Note delivered to the lead." -- the panel the operator was reading came
# back empty because the model treated the note as a substitute for answering,
# not an addition to it.
check("in addition to answering" in low or "instead of" in low,
      "says a note to the lead does not replace answering whoever asked")
check("note delivered" in low,
      "names the exact failure so it isn't repeated")

print("\n=== 8. the URL guard IS the boundary (no confirmation prompt) ===")
for url in ("http://localhost:8000/x", "http://127.0.0.1/", "http://[::1]/",
            "http://169.254.169.254/latest/meta-data/", "http://10.0.0.5/",
            "http://192.168.1.1/", "https://192.168.0.1/admin"):
    check(bool(A._url_problem(url)), f"refuses {url}")
for url in ("file:///C:/Users", "ftp://example.com/x", "not a url",
            "javascript:alert(1)"):
    check(bool(A._url_problem(url)), f"refuses {url}")
check(A._url_problem("https://arxiv.org/abs/2401.00001") == "",
      "allows a public https URL")
check(A._url_problem("http://nonexistent.invalid/") != "",
      "a host that does not resolve is refused, not attempted")


class _FakeResponse:
    def __init__(self, status, headers=None, text=""):
        self.status_code, self.headers, self.text = status, headers or {}, text


def _redirect_chain(monkey_urls):
    """Serve a canned redirect chain and record which URLs were requested."""
    asked = []

    def fake_get(url, **kw):
        asked.append(url)
        return monkey_urls[url]

    return fake_get, asked


print("\n=== 9. a redirect cannot walk around the guard ===")
# requests follows redirects itself, which would make the check above a
# formality: one 302 to 127.0.0.1 and the page is on this machine. So each hop is
# validated before it is taken.
import requests as _requests
_real_get = _requests.get
try:
    fake, asked = _redirect_chain({
        "https://example.com/start": _FakeResponse(
            302, {"location": "http://127.0.0.1:8000/secret"}),
    })
    _requests.get = fake
    out = A.fetch_web_page("https://example.com/start")
    check("refused" in out and "127.0.0.1" in out,
          f"a 302 to a private address is refused at the hop: {out[:80]}")
    check(asked == ["https://example.com/start"],
          f"and the private address was never requested ({asked})")

    hops = {f"https://example.com/{i}": _FakeResponse(
        302, {"location": f"https://example.com/{i + 1}"}) for i in range(9)}
    fake, asked = _redirect_chain(hops)
    _requests.get = fake
    out = A.fetch_web_page("https://example.com/0")
    check("redirect" in out.lower(),
          f"an endless redirect chain stops rather than looping: {out[:70]}")
    check(len(asked) <= A._MAX_HOPS + 1,
          f"at most {A._MAX_HOPS + 1} requests were made ({len(asked)})")

    fake, asked = _redirect_chain({
        "https://example.com/big": _FakeResponse(
            200, {"content-type": "text/html"},
            "<html><body><p>" + "x" * 50_000 + "</p></body></html>"),
    })
    _requests.get = fake
    out = A.fetch_web_page("https://example.com/big")
    check(len(out) < 30_000, f"an enormous page is truncated ({len(out)} chars)")
    check("truncated from" in out, "and says it was truncated")
    check("UNTRUSTED" in out,
          "and the body is framed as untrusted data, not instruction")

    fake, asked = _redirect_chain({
        "https://example.com/blob": _FakeResponse(
            200, {"content-type": "application/octet-stream"}, "\x00\x01"),
    })
    _requests.get = fake
    out = A.fetch_web_page("https://example.com/blob")
    check("not text" in out, f"a binary content-type is refused: {out[:70]}")
finally:
    _requests.get = _real_get

print("\n=== 10. the stripper ===")
html = ("<html><head><title>t</title><style>body{color:red}</style></head><body>"
        "<script>alert('x')</script><h1>Malus &amp; the PBS</h1>"
        "<p>signal &lt; 1</p><div>second   block</div><!-- hidden --><br>tail"
        "</body></html>")
text = A._html_to_text(html)
for gone in ("alert", "color:red", "hidden", "<h1>", "  "):
    check(gone not in text, f"{gone!r} does not survive")
check("Malus & the PBS" in text, "entities are unescaped")
check("second block" in text, "runs of whitespace collapse")
# Order matters: unescape AFTER stripping, or `&lt;script&gt;` in prose becomes a
# tag that the stripper has already run past.
escaped = A._html_to_text("<p>write &lt;script&gt;evil()&lt;/script&gt; here</p>")
check("evil()" in escaped and "script" in escaped,
      "escaped markup in prose stays prose rather than becoming a tag")

print("\n=== 11. search parses without a network ===")
atom = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2401.12345v1</id>
<published>2024-01-22T18:00:00Z</published>
<title>Superradiance in a
  cold ensemble</title>
<summary>We observe collective
 emission.</summary>
<author><name>A Adams</name></author><author><name>B Bell</name></author>
</entry></feed>"""
out = A._arxiv_entries(ET.fromstring(atom), "superradiance")
for want in ("2401.12345", "Superradiance in a cold ensemble", "A Adams, B Bell",
             "2024-01-22"):
    check(want in out, f"arXiv Atom yields {want!r}")
check("0 results" in A._arxiv_entries(
    ET.fromstring('<feed xmlns="http://www.w3.org/2005/Atom"/>'), "q"),
      "an empty feed says 0 results rather than raising")

hits = {"query": {"search": [
    {"title": "Malus's law",
     "snippet": 'the <span class="searchmatch">intensity</span> goes as cos&lt;sup&gt;2&lt;/sup&gt;'},
]}}
out = A._wikipedia_hits(hits, "malus")
check("en.wikipedia.org/wiki/Malus%27s_law" in out or "Malus" in out,
      "Wikipedia yields a page URL")
check("searchmatch" not in out, "and the snippet's markup is stripped")
check("0 results" in A._wikipedia_hits({}, "q"),
      "an empty payload says 0 results rather than raising")
check("must be" in A.search_web("x", source="google"),
      "an unknown source is refused by name")

print("\n=== 12. the history readers stay inside their roots ===")
for name in ("../../../Windows/System32/config", "C:/Users/someone/.env", "..",
             "", "a/b", "x:y"):
    out = A.read_report(name)
    check("is not a report name" in out or "no report matching" in out,
          f"read_report({name!r}) refuses: {out[:60]}")
check("is not a date" in A.read_notebook("yesterday"),
      "read_notebook only takes today/latest/YYYY-MM-DD")
check("is not a date" in A.read_notebook("../MEMORY"),
      "and a path is not a date either")

print("\n=== 13. the lead can summon it, on either provider ===")
# It could not, for as long as the second provider existed: `ask_advisor` was a
# Gemini FunctionDeclaration on `lead.extra_tools`, and `make_session` feeds that
# list to the Gemini branch only. Now it is a registry ToolSpec, so both views of
# the registry carry it.
lead_specs = {s.name for s in reg.specs_for("lead")}            # what Claude gets
lead_decls = {getattr(d, "name", "") for d in reg.declarations_for("lead")}
check("ask_advisor" in lead_specs, "ask_advisor reaches a Claude session")
check("ask_advisor" in lead_decls, "and a Gemini session")
check("ask_advisor" not in set(reg.names_for("coder")),
      "only the lead delegates -- a teammate cannot summon a teammate")
check(not hasattr(team["lead"], "extra_dispatch"),
      "and the second wiring path is gone, so it cannot regress to one provider")

print("\n=== 14. the advisor never writes memory, by either path ===")
# Excluding it from `compact_on_exit` was not enough: `_maybe_compact` runs
# mid-turn and is the path that actually fired on 2026-08-18, writing MEMORY.md,
# LABSCRIPT.md and two notebook entries out of a half-finished diagnosis.
import superradiant_assistant.orchestrator.agents as AG


class _FakeSession:
    def __init__(self, n):
        self._h = [{"role": "user", "content": "x" * 20_000} for _ in range(n)]

    def get_history(self):
        return self._h


curated = []
real_compact = MEMORY.compact
try:
    MEMORY.compact = lambda turns, client: curated.append(len(turns)) or True
    for role in ("advisor", "coder"):
        ag = team[role]
        ag._session = _FakeSession(20)
        ag._turns = 99
        before = len(curated)
        try:
            ag._maybe_compact()
        except Exception as e:
            # A rebuilt session needs a live provider; the curator call is what
            # this checks, and it happens first.
            print(f"    ({role} session rebuild needed a provider: "
                  f"{type(e).__name__})")
        fired = len(curated) > before
        check(fired == (role != "advisor"),
              f"{role}: curator {'ran' if fired else 'did not run'} on a "
              f"400k-char session" + (" -- correct" if fired == (role != "advisor")
                                       else " -- WRONG"))
finally:
    MEMORY.compact = real_compact

agents_src = Path(AG.__file__).read_text(encoding="utf-8")
check(agents_src.count("MEMORY.compact(") == 1,
      "there is exactly one call to the curator, and the advisor returns before it")
agent_src = (Path(__file__).resolve().parents[2] / "agent.py").read_text(
    encoding="utf-8")
check('name != "advisor"' in agent_src,
      "the exit flush filters it too -- both doors, not one")

print("\n=== 15. the blue panel, with and without the portrait ===")
import superradiant_assistant.splash as S
S.enable_ansi()


def _visible_widths(panel):
    return {S._plain_len(ln) for ln in panel.split("\n")}


def _blue(code):
    r, g, b = S.ADVISOR_BLUE
    return f"{r};{g};{b}" in code


# No asset: the header is the identity lines alone. This is the state on a machine
# where the portrait was never rendered, and it has to look deliberate.
real_load = S.load_art
try:
    S.load_art = lambda name: []
    bare = S.advisor_panel("No. The extinction is still complete.",
                           model="claude-opus-5", thinking="high", width=90)
    check(_visible_widths(bare) == {90},
          f"every row is exactly 90 visible columns ({sorted(_visible_widths(bare))})")
    check(_blue(bare), "the frame is the advisor's blue")
    cr, cg, cb = S.CARDINAL
    check(f"{cr};{cg};{cb}" not in bare,
          "and not the cardinal red every other reply uses")
    check("Advisor" in bare and "claude-opus-5" in bare and "thinking high" in bare,
          "the header names the role and the model it is running on")
    check("advisor" in bare.split("\n")[0], "the box is titled")

    # With an asset: 18 columns of half-block cells per row, each cell three
    # escape sequences and one glyph. The frame is measured in visible columns, so
    # this is where a width bug would show up.
    fake = ["".join(f"\x1b[38;2;10;20;30m\x1b[48;2;40;50;60m▀"
                    for _ in range(18)) + "\x1b[0m" for _ in range(6)]
    S.load_art = lambda name: fake
    withpic = S.advisor_panel("No.", model="claude-opus-5", thinking="high",
                              width=90)
    check(_visible_widths(withpic) == {90},
          f"still exactly 90 columns with the portrait in it "
          f"({sorted(_visible_widths(withpic))})")
    check(withpic.count("▀") == 18 * 6,
          f"all 18x6 portrait cells survive ({withpic.count(chr(0x2580))})")
    check("Advisor" in withpic, "and the identity block sits beside it")
finally:
    S.load_art = real_load

check("advisor" in S.ROLE_BLURB, "the /team screen has a blurb for it")
check(hasattr(S, "advisor_panel"), "advisor_panel is the public entry point")
# The lead's `ask_advisor` must NOT use this panel: that reply is an input to the
# lead's answer, not an answer to the operator, and two verdicts framed as verdicts
# on one screen is worse than one.
check(agent_src.count("advisor_panel") == 1,
      "exactly one call site -- the operator's /advisor, not the lead's consult")

print("\n=== 16b. search_lab_notes: the same RAG pipeline, pointed at the record ===")
# No monkeypatched embeddings here, deliberately -- this runs against whatever
# notebook pages and reports actually exist on this machine, the same way
# read_report/read_notebook already do above. What is checked is the shape, not
# the content, since the content is real lab data and changes day to day.
docs = A._notes_documents()
check(all(hasattr(d, "title") and hasattr(d, "path") and hasattr(d, "content")
          for d in docs),
      f"every document is chunk_document()-shaped ({len(docs)} documents)")
notebook_docs = [d for d in docs if d.path.startswith("notebook/")]
check(len(notebook_docs) == len(MEMORY.episode_days()),
      f"one document per non-empty notebook day ({len(notebook_docs)})")

out = A.search_lab_notes("has this signal level been seen before")
check(isinstance(out, str) and out,
      "returns a string either way -- no crash with zero documents or zero API key")
check(out.startswith("found ") or "no notebook pages or reports exist yet" in out,
      f"a recognisable result shape: {out[:70]!r}")
if docs:
    check("found 0 chunks" not in out or all(len(d.content) < 120 for d in docs),
          "non-trivial notebook/report text actually produces chunks")

print("\n=== 16c. search_lab_notes(day=...): a real filter, not a hope ===")
check(A._resolve_day("not-a-date")[1] is not None,
      "bad day strings are rejected the same way as read_notebook's")
check(A._resolve_day("today")[0] == A._resolve_day("")[0] and A._resolve_day("today")[1] is None,
      "'today' resolves cleanly")

real_days = MEMORY.episode_days()
if real_days:
    target = real_days[0]
    scoped = A._notes_documents(day=target)
    unscoped_notebook_days = {d.path for d in docs if d.path.startswith("notebook/")}
    check(len(scoped) <= len(docs), f"scoped is a subset ({len(scoped)} <= {len(docs)})")
    check(all(d.path == f"notebook/{target}.md" or Path(d.path).parent.name == target
              for d in scoped),
          f"every scoped document actually belongs to {target}, not just ranked near it")
    check(f"notebook/{target}.md" in {d.path for d in scoped} or not MEMORY.read_episode(target).strip(),
          "the target day's own page is included when it has content")
    other_days = [d for d in real_days if d != target]
    if other_days:
        check(f"notebook/{other_days[0]}.md" not in {d.path for d in scoped},
              "a different day's page is excluded, not just ranked lower")

    out_scoped = A.search_lab_notes("what was measured", day=target)
    check(out_scoped.startswith("found ") or f"no notebook page or report for {target}" in out_scoped,
          f"day-scoped search returns a recognisable result: {out_scoped[:70]!r}")
else:
    print("    (no notebook pages exist on this machine -- skipping the subset checks)")

out_bad_day = A.search_lab_notes("x", day="not-a-date")
check("is not a date" in out_bad_day,
      f"an invalid day is rejected before any retrieval runs: {out_bad_day[:60]!r}")

spec = next(s for s in build_advisor_tool_specs() if s.name == "search_lab_notes")
check("day" in spec.parameters["properties"], "the schema exposes the new parameter")
check(spec.parameters["required"] == ["query"], "but day stays optional")

print("\n=== 16. what a summon costs ===")
for role in ("lead", "coder", "planner", "advisor"):
    s = team[role].build_system_instruction()
    print(f"    {role:8} {len(reg.names_for(role)):2d} tools  {len(s):7,} chars"
          f"  ~{len(s) // 4:6,} tokens of prompt, resent on every tool call")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
