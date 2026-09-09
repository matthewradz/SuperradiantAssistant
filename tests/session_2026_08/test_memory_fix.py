"""Verify the memory fixes with a fake LLM -- no API calls, no hardware."""
import sys, types, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


from superradiant_assistant.memory import (
    MemoryStore, COMPACT_AFTER_CHARS, COMPACT_AFTER_TURNS, KEEP_RECENT_TURNS,
)
from superradiant_assistant.creative.evidence import SessionEvidence


class FakeResp:
    def __init__(self, text): self.text = text


class FakeLLM:
    """Returns the three tagged blocks the curator prompt asks for."""
    def __init__(self): self.prompts = []

    def generate(self, prompt, system=None, temperature=0.2):
        self.prompts.append(prompt)
        return FakeResp(
            "<episode>- measured -3 dB at 14.6 kHz</episode>"
            "<updated_memory>Filter cutoff 14.6 kHz. Scope CH3 unused.</updated_memory>"
            "<updated_user>Marcus runs the Cesium apparatus.</updated_user>"
        )


print("=== 1. evidence knows whether the session did anything ===")
ev = SessionEvidence()
check(not ev.did_substantive_work, "fresh session: nothing substantive")
check(ev.work_summary() == "nothing", f"summary: {ev.work_summary()!r}")
ev.record_action("wrote shot filter_scan.py")
check(ev.did_substantive_work, "after writing a script: substantive")
ev.record_queue()
ev.record_rows([{"shot_id": "2026-08-08_0009", "CHAN1_vpp": 1.08}])
print("    summary:", ev.work_summary())
check("measured shot" in ev.work_summary(), "summary mentions measurements")
ev2 = SessionEvidence()
ev2.record_action("x"); ev2.record_action("x")
check(len(ev2.actions) == 1, "duplicate actions are not double-counted")

print("\n=== 2. compact_on_exit writes all three layers ===")


class FakePart:
    def __init__(self, text=None, call=None, result=None, result_of=None):
        self.text = text
        self.function_call = (types.SimpleNamespace(name=call, args={"q": "x"})
                              if call else None)
        self.function_response = (
            types.SimpleNamespace(name=result_of, response={"result": result})
            if result_of else None)


class FakeContent:
    def __init__(self, role, *parts):
        self.role = role
        self.parts = list(parts)


class FakeSession:
    def __init__(self, contents): self._c = contents
    def get_history(self): return self._c


class FakeAgent:
    def __init__(self, name, contents):
        self.name = name
        self._session = FakeSession(contents)


with tempfile.TemporaryDirectory() as td:
    store = MemoryStore(root=Path(td))
    llm = FakeLLM()
    agents = [
        FakeAgent("lead", [FakeContent("user", FakePart("measure the filter")),
                           FakeContent("model", FakePart(call="run_sweep"))]),
        FakeAgent("coder", [FakeContent("user", FakePart("write the analysis"))]),
    ]
    wrote = store.compact_on_exit(agents, llm, evidence=ev)
    check(wrote, "compact_on_exit reports success")
    check(store.memory_file.exists(), f"MEMORY.md created: {store.read_memory()!r}")
    check(store.operator_file.exists(),
          f"OPERATOR.md created: {store.read_user()!r}")
    check(store.episode_path().exists(), "today's episode file created")
    # A store given an explicit root must keep the shared tier inside it, or a
    # test writes LABSCRIPT.md and OPERATOR.md into the repo. One did.
    check(str(store.shared_root).startswith(td),
          f"shared memory is isolated under the temp root: {store.shared_root}")
    block = store.build_context_block()
    # Layers are now labelled by how far they travel, not by filename.
    for layer in ("shared by every apparatus", "This apparatus",
                  "Today's page from the lab notebook"):
        check(layer in block, f"context block contains '{layer}'")
    check("[called run_sweep" in llm.prompts[0],
          "tool calls survive into the transcript, not just text")
    prompt_up = llm.prompts[0].upper()          # compact() upper-cases the role
    check("LEAD:USER" in prompt_up and "CODER:USER" in prompt_up,
          "both agents' histories are folded in")

print("\n=== 3. it refuses to write for an empty session ===")
with tempfile.TemporaryDirectory() as td:
    store = MemoryStore(root=Path(td))
    llm = FakeLLM()
    empty = SessionEvidence()
    wrote = store.compact_on_exit(
        [FakeAgent("lead", [FakeContent("user", FakePart("hi"))])], llm,
        evidence=empty)
    check(not wrote, "no work -> no compaction")
    check(not llm.prompts, "no API call was made")
    check(not store.memory_file.exists(), "MEMORY.md not created for idle chat")

print("\n=== 4. agents with no session are skipped, not crashed on ===")
with tempfile.TemporaryDirectory() as td:
    store = MemoryStore(root=Path(td))
    never_used = FakeAgent("answer", [])
    never_used._session = None
    wrote = store.compact_on_exit([never_used], FakeLLM(), evidence=ev)
    check(not wrote, "all-empty team -> nothing written, no exception")

print("\n=== 5. the size trigger ===")
from superradiant_assistant.orchestrator.agents import Agent
print(f"    COMPACT_AFTER_CHARS = {COMPACT_AFTER_CHARS:,}")
print(f"    COMPACT_AFTER_TURNS = {COMPACT_AFTER_TURNS} (secondary)")
big = "x" * 20_000
a = Agent("lead", registry=None)
a._session = FakeSession([FakeContent("user", FakePart(big)) for _ in range(4)])
check(a.context_chars() == 80_000, f"context_chars counts text: {a.context_chars():,}")
check(a.context_chars() > COMPACT_AFTER_CHARS,
      "80k chars would now trigger compaction (old rule: 4 messages, no trigger)")
a2 = Agent("lead", registry=None)
a2._session = None
check(a2.context_chars() == 0, "no session -> 0 chars, no crash")

print("\n=== 6. tool results are counted, not scored as zero ===")
from superradiant_assistant.memory import history_text, content_size
big_result = "line\n" * 2000                       # a file read, ~10k chars
c = FakeContent("user", FakePart(result_of="read_lab_file", result=big_result))
old_way = " ".join(getattr(p, "text", "") or "" for p in c.parts).strip()
curator = history_text(c)
measured = content_size(c)
print(f"    old _content_text     : {len(old_way):>7} chars")
print(f"    curator transcript    : {len(curator):>7} chars (truncated on purpose)")
print(f"    content_size (trigger): {measured:>7} chars (untruncated)")
check(len(old_way) == 0, "the old rule scored a 10k tool result as 0 characters")
check("read_lab_file" in curator, "the curator still sees which tool produced it")
check(len(curator) < 2000, "the curator prompt cannot be swamped by one result")
check(measured > 9000,
      "the size trigger measures the REAL context, not the truncated view")
check(measured > len(curator) * 5,
      "trigger and transcript deliberately measure different things")

print("\n=== 7. the old rule genuinely never fired on real data ===")
import json
from datetime import datetime
from superradiant_assistant.config import CONFIG
hist = CONFIG.historical_data_root.parent / "agent_memory" / "history.jsonl"
check(hist.exists(), f"history.jsonl found at {hist}")
if hist.exists():
    ins, sizes = [], {}
    for line in hist.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        agent = r["role"].split(":")[0]
        sizes[agent] = sizes.get(agent, 0) + len(str(r.get("content", "")))
        if r["role"].endswith(":in"):
            ins.append(r)
    # Split into processes and find the busiest single agent-in-a-process.
    sessions, cur, last = [], [], None
    for r in ins:
        ts = datetime.fromisoformat(r["ts"])
        if last and (ts - last).total_seconds() > 600:
            sessions.append(cur); cur = []
        cur.append(r); last = ts
    sessions.append(cur)
    worst = max(max(
        (sum(1 for x in s if x["role"].split(":")[0] == a)
         for a in {x["role"].split(":")[0] for x in s}), default=0)
        for s in sessions)
    print(f"    {len(sessions)} sessions; busiest single agent reached {worst} "
          f"messages (needed {COMPACT_AFTER_TURNS})")
    print("    logged chars per agent (EXCLUDES all tool traffic):",
          {k: f"{v:,}" for k, v in sizes.items()})
    check(worst < COMPACT_AFTER_TURNS,
          "confirms the old per-message rule could never have fired")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
