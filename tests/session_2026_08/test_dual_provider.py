"""Two providers, one context, one tool set, one skill set.

Runs without either provider's key: the sessions are built, the transcript
conversions are exercised in both directions, and the tool loop is driven
against a scripted `_create`. What it does not check is that the wire format is
accepted -- that needs a live key.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


from superradiant_assistant.llm.client import (
    ClaudeAgentSession, GeminiAgentSession, Turn, ToolCall, make_session, is_claude,
)
from superradiant_assistant.tools import build_registry
from superradiant_assistant.hooks import default_gate

print("\n=== 1. one registry, two views, same tools ===")
reg = build_registry(gate=default_gate(), creative=True)
for role in ("lead", "coder", "planner", "answer"):
    decls = reg.declarations_for(role)
    specs = reg.specs_for(role)
    check([d.name for d in decls] == [s.name for s in specs],
          f"{role}: the Gemini and neutral views list the same {len(specs)} tools")
specs = reg.specs_for("lead")
check(all(isinstance(s.parameters, dict) and s.parameters.get("type") == "object"
          for s in specs),
      "every ToolSpec.parameters is a JSON-Schema object — Anthropic takes it verbatim")
claude_tools = [{"name": s.name, "description": s.description,
                 "input_schema": s.parameters} for s in specs]
# Not a hardcoded count -- that only records how many tools existed the day the
# test was written, and fails every time one is added. The invariant is that the
# whitelist and what Claude actually receives are the same set.
check(sorted(s.name for s in specs) == sorted(reg.names_for("lead")),
      f"all {len(claude_tools)} of the lead's whitelisted tools reach Claude")
check(all(t["description"] for t in claude_tools), "and every one carries its description")

print("\n=== 2. provider is chosen by the model name ===")
check(is_claude("claude-opus-5") and not is_claude("gemini-3.6-flash"),
      "is_claude routes on the name")
s = make_session(model="claude-opus-5", system_instruction="sys",
                 tool_specs=specs, tool_declarations=[], dispatch=lambda n, a: "ok")
check(isinstance(s, ClaudeAgentSession), f"claude-opus-5 -> {type(s).__name__}")
check(len(s._tools) == len(specs), f"and it received all {len(s._tools)} tools")
check(s._system == "sys", "and the same system instruction")

print("\n=== 3. the context survives the crossing ===")
transcript = [
    Turn(role="user", text="sweep the waveplate 0 to 45"),
    Turn(role="assistant", text="Reading the globals first.",
         calls=[ToolCall(id="toolu_01", name="get_runmanager_globals", args={})]),
    Turn(role="tool", results=[("toolu_01", "get_runmanager_globals",
                                "read 1 globals: waveplate_angle=0.0")]),
    Turn(role="assistant", text="waveplate_angle is 0 deg. Running the sweep."),
]
msgs = ClaudeAgentSession.messages_from_transcript(transcript)
for m in msgs:
    kind = m["content"] if isinstance(m["content"], str) else \
        [b["type"] for b in m["content"]]
    print(f"    {m['role']:9} {kind}")
check(len(msgs) == 4, f"four Anthropic messages ({len(msgs)})")
check(msgs[1]["content"][1]["type"] == "tool_use", "the call became a tool_use block")
check(msgs[2]["content"][0]["tool_use_id"] == "toolu_01",
      "and its result is matched by tool_use_id")
check("waveplate_angle=0.0" in msgs[2]["content"][0]["content"],
      "carrying what the tool actually returned")

sess = ClaudeAgentSession(model="claude-opus-5", tool_specs=specs,
                          dispatch=lambda n, a: "ok", seed_transcript=transcript)
back = sess.export_transcript()
check([t.role for t in back] == ["user", "assistant", "tool", "assistant"],
      f"round trip preserves the roles: {[t.role for t in back]}")
check(back[1].calls and back[1].calls[0].name == "get_runmanager_globals",
      "and the tool call survives")
check(back[0].text == transcript[0].text and back[3].text == transcript[3].text,
      "and the text is unchanged")

print("\n=== 4. an unanswered tool call is dropped, not sent ===")
# A session rebuilt between a call and its result would otherwise send a
# tool_use with no tool_result, which the API rejects outright.
half = transcript[:2]
msgs2 = ClaudeAgentSession.messages_from_transcript(half)
blocks = [b["type"] for m in msgs2 for b in
          (m["content"] if isinstance(m["content"], list) else [])]
print(f"    blocks: {blocks}")
check("tool_use" not in blocks, "the dangling tool_use is gone")
check(any(b == "text" for b in blocks), "but the assistant's text is kept")
check(ClaudeAgentSession.messages_from_transcript([]) == [], "empty is empty")

print("\n=== 5. the tool loop dispatches and answers in one message ===")


class FakeBlock:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeResp:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = FakeBlock(input_tokens=10, output_tokens=5)


calls_seen = []
sess2 = ClaudeAgentSession(model="claude-opus-5", tool_specs=specs,
                           dispatch=lambda n, a: calls_seen.append((n, a)) or f"{n}-done")
scripted = [
    FakeResp([FakeBlock(type="text", text="checking"),
              FakeBlock(type="tool_use", id="toolu_a", name="list_scripts", input={}),
              FakeBlock(type="tool_use", id="toolu_b", name="get_lyse_routines", input={})],
             stop_reason="tool_use"),
    FakeResp([FakeBlock(type="text", text="Both read. Done.")]),
]
sess2._create = lambda: scripted.pop(0)
out = sess2.send_message("what is loaded?")
print(f"    dispatched: {[n for n, _ in calls_seen]}")
print(f"    reply     : {out.text!r}")
check([n for n, _ in calls_seen] == ["list_scripts", "get_lyse_routines"],
      "both parallel calls were dispatched")
def _is_result(b):
    # assistant blocks come back as SDK objects and are appended verbatim;
    # only the ones we build ourselves are dicts.
    return isinstance(b, dict) and b.get("type") == "tool_result"


result_msgs = [m for m in sess2._messages
               if isinstance(m["content"], list) and any(map(_is_result, m["content"]))]
check(len(result_msgs) == 1,
      f"and both results went back in ONE user message ({len(result_msgs)})")
check(len(result_msgs[0]["content"]) == 2, "containing two tool_result blocks")
check(out.text == "Both read. Done.", "the final text is returned")

print("\n=== 6. a refusal is raised, not indexed into ===")
sess3 = ClaudeAgentSession(model="claude-opus-5", tool_specs=[], dispatch=lambda n, a: "")
sess3._create = lambda: FakeResp([], stop_reason="refusal")
try:
    sess3.send_message("...")
    check(False, "a refusal should raise")
except RuntimeError as e:
    check("declined" in str(e), f"raised with a usable message: {str(e)[:60]}...")

print("\n=== 7. no key, no silent fallback ===")
saved = os.environ.pop("ANTHROPIC_API_KEY")
try:
    ClaudeAgentSession(model="claude-opus-5")
    check(False, "a missing key should raise")
except RuntimeError as e:
    check("ANTHROPIC_API_KEY" in str(e), "names the variable to set")
    check("Gemini key" in str(e), "and says the Gemini key will not work")
finally:
    os.environ["ANTHROPIC_API_KEY"] = saved

print("\n=== 8. the Gemini path still builds the same way ===")
import inspect
sig = inspect.signature(GeminiAgentSession.__init__)
check("seed_transcript" in sig.parameters, "it accepts a neutral transcript too")
check("seed_history" in sig.parameters, "and still accepts native history")
for name in ("export_transcript", "get_history", "send_message"):
    check(hasattr(GeminiAgentSession, name) and hasattr(ClaudeAgentSession, name),
          f"both sessions expose {name}()")

print("\n=== 9. a sliced history is legal to send back ===")
from superradiant_assistant.llm.client import _drop_unanswered_calls, make_client
from superradiant_assistant.memory import content_size, history_text

full = ClaudeAgentSession.messages_from_transcript(transcript)
# Compaction keeps the tail, which can start with results whose call was cut.
tail = _drop_unanswered_calls(full[2:])
print(f"    tail roles: {[m['role'] for m in tail]}")
check(all(not (isinstance(m['content'], list)
               and any(isinstance(b, dict) and b.get('type') == 'tool_result'
                       for b in m['content']))
          for m in tail),
      "an orphaned tool_result at the head is dropped")
check(tail and tail[-1]["role"] == "assistant", "and the real turns survive")
head = _drop_unanswered_calls(full[:2])
blocks = [b["type"] for m in head for b in m["content"] if isinstance(b, dict)]
check("tool_use" not in blocks, "an unanswered tool_use at the tail is dropped")
check(_drop_unanswered_calls(full) == full, "an intact history is left alone")

reborn = ClaudeAgentSession(model="claude-opus-5", tool_specs=[],
                            dispatch=lambda n, a: "", seed_messages=full)
check(len(reborn.get_history()) == len(full),
      "a session rebuilt from its own history keeps every message")

print("\n=== 10. the size trigger and the curator can read Claude history ===")
sizes = [content_size(m) for m in full]
texts = [history_text(m) for m in full]
print(f"    sizes: {sizes}")
print(f"    curator sees: {texts[1]!r}")
check(all(s > 0 for s in sizes), "every message measures more than zero chars")
check("[called get_runmanager_globals(" in texts[1], "the tool call is rendered")
check("waveplate_angle=0.0" in texts[2], "and so is what the tool returned")

print("\n=== 11. single-shot clients route too ===")
from superradiant_assistant.llm.client import ClaudeClient
c = make_client("claude-opus-5")
check(isinstance(c, ClaudeClient), f"claude-opus-5 -> {type(c).__name__}")
check(type(make_client("gemini-3.6-flash")).__name__ == "GeminiClient",
      "gemini-3.6-flash -> GeminiClient")
from superradiant_assistant.llm.cost_tracker import PRICE_TABLE
check("claude-opus-5" in PRICE_TABLE, "and its price is known, so cost is not $0")

print("\n=== 12. a real Agent, on Claude, keeps its tools and its memory ===")
from superradiant_assistant.orchestrator.agents import Agent
from superradiant_assistant.llm.cost_tracker import CostTracker

reg = build_registry(gate=default_gate(), creative=True)
a = Agent(name="lead", model="claude-opus-5", registry=reg,
          cost_tracker=CostTracker())
sess = a.session
check(isinstance(sess, ClaudeAgentSession), f"the lead built a {type(sess).__name__}")
check(len(sess._tools) == len(reg.names_for("lead")),
      f"holding all {len(sess._tools)} of the lead's tools")
sys_text = sess._system
check("SKILL" in sys_text.upper() or "skill" in sys_text,
      "and the same system instruction, skills included")
check(len(sys_text) > 2000, f"({len(sys_text):,} chars of role + rules + memory)")

sess._create = lambda: FakeResp([FakeBlock(type="text", text="Angle is 0 deg.")])
reply = a.send("what angle is the waveplate at?")
check(reply == "Angle is 0 deg.", f"a turn runs end to end: {reply!r}")

note = a.switch_model("gemini-3.6-flash")
print(f"    {note}")
check(type(a.session).__name__ == "GeminiAgentSession",
      "switching mid-session lands on the other provider")
carried = a.session.get_history()
check(len(carried) >= 2, f"carrying {len(carried)} history entries across")
check(any("waveplate" in history_text(c) for c in carried),
      "and the question it was asked is still in context")
print("\n=== 13. the growing history gets a cache breakpoint too, not just system ===")
from superradiant_assistant.llm.client import _with_trailing_cache_mark

check(_with_trailing_cache_mark([]) == [], "empty history is left alone")

str_msgs = [{"role": "user", "content": "sweep the waveplate"}]
marked = _with_trailing_cache_mark(str_msgs)
check(marked[-1]["content"][0]["cache_control"] == {"type": "ephemeral"},
      "a plain string message is wrapped in a block and marked")
check(marked[-1]["content"][0]["text"] == "sweep the waveplate", "text preserved")
check(str_msgs[0]["content"] == "sweep the waveplate",
      "the original message is untouched -- this is a copy for the wire")

dict_msgs = [{"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "t1", "content": "a"},
    {"type": "tool_result", "tool_use_id": "t2", "content": "b"},
]}]
marked2 = _with_trailing_cache_mark(dict_msgs)
check("cache_control" not in marked2[-1]["content"][0],
      "only the LAST block of the last message is marked")
check(marked2[-1]["content"][1]["cache_control"] == {"type": "ephemeral"},
      "the last block is")
check("cache_control" not in dict_msgs[-1]["content"][1],
      "and the original dict is untouched")


class FakeSDKBlock:
    """Stands in for anthropic's TextBlock/ToolUseBlock -- a real object, not a dict."""
    def __init__(self, **kw):
        self._data = kw

    def model_dump(self, exclude_none=True):
        return dict(self._data)


sdk_msgs = [{"role": "assistant",
            "content": [FakeSDKBlock(type="text", text="checking")]}]
marked3 = _with_trailing_cache_mark(sdk_msgs)
check(marked3[-1]["content"][0]["cache_control"] == {"type": "ephemeral"},
      "an SDK response block (not a dict) can be marked too")
check(marked3[-1]["content"][0]["text"] == "checking",
      "without losing its own fields")
check(isinstance(sdk_msgs[-1]["content"][0], FakeSDKBlock),
      "and the original SDK object in history is never mutated")

print("\n=== 14. the wire request actually carries that breakpoint ===")
captured = {}
sess4 = ClaudeAgentSession(model="claude-opus-5", tool_specs=[],
                          dispatch=lambda n, a: "")


def fake_create(**kwargs):
    captured.update(kwargs)
    return FakeResp([FakeBlock(type="text", text="ok")])


sess4._client.messages.create = fake_create
sess4.send_message("what angle is the waveplate at?")
last_block = captured["messages"][-1]["content"][-1]
check(last_block.get("cache_control") == {"type": "ephemeral"},
      "the real _create() marks the trailing block, not just a helper in isolation")
check(sess4._messages[0]["content"] == "what angle is the waveplate at?",
      "and the session's own stored history stays an unmarked plain string")

print("\n=== 15. cache tokens are tracked and priced separately from fresh input ===")
from superradiant_assistant.llm.cost_tracker import CostTracker as _CT

ct = _CT(use_free_tier_pricing=False)
ct.record("claude-opus-5", input_tokens=1000, output_tokens=100,
          cache_read_tokens=2000, cache_creation_tokens=500)
u = ct._by_model["claude-opus-5"]
check(u.cache_read_tokens == 2000 and u.cache_creation_tokens == 500,
      "both cache fields are stored")
check(ct.total_tokens == 1000 + 100 + 2000 + 500,
      f"and counted in the total, not silently dropped ({ct.total_tokens})")
price = 5.00 / 1_000_000       # claude-opus-5 input price per token
expected = (1000 * price + 100 * (25.00 / 1_000_000)
            + 2000 * price * 0.1 + 500 * price * 1.25)
check(abs(ct.dollars() - expected) < 1e-9,
      f"cache reads price at 0.1x input and writes at 1.25x ({ct.dollars():.6f})")
ct_default = _CT()
ct_default.record("claude-opus-5", 1000, 100, cache_read_tokens=2000)
check(ct_default.dollars() == 0.0, "free-tier reporting still zeroes it out")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
