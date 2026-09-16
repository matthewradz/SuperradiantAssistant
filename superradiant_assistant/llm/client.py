"""LLM client abstraction — Gemini only, via the google-genai SDK.

Two ways to talk to the model:
- GeminiClient.generate(): stateless single-shot call (system + prompt in, text out).
  Used by call sites that don't need multi-turn history (preprocessor, post-hoc
  interpretation of analysis results, etc).
- GeminiAgentSession: persistent multi-turn chat with tool use. Automatic Function
  Calling is explicitly disabled so tool calls surface to application code instead
  of being executed inside the SDK — this is what lets a PreToolUse-style hook see
  and gate every tool call before it runs (required for anything that can touch
  real hardware). Tool dispatch is injected via a callback so this module has no
  dependency on the tools package.
"""
from __future__ import annotations
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable

# Auto-load .env if present (no error if dotenv isn't installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from superradiant_assistant.llm.cost_tracker import CostTracker

DEFAULT_MODEL = "gemini-3.6-flash"


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Optional[Any] = None


@dataclass
class ToolCall:
    """One function call requested by the model."""
    id: str
    name: str
    args: Dict[str, Any]


@dataclass
class Turn:
    """One conversation turn, in neither provider's dialect.

    This is what lets two providers share a context. Each session keeps its own
    native history for the live conversation -- nothing is lost in translation
    while a model is running -- and converts to and from this shape only at the
    seam, when a session is rebuilt or the model is switched.

    A turn is either plain text or a tool exchange:
      role="user"      text=...                     the operator, or a seeded turn
      role="assistant" text=... calls=[ToolCall]    the model, optionally calling tools
      role="tool"      results=[(id, name, output)] what the tools returned
    """
    role: str
    text: str = ""
    calls: List[ToolCall] = field(default_factory=list)
    results: List[tuple] = field(default_factory=list)   # (call_id, name, output)


def _resolve_gemini_key(api_key: Optional[str] = None) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env or environment."
        )
    return key


def _extract_usage(resp) -> tuple[int, int]:
    meta = getattr(resp, "usage_metadata", None)
    if meta is None:
        return 0, 0
    return (
        int(getattr(meta, "prompt_token_count", 0) or 0),
        int(getattr(meta, "candidates_token_count", 0) or 0),
    )


def _extract_text(resp) -> str:
    try:
        return resp.text or ""
    except Exception:
        try:
            return resp.candidates[0].content.parts[0].text
        except Exception:
            return ""


#: Hard deadline for one provider request, in milliseconds. A Pro model with a
#: large tool schema can legitimately take tens of seconds, so this is well
#: above normal; its job is to stop an unanswered connection hanging forever.
REQUEST_TIMEOUT_MS = 180_000

#: Ceiling on the backoff between retries, in seconds. Unbounded doubling makes
#: the tail waits longer than the spike they are waiting out.
MAX_RETRY_DELAY = 15.0


def _sleep_interruptibly(delay: float, label: str = "") -> None:
    """Sleep in short steps, showing progress, so Ctrl-C is honoured promptly.

    One long sleep is interruptible in principle, but it leaves the terminal
    silent, which is indistinguishable from a hang. Counting down out loud also
    means the operator can see a retry is in progress rather than guessing.
    """
    import time
    waited = 0.0
    while waited < delay:
        step = min(1.0, delay - waited)
        time.sleep(step)
        waited += step
        if label and waited % 5 < 1.0 and waited < delay:
            print(f"  [LLM] {label} {delay - waited:.0f}s...", flush=True)


#: Provider-side conditions worth waiting out.
_TRANSIENT_MARKERS = (
    "503", "429", "500", "502", "504", "UNAVAILABLE", "RATE_LIMIT",
    # Network and protocol failures are just as transient as an HTTP 503, and
    # matching only on status codes let a dropped connection kill a session
    # several minutes into its work.
    "RemoteProtocolError", "Server disconnected", "ConnectionError",
    "ConnectError", "ReadTimeout", "ReadError", "WriteError",
    "Connection reset", "Connection aborted", "timed out", "TimeoutError",
    "ssl", "SSLError", "IncompleteRead",
)


def _is_transient(exc: Exception) -> bool:
    """Whether `exc` is worth retrying rather than surfacing."""
    text = f"{type(exc).__name__}: {exc}"
    return any(marker.lower() in text.lower() for marker in _TRANSIENT_MARKERS)


def _call_interruptibly(fn, poll: float = 0.2):
    """Run fn() on a worker thread while the main thread waits in small steps.

    Ctrl-C is delivered to the main thread, but Python can only act on it
    between bytecodes: while the main thread sits inside a C-level socket read
    the interrupt is simply queued, and the session cannot be stopped at all.
    Waiting on a thread instead keeps the main thread in Python, so the
    interrupt lands immediately.

    The worker is a daemon, so an abandoned request cannot hold the process open.
    """
    import threading

    box: Dict[str, Any] = {}

    def work():
        try:
            box["value"] = fn()
        except BaseException as e:      # noqa: BLE001 - re-raised on the caller's thread
            box["error"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    while t.is_alive():
        t.join(poll)
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _retry_on_transient(fn, max_retries: int, base_delay: float):
    """Run fn() with exponential backoff on transient provider errors."""
    import time

    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(max_retries):
        started = time.time()
        try:
            return _call_interruptibly(fn)
        except KeyboardInterrupt:
            # Never swallow the operator's interrupt as a "transient error":
            # doing so turns Ctrl-C into a longer wait.
            print("\n  [LLM] interrupted by operator", flush=True)
            raise
        except Exception as e:
            last_exc = e
            err_str = str(e)
            elapsed = time.time() - started
            if _is_transient(e):
                # Cap the backoff. Doubling without a ceiling means the last
                # waits dominate: 5+10+20+40 spends over a minute and then
                # gives up anyway, which is the worst of both. A capped delay
                # with more attempts rides out a demand spike instead.
                import random
                delay = min(base_delay * (2 ** attempt), MAX_RETRY_DELAY)
                delay *= 0.7 + 0.6 * random.random()   # jitter, to desynchronise retries
                print(f"  [LLM] {type(e).__name__} after {elapsed:.0f}s "
                      f"(attempt {attempt+1}/{max_retries}), retrying in "
                      f"{delay:.0f}s", flush=True)
                _sleep_interruptibly(delay, label="retrying in")
                continue
            raise
    raise RuntimeError(f"LLM call failed after {max_retries} attempts. Last error: {last_exc}")


class LLMClient(ABC):
    def __init__(self, model: str, cost_tracker: Optional[CostTracker] = None):
        self.model = model
        self.cost_tracker = cost_tracker or CostTracker()

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.2,
    ) -> LLMResponse: ...


class GeminiClient(LLMClient):
    """Stateless single-shot Gemini calls, via google-genai."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        cost_tracker: Optional[CostTracker] = None,
    ):
        super().__init__(model=model, cost_tracker=cost_tracker)
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as e:
            raise ImportError(
                "google-genai not installed. Run: pip install google-genai"
            ) from e

        # Give every request a hard deadline.
        #
        # Without one, a connection the provider accepts but never answers
        # blocks forever, and on Windows that wait sits inside a C-level socket
        # read where Ctrl-C cannot reach: the session looks hung and refuses to
        # die. A timeout turns that into an ordinary error the retry logic can
        # handle. The value is generous because a large tool schema on a Pro
        # model genuinely takes tens of seconds.
        self._client = genai.Client(
            api_key=_resolve_gemini_key(api_key),
            http_options=genai_types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )
        self._types = genai_types

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.2,
        max_retries: int = 8,
        base_delay: float = 3.0,
    ) -> LLMResponse:
        if self.cost_tracker.over_budget():
            raise RuntimeError(f"Cost cap exceeded: {self.cost_tracker.summary()}")

        cfg_kwargs: Dict[str, Any] = {"temperature": float(temperature)}
        if system:
            cfg_kwargs["system_instruction"] = system
        if max_output_tokens is not None:
            cfg_kwargs["max_output_tokens"] = int(max_output_tokens)
        config = self._types.GenerateContentConfig(**cfg_kwargs)

        def _call():
            resp = self._client.models.generate_content(
                model=self.model, contents=prompt, config=config,
            )
            text = _extract_text(resp)
            in_toks, out_toks = _extract_usage(resp)
            self.cost_tracker.record(self.model, in_toks, out_toks)
            return LLMResponse(text=text, model=self.model, input_tokens=in_toks,
                                output_tokens=out_toks, raw=resp)

        return _retry_on_transient(_call, max_retries, base_delay)


#: Thinking is billed as output tokens and paid for in wall-clock. Left
#: unconfigured, Gemini 3 picks its own level and spends 15-20 s deliberating
#: before every tool call -- a session that made 20 calls took 462 s, of which
#: the tools themselves accounted for under a minute. Almost all of those turns
#: are "which tool next", which does not need deliberation; the ones that do
#: (writing code, drawing a conclusion) are worth raising the level for by hand.
_THINKING_LEVELS = ("minimal", "low", "medium", "high")

#: Rough token equivalents, for SDKs/models that take a budget rather than a level.
_THINKING_BUDGETS = {"minimal": 0, "low": 512, "medium": 4096, "high": 16384}


def _thinking_config(genai_types, level: str):
    """Build a ThinkingConfig, or None if this SDK/model has no such knob.

    Gemini 3 takes `thinking_level`; 2.5 takes an integer `thinking_budget`.
    Both spellings are attempted because the installed SDK decides which exists,
    and guessing wrong would raise at construction and take the session with it.
    """
    level = (level or "low").lower()
    if level in ("", "default", "auto"):
        return None
    if level not in _THINKING_LEVELS:
        level = "low"
    cls = getattr(genai_types, "ThinkingConfig", None)
    if cls is None:
        return None
    for kwargs in ({"thinking_level": level},
                   {"thinking_budget": _THINKING_BUDGETS[level]}):
        try:
            return cls(**kwargs)
        except Exception:
            continue
    return None


class GeminiAgentSession:
    """Persistent multi-turn Gemini chat with tool use and hook-gated dispatch.

    `tool_declarations` are plain google.genai.types.FunctionDeclaration objects
    (built by the tools package). `dispatch` is called as dispatch(name, args)
    for every function call the model requests, and must return a string or a
    JSON-serializable dict — whatever comes back is what the model sees as the
    tool result. Automatic Function Calling is disabled so every call passes
    through `dispatch` (and therefore through whatever hook `dispatch` wraps).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system_instruction: Optional[str] = None,
        tool_declarations: Optional[list] = None,
        dispatch: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        api_key: Optional[str] = None,
        cost_tracker: Optional[CostTracker] = None,
        temperature: float = 0.2,
        seed_history: Optional[list] = None,
        thinking_level: str = "low",
        seed_transcript: Optional[List[Turn]] = None,
    ):
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as e:
            raise ImportError(
                "google-genai not installed. Run: pip install google-genai"
            ) from e

        self.model = model
        self.cost_tracker = cost_tracker or CostTracker()
        self._dispatch = dispatch
        self._types = genai_types
        # Same hard deadline as GeminiClient. It was missing here, which is the
        # path the agents actually use: a request the provider accepted but never
        # answered blocked forever inside a C-level socket read, where Ctrl-C
        # cannot reach it.
        self._client = genai.Client(
            api_key=_resolve_gemini_key(api_key),
            http_options=genai_types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )

        cfg_kwargs: Dict[str, Any] = {"temperature": float(temperature)}
        if system_instruction:
            cfg_kwargs["system_instruction"] = system_instruction
        if tool_declarations:
            cfg_kwargs["tools"] = [genai_types.Tool(function_declarations=tool_declarations)]
            cfg_kwargs["automatic_function_calling"] = genai_types.AutomaticFunctionCallingConfig(disable=True)
        thinking = _thinking_config(genai_types, thinking_level)
        if thinking is not None:
            cfg_kwargs["thinking_config"] = thinking

        if seed_history is None and seed_transcript:
            seed_history = self.history_from_transcript(genai_types, seed_transcript)

        self._chat = self._client.chats.create(
            model=model,
            config=genai_types.GenerateContentConfig(**cfg_kwargs),
            history=seed_history or None,
        )

    def get_history(self) -> list:
        return self._chat.get_history()

    # ---------------------------------------------------------- portability

    def export_transcript(self) -> List[Turn]:
        """This session's history as provider-neutral `Turn`s."""
        turns: List[Turn] = []
        for content in self._chat.get_history():
            role = "user" if getattr(content, "role", "") == "user" else "assistant"
            text_parts, calls, results = [], [], []
            for part in getattr(content, "parts", None) or []:
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    calls.append(ToolCall(id=getattr(fc, "id", "") or fc.name,
                                          name=fc.name, args=dict(fc.args or {})))
                fr = getattr(part, "function_response", None)
                if fr is not None:
                    payload = fr.response or {}
                    out = payload.get("result", payload) if isinstance(payload, dict) else payload
                    results.append((getattr(fr, "id", "") or fr.name, fr.name, str(out)))
            if results:
                # Gemini carries tool results on a `user` turn; the neutral form
                # gives them their own role so the other provider can place them
                # wherever its own protocol requires.
                turns.append(Turn(role="tool", results=results))
                if text_parts:
                    turns.append(Turn(role="user", text="\n".join(text_parts)))
            else:
                turns.append(Turn(role=role, text="\n".join(text_parts), calls=calls))
        return turns

    @staticmethod
    def history_from_transcript(genai_types, turns: List[Turn]) -> list:
        """Neutral `Turn`s back into google.genai Content objects."""
        history = []
        for t in turns or []:
            if t.role == "tool":
                parts = [genai_types.Part.from_function_response(
                    name=name, response={"result": out}) for _cid, name, out in t.results]
                history.append(genai_types.Content(role="user", parts=parts))
                continue
            parts = []
            if t.text:
                parts.append(genai_types.Part(text=t.text))
            for c in t.calls:
                parts.append(genai_types.Part.from_function_call(name=c.name, args=c.args))
            if not parts:
                continue
            role = "user" if t.role == "user" else "model"
            history.append(genai_types.Content(role=role, parts=parts))
        return history

    def send_message(self, text: str, max_retries: int = 8, base_delay: float = 3.0,
                      max_tool_rounds: int = 60) -> LLMResponse:
        if self.cost_tracker.over_budget():
            raise RuntimeError(f"Cost cap exceeded: {self.cost_tracker.summary()}")

        def _send(content):
            resp = self._chat.send_message(content)
            in_toks, out_toks = _extract_usage(resp)
            self.cost_tracker.record(self.model, in_toks, out_toks)
            return resp

        resp = _retry_on_transient(lambda: _send(text), max_retries, base_delay)

        rounds = 0
        while getattr(resp, "function_calls", None):
            if rounds >= max_tool_rounds:
                # A whole experiment -- read the apparatus, write a sequence and
                # an analysis, sweep, read results, report -- is legitimately
                # dozens of calls. Dying at the cap throws away everything done
                # so far, so say what was in flight and what to do about it.
                raise RuntimeError(
                    f"Stopped after {max_tool_rounds} tool calls without a final "
                    f"answer. Work already done (files written, routines set, "
                    f"shots queued) has NOT been undone — ask again to continue "
                    f"from where it stopped, or break the request into smaller "
                    f"steps."
                )
            if rounds and rounds % 10 == 0:
                print(f"  [{self.model}] {rounds}/{max_tool_rounds} tool calls "
                      f"this turn", flush=True)
            rounds += 1
            function_responses = []
            for call in resp.function_calls:
                if self._dispatch is None:
                    raise RuntimeError(
                        f"Model called tool '{call.name}' but this session has no dispatch callback."
                    )
                result = self._dispatch(call.name, dict(call.args or {}))
                function_responses.append(
                    self._types.Part.from_function_response(name=call.name, response={"result": result})
                )
            resp = _retry_on_transient(lambda fr=function_responses: _send(fr), max_retries, base_delay)

        text = _extract_text(resp)
        in_toks, out_toks = _extract_usage(resp)
        return LLMResponse(text=text, model=self.model, input_tokens=in_toks,
                            output_tokens=out_toks, raw=resp)


#: `thinking_level` is this project's vocabulary, chosen for Gemini. Claude
#: spends thinking adaptively and takes an `effort` hint instead, so the level
#: maps onto that rather than onto a token budget.
_CLAUDE_EFFORT = {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}

#: Room for one turn's thinking plus its answer. Tool-calling turns are short;
#: this is sized for the occasional long report, and stays under the SDK's
#: non-streaming timeout guard.
CLAUDE_MAX_TOKENS = 16000

#: Used when something asks for Claude without naming a version.
CLAUDE_DEFAULT_MODEL = "claude-opus-5"


def _block_type(block) -> str:
    """The `type` of a content block, dict or SDK object.

    Both shapes end up in the same list: blocks we build are dicts, blocks the
    model returns are appended verbatim as SDK objects.
    """
    return (block.get("type", "") if isinstance(block, dict)
            else getattr(block, "type", "")) or ""


def _drop_unanswered_calls(messages: list) -> list:
    """Make a sliced message list legal to send again.

    Compaction keeps the last N entries and a thinking-level change rebuilds the
    session, so either end of the history can land mid-exchange. The API rejects
    a `tool_use` with no matching `tool_result` *and* a `tool_result` with no
    matching `tool_use`, so both ends need trimming: unmatched results at the
    head, unanswered calls at the tail. Dropping a message whose blocks all go
    is what keeps the user/assistant alternation intact.
    """
    seen_calls: set = set()
    for msg in messages:
        for block in msg.get("content") or []:
            if isinstance(block, str):
                continue
            if _block_type(block) == "tool_use":
                seen_calls.add(block.get("id") if isinstance(block, dict)
                               else getattr(block, "id", None))

    answered: set = set()
    out: list = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str) or not content:
            out.append(msg)
            continue
        kept = []
        for block in content:
            if _block_type(block) == "tool_result":
                cid = (block.get("tool_use_id") if isinstance(block, dict)
                       else getattr(block, "tool_use_id", None))
                if cid not in seen_calls:
                    continue
                answered.add(cid)
            kept.append(block)
        if kept:
            out.append({"role": msg["role"], "content": kept})

    if out and out[-1]["role"] == "assistant" and not isinstance(out[-1]["content"], str):
        kept = [b for b in out[-1]["content"]
                if _block_type(b) != "tool_use"
                or (b.get("id") if isinstance(b, dict) else getattr(b, "id", None)) in answered]
        if kept:
            out[-1] = {"role": "assistant", "content": kept}
        else:
            out.pop()
    return out


def _with_trailing_cache_mark(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A copy of `messages` with one cache breakpoint on its last content block.

    Before this, the only `cache_control` in a Claude request sat on the system
    block (see `_create`), so the growing message history -- everything the
    conversation has said and every tool result it has read -- was reprocessed
    at full price on every single tool round. For the 14-tool-call turn that
    prompted this fix, that was the dominant cost.

    A single breakpoint here would not be enough on its own: Anthropic's cache
    lookup only scans back ~20 content blocks from a marked position to find a
    matching cached prefix, and one turn with several tool rounds can put that
    many blocks between "where the last turn's answer ended" and "where this
    turn's marker sits." Calling this on every `_create()` -- once per tool
    round, not once per `send_message` -- keeps consecutive breakpoints close
    enough together that the lookback always finds the previous one, so each
    round only pays full price for what it actually added.

    `messages` itself is never mutated: the mark is added to a shallow copy
    built fresh for the wire, so the canonical history used for export,
    compaction and provider switches stays exactly what was actually said.
    """
    if not messages:
        return messages
    last = messages[-1]
    content = last["content"]
    if isinstance(content, str):
        new_content = [{"type": "text", "text": content,
                         "cache_control": {"type": "ephemeral"}}]
    else:
        content = list(content)
        if not content:
            return messages
        block = content[-1]
        if isinstance(block, dict):
            marked = {**block, "cache_control": {"type": "ephemeral"}}
        else:
            # A block the model returned (TextBlock, ToolUseBlock, ...) is an
            # SDK object, not a dict -- dump it so cache_control can be added
            # without mutating the object `messages` still holds.
            marked = {**block.model_dump(exclude_none=True),
                      "cache_control": {"type": "ephemeral"}}
        new_content = content[:-1] + [marked]
    return messages[:-1] + [{"role": last["role"], "content": new_content}]


class ClaudeAgentSession:
    """Persistent multi-turn Claude chat with tool use and hook-gated dispatch.

    Same constructor and the same `send_message` / `get_history` contract as
    `GeminiAgentSession`, so `Agent` does not know which provider it is holding.
    Tools arrive as `ToolSpec`s rather than Gemini declarations: `spec.parameters`
    is already JSON Schema, which is what `input_schema` wants unchanged.

    The tool loop is written out rather than delegated to the SDK's tool runner.
    Every call has to pass through `dispatch`, because that is what the
    confirmation hook wraps -- a runner that executed tools itself would take
    hardware-touching calls out from behind the operator's y/N.
    """

    def __init__(
        self,
        model: str,
        system_instruction: Optional[str] = None,
        tool_specs: Optional[list] = None,
        dispatch: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        api_key: Optional[str] = None,
        cost_tracker: Optional[CostTracker] = None,
        thinking_level: str = "low",
        seed_transcript: Optional[List[Turn]] = None,
        seed_messages: Optional[list] = None,
        max_tokens: int = CLAUDE_MAX_TOKENS,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("anthropic not installed. Run: pip install anthropic") from e

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY in .env or the "
                "environment (the Gemini key is a different provider's and will "
                "not work here)."
            )
        self.model = model
        self.cost_tracker = cost_tracker or CostTracker()
        self._dispatch = dispatch
        self._system = system_instruction or ""
        self._max_tokens = int(max_tokens)
        self._effort = _CLAUDE_EFFORT.get((thinking_level or "low").lower(), "low")
        self._client = anthropic.Anthropic(
            api_key=key, timeout=REQUEST_TIMEOUT_MS / 1000.0,
        )
        self._tools = [
            {"name": s.name, "description": s.description, "input_schema": s.parameters}
            for s in (tool_specs or [])
        ]
        # `seed_messages` is this provider's own history coming back -- what
        # compaction and a thinking-level change hand over. `seed_transcript` is
        # the neutral form, used when the context arrives from the other
        # provider. Both are trimmed the same way, because either can be cut
        # between a tool call and its result.
        self._messages: List[Dict[str, Any]] = _drop_unanswered_calls(
            list(seed_messages)) if seed_messages else self.messages_from_transcript(
                seed_transcript or [])

    # ---------------------------------------------------------- portability

    def get_history(self) -> list:
        return list(self._messages)

    def export_transcript(self) -> List[Turn]:
        turns: List[Turn] = []
        for msg in self._messages:
            content = msg.get("content")
            if isinstance(content, str):
                turns.append(Turn(role=msg["role"], text=content))
                continue
            text_parts, calls, results = [], [], []
            for block in content or []:
                btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", "")
                get = (block.get if isinstance(block, dict)
                       else lambda k, d=None: getattr(block, k, d))
                if btype == "text":
                    text_parts.append(get("text") or "")
                elif btype == "tool_use":
                    calls.append(ToolCall(id=get("id") or "", name=get("name") or "",
                                          args=dict(get("input") or {})))
                elif btype == "tool_result":
                    results.append((get("tool_use_id") or "", "", str(get("content") or "")))
            if results:
                turns.append(Turn(role="tool", results=results))
            else:
                turns.append(Turn(role=msg["role"], text="\n".join(text_parts), calls=calls))
        return turns

    @staticmethod
    def messages_from_transcript(turns: List[Turn]) -> List[Dict[str, Any]]:
        """Neutral `Turn`s into Anthropic messages.

        Tool calls that never got a result are dropped along with the turn's
        other blocks: the API rejects a `tool_use` with no matching
        `tool_result`, and a transcript can end mid-exchange when a session is
        rebuilt or a model is switched between the call and its answer.
        """
        out: List[Dict[str, Any]] = []
        pending: set = set()
        for t in turns or []:
            if t.role == "tool":
                blocks = [{"type": "tool_result", "tool_use_id": cid, "content": out_text}
                          for cid, _name, out_text in t.results if cid in pending]
                if blocks:
                    out.append({"role": "user", "content": blocks})
                    pending.clear()
                continue
            if t.role == "user":
                if t.text:
                    out.append({"role": "user", "content": t.text})
                continue
            blocks = []
            if t.text:
                blocks.append({"type": "text", "text": t.text})
            for c in t.calls:
                if not c.id:
                    continue
                blocks.append({"type": "tool_use", "id": c.id, "name": c.name,
                               "input": c.args})
                pending.add(c.id)
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        if pending and out and out[-1]["role"] == "assistant":
            # Unanswered tool calls at the tail: keep any text, drop the calls.
            kept = [b for b in out[-1]["content"] if b.get("type") != "tool_use"]
            if kept:
                out[-1] = {"role": "assistant", "content": kept}
            else:
                out.pop()
        return out

    # ---------------------------------------------------------- the turn

    def _create(self):
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": _with_trailing_cache_mark(self._messages),
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self._effort},
        }
        if self._system:
            # Cached: the system instruction carries the role, the shared rules
            # and the whole memory block, and is byte-identical every turn.
            kwargs["system"] = [{"type": "text", "text": self._system,
                                 "cache_control": {"type": "ephemeral"}}]
        if self._tools:
            kwargs["tools"] = self._tools
        resp = self._client.messages.create(**kwargs)
        usage = getattr(resp, "usage", None)
        self.cost_tracker.record(
            self.model,
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_creation_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        )
        return resp

    def send_message(self, text: str, max_retries: int = 8, base_delay: float = 3.0,
                      max_tool_rounds: int = 60) -> LLMResponse:
        if self.cost_tracker.over_budget():
            raise RuntimeError(f"Cost cap exceeded: {self.cost_tracker.summary()}")

        self._messages.append({"role": "user", "content": text})
        resp = _retry_on_transient(self._create, max_retries, base_delay)

        rounds = 0
        while True:
            # Checked before the content is read: a declined request returns a
            # normal 200 with an empty or partial body, so indexing into it
            # first turns a refusal into an IndexError.
            if getattr(resp, "stop_reason", "") == "refusal":
                details = getattr(resp, "stop_details", None)
                raise RuntimeError(
                    f"{self.model} declined this request"
                    + (f" (category: {getattr(details, 'category', '?')})" if details else "")
                    + ". This is a safety classifier, not an API error; rephrasing "
                      "or switching to the other provider are the options."
                )

            self._messages.append({"role": "assistant", "content": resp.content})
            tool_uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            if not tool_uses:
                break
            if rounds >= max_tool_rounds:
                raise RuntimeError(
                    f"Stopped after {max_tool_rounds} tool calls without a final "
                    f"answer. Work already done (files written, routines set, "
                    f"shots queued) has NOT been undone — ask again to continue "
                    f"from where it stopped, or break the request into smaller "
                    f"steps."
                )
            if rounds and rounds % 10 == 0:
                print(f"  [{self.model}] {rounds}/{max_tool_rounds} tool calls "
                      f"this turn", flush=True)
            rounds += 1

            results = []
            for call in tool_uses:
                if self._dispatch is None:
                    raise RuntimeError(
                        f"Model called tool '{call.name}' but this session has no "
                        f"dispatch callback.")
                results.append({"type": "tool_result", "tool_use_id": call.id,
                                "content": str(self._dispatch(call.name,
                                                              dict(call.input or {})))})
            # All results go back in ONE user message. Splitting them teaches the
            # model to stop calling tools in parallel.
            self._messages.append({"role": "user", "content": results})
            resp = _retry_on_transient(self._create, max_retries, base_delay)

        text_out = "".join(b.text for b in resp.content
                           if getattr(b, "type", "") == "text")
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text_out, model=self.model,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            raw=resp,
        )


class ClaudeClient(LLMClient):
    """Stateless single-shot Claude calls, matching `GeminiClient.generate`.

    Not everything that talks to a model is a chat: the memory curator gets one
    prompt and returns one summary. Without this, an agent on Claude would still
    have its memory compacted by Gemini -- or, since the model name is passed
    straight through, by a Gemini client handed a Claude model id.
    """

    def __init__(
        self,
        model: str = CLAUDE_DEFAULT_MODEL,
        api_key: Optional[str] = None,
        cost_tracker: Optional[CostTracker] = None,
    ):
        super().__init__(model=model, cost_tracker=cost_tracker)
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("anthropic not installed. Run: pip install anthropic") from e
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY in .env or the "
                "environment (the Gemini key is a different provider's and will "
                "not work here)."
            )
        self._client = anthropic.Anthropic(
            api_key=key, timeout=REQUEST_TIMEOUT_MS / 1000.0,
        )

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.2,
        max_retries: int = 8,
        base_delay: float = 3.0,
    ) -> LLMResponse:
        if self.cost_tracker.over_budget():
            raise RuntimeError(f"Cost cap exceeded: {self.cost_tracker.summary()}")

        # `temperature` is accepted and ignored: it is part of the shared
        # signature but a 400 on this API.
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": int(max_output_tokens or CLAUDE_MAX_TOKENS),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        def _call():
            return self._client.messages.create(**kwargs)

        resp = _retry_on_transient(_call, max_retries, base_delay)
        if getattr(resp, "stop_reason", "") == "refusal":
            raise RuntimeError(f"{self.model} declined this request.")
        usage = getattr(resp, "usage", None)
        self.cost_tracker.record(
            self.model,
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
        )
        return LLMResponse(
            text="".join(b.text for b in resp.content
                         if getattr(b, "type", "") == "text"),
            model=self.model,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            raw=resp,
        )


def is_claude(model: str) -> bool:
    return "claude" in (model or "").lower()


def make_session(
    model: str,
    system_instruction: Optional[str] = None,
    tool_specs: Optional[list] = None,
    tool_declarations: Optional[list] = None,
    dispatch: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    cost_tracker: Optional[CostTracker] = None,
    temperature: float = 0.2,
    thinking_level: str = "low",
    seed_history: Optional[list] = None,
    seed_transcript: Optional[List[Turn]] = None,
):
    """Build a chat session for `model`, picking the provider from its name.

    The caller passes both tool forms because it holds a registry that can
    produce either; each session takes the one its provider speaks. Nothing else
    about the call differs, which is the point -- the same system instruction,
    the same tool whitelist, the same skills, the same context.
    """
    if is_claude(model):
        return ClaudeAgentSession(
            model=model, system_instruction=system_instruction,
            tool_specs=tool_specs, dispatch=dispatch, cost_tracker=cost_tracker,
            thinking_level=thinking_level, seed_transcript=seed_transcript,
            seed_messages=seed_history,
        )
    return GeminiAgentSession(
        model=model, system_instruction=system_instruction,
        tool_declarations=tool_declarations, dispatch=dispatch,
        cost_tracker=cost_tracker, temperature=temperature,
        thinking_level=thinking_level, seed_history=seed_history,
        seed_transcript=seed_transcript,
    )


def make_client(model: Optional[str] = None,
                cost_tracker: Optional[CostTracker] = None) -> LLMClient:
    """A single-shot client for `model`, picking the provider from its name."""
    model = model or DEFAULT_MODEL
    if is_claude(model):
        return ClaudeClient(model=model, cost_tracker=cost_tracker)
    return GeminiClient(model=model, cost_tracker=cost_tracker)
