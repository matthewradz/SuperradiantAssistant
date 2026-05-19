"""LLM client abstraction.

Uniform interface so the rest of the codebase doesn't care which provider
is in use. First impl: Google Gemini via the new google-genai SDK.
"""
from __future__ import annotations
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# Auto-load .env if present (no error if dotenv isn't installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from superradiant_assistant.llm.cost_tracker import CostTracker


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Optional[Any] = None


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

    def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        joined = ""
        for m in messages:
            role = m.get("role", "user").upper()
            joined += f"\n[{role}]\n{m.get('content','')}\n"
        return self.generate(
            prompt=joined.strip(),
            system=system,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )


class GeminiClient(LLMClient):
    """Gemini via google-genai (the current, supported SDK)."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
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

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEY or pass api_key=..."
            )
        self._client = genai.Client(api_key=key)
        self._types = genai_types

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.2,
        max_retries: int = 4,
        base_delay: float = 5.0,
    ) -> LLMResponse:
        if self.cost_tracker.over_budget():
            raise RuntimeError(f"Cost cap exceeded: {self.cost_tracker.summary()}")

        cfg_kwargs: Dict[str, Any] = {"temperature": float(temperature)}
        if system:
            cfg_kwargs["system_instruction"] = system
        if max_output_tokens is not None:
            cfg_kwargs["max_output_tokens"] = int(max_output_tokens)

        config = self._types.GenerateContentConfig(**cfg_kwargs)

        last_exc: Exception = RuntimeError("no attempts made")
        for attempt in range(max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )

                text = ""
                try:
                    text = resp.text or ""
                except Exception:
                    try:
                        text = resp.candidates[0].content.parts[0].text
                    except Exception:
                        text = ""

                in_toks = out_toks = 0
                meta = getattr(resp, "usage_metadata", None)
                if meta is not None:
                    in_toks = int(getattr(meta, "prompt_token_count", 0) or 0)
                    out_toks = int(getattr(meta, "candidates_token_count", 0) or 0)

                self.cost_tracker.record(self.model, in_toks, out_toks)

                return LLMResponse(
                    text=text,
                    model=self.model,
                    input_tokens=in_toks,
                    output_tokens=out_toks,
                    raw=resp,
                )

            except Exception as e:
                last_exc = e
                err_str = str(e)
                # retry on transient server errors
                if any(code in err_str for code in ("503", "429", "500", "UNAVAILABLE", "RATE_LIMIT")):
                    delay = base_delay * (2 ** attempt)
                    print(f"  [LLM] transient error (attempt {attempt+1}/{max_retries}), "
                        f"retrying in {delay:.0f}s: {err_str[:80]}")
                    import time
                    time.sleep(delay)
                    continue
                # non-retryable error
                raise

        raise RuntimeError(
            f"LLM call failed after {max_retries} attempts. Last error: {last_exc}"
        )

    


def make_client(
    provider: str = "gemini",
    model: Optional[str] = None,
    cost_tracker: Optional[CostTracker] = None,
) -> LLMClient:
    provider = provider.lower()
    if provider == "gemini":
        return GeminiClient(
            model=model or "gemini-2.5-flash",
            cost_tracker=cost_tracker,
        )
    raise ValueError(f"Unknown provider: {provider}")