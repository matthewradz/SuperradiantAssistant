"""Tracks LLM token usage and dollar cost across a run."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


# Dollars per 1M tokens. Update as providers change pricing.
PRICE_TABLE: Dict[str, Dict[str, float]] = {
    # Gemini (direct)
    "gemini-2.5-flash":                 {"input": 0.30,  "output": 2.50},
    "gemini-2.0-flash":                 {"input": 0.10,  "output": 0.40},
    "gemini-2.5-pro":                   {"input": 1.25,  "output": 10.00},
    # Parley — Claude via AWS Bedrock
    "bedrock/claude-haiku-4-5":         {"input": 1.00,  "output": 5.00},
    "bedrock/claude-sonnet-4-6":        {"input": 3.00,  "output": 15.00},
    "bedrock/claude-opus-4-6":          {"input": 5.00,  "output": 25.00},
    "bedrock/claude-opus-4-7":          {"input": 5.00,  "output": 25.00},
    # Parley — OpenAI
    "openai/gpt-5-nano":                {"input": 0.10,  "output": 0.50},
    "openai/gpt-5-mini":                {"input": 0.25,  "output": 2.00},
    "openai/gpt-5":                     {"input": 1.25,  "output": 10.00},
    "openai/gpt-5.4":                   {"input": 2.50,  "output": 15.00},
    "openai/gpt-5.5":                   {"input": 5.00,  "output": 30.00},
    # Parley — Google
    "google/gemini-3.0-flash":          {"input": 0.50,  "output": 3.00},
    "google/gemini-2.5-pro":            {"input": 2.50,  "output": 15.00},
    "google/gemini-3.1-pro":            {"input": 4.00,  "output": 18.00},
    # Parley — Meta (free)
    "meta/llama-4-maverick":            {"input": 0.00,  "output": 0.00},
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


@dataclass
class CostTracker:
    max_tokens: int = 1_000_000
    max_dollars: float = 100.0
    use_free_tier_pricing: bool = True   # report $0 if True
    _by_model: Dict[str, Usage] = field(default_factory=dict)

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        u = self._by_model.setdefault(model, Usage())
        u.input_tokens += int(input_tokens)
        u.output_tokens += int(output_tokens)
        u.calls += 1

    @property
    def total_tokens(self) -> int:
        return sum(u.input_tokens + u.output_tokens for u in self._by_model.values())

    def dollars(self) -> float:
        if self.use_free_tier_pricing:
            return 0.0
        total = 0.0
        for model, u in self._by_model.items():
            price = PRICE_TABLE.get(model)
            if not price:
                continue
            total += (u.input_tokens / 1_000_000) * price["input"]
            total += (u.output_tokens / 1_000_000) * price["output"]
        return total

    def over_budget(self) -> bool:
        return self.total_tokens > self.max_tokens or self.dollars() > self.max_dollars

    def summary(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "estimated_dollars": round(self.dollars(), 4),
            "by_model": {
                m: {
                    "calls": u.calls,
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                }
                for m, u in self._by_model.items()
            },
        }