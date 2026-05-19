"""Tracks LLM token usage and dollar cost across a run."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


# Dollars per 1M tokens. Update as providers change pricing.
PRICE_TABLE: Dict[str, Dict[str, float]] = {
    # Gemini 2.5 Flash (paid tier; free tier is $0)
    "gemini-2.5-flash":     {"input": 0.30, "output": 2.50},
    "gemini-2.0-flash":     {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro":       {"input": 1.25, "output": 10.00},
    # Anthropic
    "claude-3-5-sonnet":    {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku":     {"input": 0.80, "output": 4.00},
    # OpenAI
    "gpt-4o-mini":          {"input": 0.15, "output": 0.60},
    "gpt-4o":               {"input": 2.50, "output": 10.00},
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