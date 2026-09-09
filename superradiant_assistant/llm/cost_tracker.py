"""Tracks LLM token usage and dollar cost across a run."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


# Dollars per 1M tokens. Update as provider pricing changes.
#
# A model missing from this table is still counted in tokens but costs $0, so an
# entry per model actually used is what keeps the dollar figure honest.
PRICE_TABLE: Dict[str, Dict[str, float]] = {
    "gemini-2.0-flash":                  {"input": 0.10,  "output": 0.40},
    "gemini-2.5-flash":                  {"input": 0.30,  "output": 2.50},
    "gemini-2.5-pro":                    {"input": 1.25,  "output": 10.00},
    "gemini-3.6-flash":                  {"input": 0.30,  "output": 2.50},
    "claude-opus-5":                     {"input": 5.00,  "output": 25.00},
    "claude-sonnet-5":                   {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5":                  {"input": 1.00,  "output": 5.00},
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    #: Tokens served from a prompt cache (~0.1x the price of a fresh input
    #: token) and tokens written to one (~1.25x, at the 5-minute TTL this
    #: project uses). Anthropic reports both separately from `input_tokens`,
    #: which counts only the uncached remainder -- summing just input+output
    #: would silently drop however much of a turn was actually cached.
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    calls: int = 0


@dataclass
class CostTracker:
    max_tokens: int = 1_000_000
    max_dollars: float = 100.0
    use_free_tier_pricing: bool = True   # report $0 if True
    _by_model: Dict[str, Usage] = field(default_factory=dict)

    def record(self, model: str, input_tokens: int, output_tokens: int,
               cache_read_tokens: int = 0, cache_creation_tokens: int = 0) -> None:
        u = self._by_model.setdefault(model, Usage())
        u.input_tokens += int(input_tokens)
        u.output_tokens += int(output_tokens)
        u.cache_read_tokens += int(cache_read_tokens)
        u.cache_creation_tokens += int(cache_creation_tokens)
        u.calls += 1

    @property
    def total_tokens(self) -> int:
        return sum(u.input_tokens + u.output_tokens + u.cache_read_tokens
                   + u.cache_creation_tokens for u in self._by_model.values())

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
            # Anthropic's published 5-minute-TTL multipliers: a cache read is
            # ~10% of a fresh input token, a cache write ~125%.
            total += (u.cache_read_tokens / 1_000_000) * price["input"] * 0.1
            total += (u.cache_creation_tokens / 1_000_000) * price["input"] * 1.25
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
                    "cache_read_tokens": u.cache_read_tokens,
                    "cache_creation_tokens": u.cache_creation_tokens,
                }
                for m, u in self._by_model.items()
            },
        }