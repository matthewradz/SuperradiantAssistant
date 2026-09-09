"""Smoke test for the Gemini LLM client."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superradiant_assistant.llm.client import make_client


def main():
    client = make_client()
    resp = client.generate(
        prompt="Say 'hello from Gemini' and nothing else.",
        temperature=0.0,
        # leave max_output_tokens unset; Flash 2.5 needs headroom for thinking
    )
    print("=== Response ===")
    print(repr(resp.text))
    print("\n=== Usage ===")
    print(f"  model:         {resp.model}")
    print(f"  input tokens:  {resp.input_tokens}")
    print(f"  output tokens: {resp.output_tokens}")
    print("\n=== Cost tracker ===")
    print(client.cost_tracker.summary())


if __name__ == "__main__":
    main()