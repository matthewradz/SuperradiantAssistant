"""List what the knowledge-base loader picks up."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superradiant_assistant.knowledge.loader import (
    load_knowledge_base,
    summarize_knowledge_base,
)


def main():
    docs = load_knowledge_base()
    print("Summary:", summarize_knowledge_base(docs))
    print()
    print(f"{'kind':<14}{'title':<60}chars")
    print("-" * 90)
    for d in docs:
        print(f"{d.kind:<14}{d.title[:58]:<60}{d.char_count}")


if __name__ == "__main__":
    main()