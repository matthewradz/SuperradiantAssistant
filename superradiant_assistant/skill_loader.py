"""SKILL.md discovery and two-tier loading.

Only names and one-line descriptions go into the system prompt; the full body is
fetched on demand through the `load_skill` tool. Preloading every procedure would
crowd the context with instructions irrelevant to the task at hand.

A SKILL.md is YAML frontmatter (name, description, optional tags) followed by
free-form Markdown.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Optional

SKILLS_DIR = Path(__file__).resolve().parent / "skills"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    # A BOM makes the text start with ﻿ instead of '---', so the whole
    # frontmatter silently fails to match and every skill loses its description.
    # Windows editors and PowerShell's Set-Content add one by default, so any
    # hand-edited SKILL.md can arrive this way.
    text = text.lstrip("﻿").replace("\r\n", "\n")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw_meta, body = m.group(1), m.group(2)
    meta: dict = {}
    for line in raw_meta.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip().strip("'\"")
    return meta, body.strip()


class SkillLoader:
    def __init__(self, skills_dir: Optional[Path] = None):
        self.skills_dir = Path(skills_dir or SKILLS_DIR)
        self.skills: Dict[str, dict] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.skills_dir.exists():
            return
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            try:
                meta, body = _parse_frontmatter(f.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[skills] could not read {f}: {e}")
                continue
            name = meta.get("name") or f.parent.name
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    def names(self) -> List[str]:
        return sorted(self.skills)

    #: The apparatus this session actually controls. Skills declaring a different
    #: one are still available -- the method in them is often what is wanted --
    #: but they are marked wherever they appear.
    #:
    #: Read live rather than captured: this was hardcoded to "cesium", so a
    #: testbench session marked every cesium skill as usable and would have
    #: marked the bench's own skills "NOT this bench". `--apparatus` and the boot
    #: screen can both change it after this module is imported, which is why it
    #: is a property and not a constant.
    @property
    def CURRENT_APPARATUS(self) -> str:
        from superradiant_assistant import config
        return config.APPARATUS

    def apparatus_of(self, skill_name: str) -> str:
        s = self.skills.get(skill_name) or {}
        return (s.get("meta") or {}).get("apparatus", "").strip()

    def get_descriptions(self) -> str:
        """The menu injected into system prompts — names and one-liners only.

        Skills for another apparatus are labelled here rather than hidden. A
        skill written for a different bench may name globals, metrics and
        hardware this one does not have, so its procedure cannot run at all.
        Unlabelled, the menu reads as a list of things this agent can do, and it
        was read that way — asked about the lab, it once described a foreign
        apparatus as the one in front of it.
        """
        if not self.skills:
            return "(no skills installed)"
        lines = []
        for name, s in sorted(self.skills.items()):
            desc = s["meta"].get("description", "(no description)")
            app = (s["meta"].get("apparatus") or "").strip()
            tag = (f"[{app}, NOT this bench] "
                   if app and app != self.CURRENT_APPARATUS else "")
            lines.append(f"  - {name}: {tag}{desc}")
        return "\n".join(lines)

    def get_content(self, skill_name: str) -> str:
        s = self.skills.get(skill_name)
        if s is None:
            return f"error: unknown skill '{skill_name}'. Available: {self.names()}"
        app = (s["meta"].get("apparatus") or "").strip()
        warn = ""
        if app and app != self.CURRENT_APPARATUS:
            warn = (f"\n\nNOTE: this procedure belongs to the {app} apparatus, "
                    f"which is NOT the bench in front of you. Its sequences, "
                    f"globals and metrics do not exist here. Read it for method, "
                    f"and do not tell the operator you can run it.")
        return f'<skill name="{skill_name}">\n{s["body"]}{warn}\n</skill>'


SKILLS = SkillLoader()
