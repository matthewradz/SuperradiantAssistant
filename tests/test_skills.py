"""SkillLoader tests: discovery, frontmatter parsing, and two-tier loading."""
from __future__ import annotations

from superradiant_assistant.skill_loader import SkillLoader, SKILLS, _parse_frontmatter


class TestFrontmatter:
    def test_parses_metadata_and_body(self):
        meta, body = _parse_frontmatter(
            "---\nname: demo\ndescription: A demo skill.\ntags: a, b\n---\n\n# Title\ntext\n"
        )
        assert meta == {"name": "demo", "description": "A demo skill.", "tags": "a, b"}
        assert body.startswith("# Title")

    def test_file_without_frontmatter_keeps_whole_body(self):
        meta, body = _parse_frontmatter("# Just markdown\n")
        assert meta == {}
        assert "Just markdown" in body

    def test_quotes_stripped_from_values(self):
        meta, _ = _parse_frontmatter("---\nname: 'quoted'\n---\nbody\n")
        assert meta["name"] == "quoted"


class TestDiscovery:
    def test_finds_shipped_skills(self):
        names = SKILLS.names()
        for expected in ("resonance-scan", "larmor-calibration",
                         "atom-loading-optimization", "hardware-safety",
                         "lab-knowledge-search"):
            assert expected in names

    def test_every_skill_declares_a_description(self):
        for name, skill in SKILLS.skills.items():
            assert skill["meta"].get("description"), f"{name} has no description"

    def test_missing_directory_is_not_an_error(self, tmp_path):
        loader = SkillLoader(tmp_path / "nope")
        assert loader.names() == []
        assert loader.get_descriptions() == "(no skills installed)"


class TestTwoTierLoading:
    def test_menu_lists_names_without_bodies(self):
        menu = SKILLS.get_descriptions()
        assert "resonance-scan" in menu
        # The menu goes in every system prompt, so it must stay short — body text
        # is only fetched via load_skill.
        assert "delta_duration" not in menu
        assert len(menu.splitlines()) == len(SKILLS.names())

    def test_get_content_returns_wrapped_body(self):
        content = SKILLS.get_content("resonance-scan")
        assert content.startswith('<skill name="resonance-scan">')
        assert content.endswith("</skill>")
        assert "delta_duration" in content

    def test_unknown_skill_reports_available_names(self):
        r = SKILLS.get_content("no-such-skill")
        assert "unknown skill" in r
        assert "resonance-scan" in r

    def test_body_matches_file_on_disk(self, tmp_path):
        d = tmp_path / "skills" / "demo"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            "---\nname: demo\ndescription: d\n---\n\nbody line\n", encoding="utf-8"
        )
        loader = SkillLoader(tmp_path / "skills")
        assert "body line" in loader.get_content("demo")
