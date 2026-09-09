"""Configuration for the Superradiant Assistant wrapper.

Loads from config.json at repo root, then falls back to environment variables.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Auto-load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

#: Which apparatus this session is for. `config.json` is the default; any
#: `config.<name>.json` beside it is another apparatus.
#:
#: These are not interchangeable, and that is the whole reason for the switch.
#: The bench has no atoms and no cavity; its globals cap `amplitude` at 1.0 V and
#: its memory says so. Carrying that into a real lab would have the assistant
#: reasoning from facts about a function generator.
DEFAULT_APPARATUS = "testbench"


def apparatus_config_path(name: str) -> Path:
    """The config file for `name`, whether or not it exists."""
    if not name or name == DEFAULT_APPARATUS:
        return REPO_ROOT / "config.json"
    return REPO_ROOT / f"config.{name}.json"


def available_apparatus() -> List[str]:
    """Every apparatus that has a config file, the default first."""
    found = [p.stem.split(".", 1)[1] for p in sorted(REPO_ROOT.glob("config.*.json"))
             if p.stem.split(".", 1)[1:]]
    base = [DEFAULT_APPARATUS] if (REPO_ROOT / "config.json").exists() else []
    return base + [n for n in found if n]


def _load_cfg_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[config] Warning: could not parse {path.name}: {e}")
        return {}


APPARATUS: str = os.environ.get("AGENT_APPARATUS") or DEFAULT_APPARATUS
_cfg_path = apparatus_config_path(APPARATUS)
_cfg_json: dict = _load_cfg_json(_cfg_path)


def _get(key: str, default=None):
    """Read from the apparatus config first, then env var, then default."""
    return _cfg_json.get(key) or os.environ.get(key) or default


def _suite_root(raw: str) -> Path:
    """Find the labscript-suite installation.

    Unlike the data folders this one is NOT beside the repo -- it is a separate
    installation that can be anywhere -- so it cannot simply be derived from
    REPO_ROOT. But it does not have to be hard-coded either: labscript publishes
    its location in LABSCRIPT_SUITE_PROFILE and installs to ~/labscript-suite by
    default, so an explicit setting that has gone stale can fall back to those
    rather than leaving the agent pointing at another machine's home directory.
    """
    candidates = []
    if raw:
        candidates.append(Path(raw))
    env_profile = os.environ.get("LABSCRIPT_SUITE_PROFILE")
    if env_profile:
        candidates.append(Path(env_profile))
    candidates.append(Path.home() / "labscript-suite")

    for p in candidates:
        if p.exists():
            if raw and str(p) != str(Path(raw)):
                print(f"[config] LABSCRIPT_SUITE_PATH {raw} not found; using {p}")
            return p
    return Path(raw) if raw else candidates[-1]


def _data_root(raw: str) -> Path:
    """Resolve the data root, surviving a rename of the project folder.

    The lab data, memory and reports live in sibling folders next to the repo,
    and config.json records that location as an absolute path. Renaming or
    copying the project therefore silently broke it: the folder moved, the path
    in the file did not, and the agent started up pointing at a directory that
    no longer existed -- no shots, no memory, no reports, and no error saying so.

    So: a relative path is resolved against the repo's parent (which is what it
    should have been all along), and an absolute path that has ceased to exist
    falls back to a same-named folder beside the repo before it is trusted.
    """
    p = Path(raw)
    if not p.is_absolute():
        return (REPO_ROOT.parent / p).resolve()
    if p.exists():
        return p
    beside = REPO_ROOT.parent / p.name
    if beside.exists():
        print(f"[config] {p} does not exist; using {beside} instead "
              f"(the project folder looks renamed -- update config.json)")
        return beside
    return p


@dataclass
class KnowledgeSource:
    path: Path
    kind: str
    recursive: bool = True
    glob: str = "*.py"
    include: Tuple[str, ...] = ()
    exclude: Tuple[str, ...] = ()


@dataclass
class Config:
    labscript_suite_root: Path = field(
        default_factory=lambda: _suite_root(_get("LABSCRIPT_SUITE_PATH", ""))
    )
    historical_data_root: Path = field(
        # The last resort is a folder beside this repo, never an absolute path
        # on whoever's machine wrote this file. An unparseable config.json falls
        # back to these defaults silently, and a stranger's absolute path turns
        # that into "no shots found" with nothing pointing at the real mistake.
        default_factory=lambda: _data_root(
            _get("MEMORY_DATA_PATH") or
            _get("HISTORICAL_DATA_PATH", "local_data")
        )
    )
    mock_mode: bool = field(
        default_factory=lambda: str(_get("MOCK_MODE", "true")).lower() != "false"
    )
    gemini_api_key: str = field(
        default_factory=lambda: _get("GEMINI_API_KEY", "")
    )
    anthropic_api_key: str = field(
        default_factory=lambda: _get("ANTHROPIC_API_KEY", "")
    )
    max_dollars_per_run: float = field(
        default_factory=lambda: float(_get("MAX_DOLLARS_PER_RUN", 100.0))
    )
    max_tokens_per_run: int = field(
        default_factory=lambda: int(_get("MAX_TOKENS_PER_RUN", 1_000_000))
    )
    default_params_file: str = field(
        default_factory=lambda: _get("DEFAULT_PARAMS_FILE", "examples/params.txt")
    )
    exec_import: str = field(
        default_factory=lambda: _get(
            "EXEC_IMPORT",
            "import numpy as np\nimport h5py\nimport pandas as pd\n"
        )
    )

    # Experiment-specific lists loaded from config.json
    sequences: list = field(
        default_factory=lambda: _cfg_json.get("sequences", [])
    )
    analysis_scripts: list = field(
        default_factory=lambda: _cfg_json.get("analysis_scripts", [])
    )
    experiment_globals: list = field(
        default_factory=lambda: _cfg_json.get("globals", [])
    )
    # Optional: metric names this apparatus produces. Leave empty to let them be
    # discovered from what lyse actually saved into recent shots.
    metrics: list = field(
        default_factory=lambda: _cfg_json.get("metrics", [])
    )

    knowledge_root: Path = REPO_ROOT / "superradiant_assistant" / "knowledge"
    knowledge_cache: Path = REPO_ROOT / "superradiant_assistant" / "knowledge" / ".cache"
    embedding_model: str = "text-embedding-004"
    embedding_max_chars_per_doc: int = 8000

    @property
    def labscriptlib_root(self) -> Path:
        return self.labscript_suite_root / "userlib" / "labscriptlib"

    @property
    def ybclock_root(self) -> Path:
        return self.labscriptlib_root / "ybclock"

    @property
    def user_devices_root(self) -> Path:
        return self.labscript_suite_root / "userlib" / "user_devices"

    @property
    def analysis_scripts_root(self) -> Path:
        return self.ybclock_root / "analysis" / "scripts"

    @property
    def sequences_root(self) -> Path:
        return self.ybclock_root / "sequences"

    @property
    def subsequences_root(self) -> Path:
        return self.ybclock_root / "subsequences"

    @property
    def classes_root(self) -> Path:
        return self.ybclock_root / "classes"

    @property
    def connection_functions_root(self) -> Path:
        return self.ybclock_root / "connection_functions"

    def knowledge_sources(self) -> List[KnowledgeSource]:
        return [
            KnowledgeSource(
                path=self.knowledge_root / "handwritten",
                kind="manual", glob="*.md",
            ),
            KnowledgeSource(
                path=self.knowledge_root / "code_examples",
                kind="code_example", glob="*.md",
            ),
            KnowledgeSource(
                path=self.user_devices_root,
                kind="device", glob="*.py",
                exclude=("__init__.py", "__pycache__"),
            ),
            KnowledgeSource(
                path=self.classes_root,
                kind="class", glob="*.py",
                exclude=("__init__.py", "__pycache__"),
            ),
            KnowledgeSource(
                path=self.connection_functions_root,
                kind="connection", glob="*.py",
                exclude=("__init__.py", "__pycache__"),
            ),
            KnowledgeSource(
                path=self.sequences_root / "cooling",
                kind="sequence", glob="*.py",
                include=(
                    "recycling_on_clock_transition_release_recap_FPGA_yellow_FNC.py",
                ),
            ),
            KnowledgeSource(
                path=self.sequences_root / "assistant_sequences",
                kind="sequence", glob="*.py",
                exclude=("__init__.py",),
            ),
            KnowledgeSource(
                path=self.subsequences_root,
                kind="subsequence", glob="*.py",
                exclude=("__init__.py", "__pycache__", "dummy_filename.py"),
            ),
            KnowledgeSource(
                path=self.analysis_scripts_root,
                kind="analysis", glob="*.py",
                include=(
                    "cavity_scan_analysis.py",
                    "cavity_photon_count_analysis.py",
                    "extract_photon_arrival_times.py",
                    "recycling_single_shot_analysis.py",
                    "measure_sz_over_s.py",
                    "get_expected_sz.py",
                    "calculate_squeezing_photon_imbalance.py",
                ),
            ),
            KnowledgeSource(
                path=self.analysis_scripts_root / "meta",
                kind="analysis", glob="*.py",
                include=(
                    "improved_cost_clean.py",
                    "calibrate_larmor_frequency_clean.py",
                ),
            ),
        ]


CONFIG = Config()


def select_apparatus(name: str) -> str:
    """Switch the whole session to another apparatus, in place.

    Rebuilt rather than re-imported because `CONFIG` is a module-level singleton
    that other modules have already bound. Mutating its fields is the only way to
    change apparatus after import without every caller re-fetching it.

    Everything apparatus-specific follows from here: the globals and their limits,
    the sequence list, the data root -- and therefore the memory directory, which
    is derived from the data root. Returns the name actually selected.
    """
    global APPARATUS, _cfg_path, _cfg_json
    name = name or DEFAULT_APPARATUS
    path = apparatus_config_path(name)
    if not path.exists():
        print(f"[config] no {path.name} — staying on {APPARATUS!r}")
        return APPARATUS

    APPARATUS = name
    os.environ["AGENT_APPARATUS"] = name          # for subprocesses
    _cfg_path = path
    _cfg_json = _load_cfg_json(path)

    fresh = Config()
    for f in fields(Config):
        setattr(CONFIG, f.name, getattr(fresh, f.name))

    # Anything that cached a path derived from CONFIG has to be told. The memory
    # store is the one that matters: its directory comes from the data root, and
    # leaving it stale meant the new lab's name was shown above the old lab's
    # facts. Imported here rather than at module scope because memory imports
    # config.
    try:
        from superradiant_assistant.memory.store import MEMORY
        MEMORY.reconfigure()
    except Exception as e:                          # never block a switch
        print(f"[config] memory did not follow the switch: {type(e).__name__}: {e}")
    return name