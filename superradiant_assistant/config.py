"""Configuration for the Superradiant Assistant wrapper.

Loads from config.json at repo root, then falls back to environment variables.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Auto-load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Load config.json if present
_cfg_path = REPO_ROOT / "config.json"
_cfg_json: dict = {}
if _cfg_path.exists():
    try:
        _cfg_json = json.loads(_cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[config] Warning: could not parse config.json: {e}")


def _get(key: str, default=None):
    """Read from config.json first, then env var, then default."""
    return _cfg_json.get(key) or os.environ.get(key) or default


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
        default_factory=lambda: Path(
            _get("LABSCRIPT_SUITE_PATH",
                 r"C:\Users\radzi\Documents\labscript-suite\labscript-suite")
        )
    )
    historical_data_root: Path = field(
        default_factory=lambda: Path(
            _get("MEMORY_DATA_PATH") or
            _get("HISTORICAL_DATA_PATH",
                 r"C:\Users\radzi\Documents\data_2026_05_18")
        )
    )
    mock_mode: bool = field(
        default_factory=lambda: str(_get("MOCK_MODE", "true")).lower() != "false"
    )
    gemini_api_key: str = field(
        default_factory=lambda: _get("GEMINI_API_KEY", "")
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