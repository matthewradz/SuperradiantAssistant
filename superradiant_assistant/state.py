"""Shared experiment state — analog of HAL's STATE dictionary."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List


class ExperimentState:
    def __init__(self) -> None:
        self._data: Dict[str, Any] = {
            "phase": "idle",
            "iteration": 0,
            "history": [],
            "best_value": None,
            "best_params": None,
        }

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def append_history(self, entry: Dict[str, Any]) -> None:
        self._data["history"].append(entry)

    def history(self) -> List[Dict[str, Any]]:
        return list(self._data["history"])

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self._data, indent=2, default=str))


STATE = ExperimentState()