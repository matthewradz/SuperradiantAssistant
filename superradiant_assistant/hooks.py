"""Lightweight hook manager so labs can plug in safety / confirmation logic."""
from __future__ import annotations
from typing import Callable, Dict, List, Any


class HookManager:
    def __init__(self) -> None:
        self._hooks: Dict[str, List[Callable[..., Any]]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self._hooks.setdefault(name, []).append(fn)

    def run(self, name: str, **kwargs: Any) -> List[Any]:
        return [fn(**kwargs) for fn in self._hooks.get(name, [])]

    def any_true(self, name: str, **kwargs: Any) -> bool:
        return any(bool(r) for r in self.run(name, **kwargs))

    def all_true(self, name: str, **kwargs: Any) -> bool:
        results = self.run(name, **kwargs)
        if not results:
            return True   # no hooks registered → don't block
        return all(bool(r) for r in results)