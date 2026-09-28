"""Lazy imports for optional dependencies.

Optional packages (e.g. ``scvelo``, ``d3blocks``) are only imported when a
function that needs them is called, so ``import scrich`` works without them.
"""

from __future__ import annotations

import importlib
from types import ModuleType


class LazyModule:
    """Proxy that imports ``name`` on first attribute access.

    Parameters
    ----------
    name
        Module to import, e.g. ``"scvelo"``.
    extra
        Name of the ``scrich`` extra that installs it, used in the error message.
    """

    def __init__(self, name: str, extra: str) -> None:
        self._name, self._extra, self._mod = name, extra, None

    def _load(self) -> ModuleType:
        if self._mod is None:
            try:
                self._mod = importlib.import_module(self._name)
            except ImportError as e:
                raise ImportError(
                    f"This function requires the optional dependency {self._name!r}. "
                    f"Install it with: pip install 'scrich[{self._extra}]'"
                ) from e
        return self._mod

    def __getattr__(self, attr: str):
        return getattr(self._load(), attr)
