"""Progress messages of scRICH.

scRICH reports progress through the standard :mod:`logging` module (logger name ``"scrich"``).
Messages are printed to standard output, so they appear in notebooks like ``print`` output.
Use :func:`set_verbosity` to show fewer or more messages::

    import scrich as ri
    ri.set_verbosity("warning")   # only warnings and errors
    ri.set_verbosity("info")      # default: progress messages
    ri.set_verbosity("debug")     # detailed messages (e.g. every clustering run)
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("scrich")

_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}


class _StdoutHandler(logging.StreamHandler):
    """Write to the *current* ``sys.stdout`` (notebooks replace it after import)."""

    def __init__(self):
        super().__init__(stream=sys.stdout)

    def emit(self, record):
        self.stream = sys.stdout
        super().emit(record)


if not logger.handlers:
    _handler = _StdoutHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def set_verbosity(level: str | int = "info") -> None:
    """Set how many progress messages scRICH prints.

    Parameters
    ----------
    level
        ``"debug"``, ``"info"`` (default), ``"warning"`` or ``"error"``, or a :mod:`logging` level.
    """
    if isinstance(level, str):
        if level.lower() not in _LEVELS:
            raise ValueError(f"Unknown verbosity {level!r}. Choose one of {list(_LEVELS)}.")
        level = _LEVELS[level.lower()]
    logger.setLevel(level)
