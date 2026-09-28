"""scRICH: robust identification of cell-cell communication heterogeneity in single cells.

The package follows the scanpy convention:

- :mod:`scrich.tl` -- analysis tools (clustering of communication modes, CCC matrices, ...)
- :mod:`scrich.pl` -- plotting functions
- :mod:`scrich.db` -- access to the ligand-receptor and signaling databases

Developed by Federico Bocci, Radboud University.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from . import db, pl, tl

try:
    __version__ = _version("scrich")
except PackageNotFoundError:  # running from a source checkout without installation
    __version__ = "0.0.0"

__all__ = ["tl", "pl", "db", "__version__"]
