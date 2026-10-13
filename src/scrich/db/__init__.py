"""Access to the ligand-receptor and signaling databases used by scRICH.

There are two kinds of database files:

**Bundled tables** (small, shipped inside the package)
    Pathway -> ligand and pathway -> receptor tables derived from CellChatDB,
    loaded with :func:`load_pathway_genes`.

**External tables** (large, downloaded on first use)
    exFinder layers (ligand-receptor, receptor-TF, TF-target) and the CellChat
    interaction table, loaded with :func:`load_exfinder`. Files are downloaded
    once from Zenodo (DOI `10.5281/zenodo.23319307
    <https://doi.org/10.5281/zenodo.23319307>`_), verified against a SHA256
    checksum and cached on disk.

Where external files are stored
-------------------------------
By default files are cached in the operating system's cache folder
(e.g. ``~/Library/Caches/scrich`` on macOS, ``~/.cache/scrich`` on Linux).
To use another folder, for example a shared folder on a cluster without
internet access, either set the environment variable::

    export SCRICH_DATA_DIR=/path/to/scrich_db

or call :func:`set_data_dir` at the start of a session. If the folder already
contains the files (with matching checksums), nothing is downloaded.

Examples
--------
>>> import scrich as ri
>>> lig = ri.db.load_pathway_genes("ligand", species="mouse")
>>> rtf = ri.db.load_exfinder("rtf", species="human")  # doctest: +SKIP
"""

from __future__ import annotations

import os
from functools import cache, lru_cache
from importlib import resources
from pathlib import Path
from typing import Literal

import pandas as pd
import pooch

from ._registry import DB_VERSION, REGISTRY, ZENODO_RECORD

__all__ = [
    "SPECIES",
    "EXFINDER_LAYERS",
    "check_species",
    "load_pathway_genes",
    "load_exfinder",
    "fetch_all",
    "data_dir",
    "set_data_dir",
    "clear_memory_cache",
]

#: Species supported by the bundled and external databases.
SPECIES: tuple[str, ...] = ("human", "mouse")

#: Short layer name -> file stem of the external exFinder / CellChat tables.
EXFINDER_LAYERS: dict[str, str] = {
    "cellchat": "cellchat_interactions",  # CellChatDB interaction table
    "lr": "lr_layer1",  # layer 1: ligand -> receptor
    "rtf": "rtf_layer2",  # layer 2: receptor -> transcription factor
    "tft": "tft_layer3",  # layer 3: transcription factor -> target gene
}

_ENV_VAR = "SCRICH_DATA_DIR"
_data_dir_override: Path | None = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def check_species(species: str) -> str:
    """Validate and normalise a species name.

    Parameters
    ----------
    species
        ``"human"`` or ``"mouse"`` (case-insensitive).

    Returns
    -------
    str
        The lower-case species name.

    Raises
    ------
    ValueError
        If the species is not supported.
    """
    sp = str(species).lower()
    if sp not in SPECIES:
        raise ValueError(f"Unsupported species {species!r}. Choose one of {SPECIES}.")
    return sp


def data_dir() -> Path:
    """Folder where external database files are looked up and cached.

    Resolution order: :func:`set_data_dir` > ``$SCRICH_DATA_DIR`` > OS cache folder.

    Returns
    -------
    pathlib.Path
        The data folder (it may not exist yet).
    """
    if _data_dir_override is not None:
        return _data_dir_override
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env).expanduser()
    return Path(pooch.os_cache("scrich")) / DB_VERSION


def set_data_dir(path: str | os.PathLike | None) -> None:
    """Set the folder used for the external database files in this session.

    Parameters
    ----------
    path
        Folder containing (or that will receive) the database files.
        ``None`` restores the default behaviour.
    """
    global _data_dir_override
    _data_dir_override = None if path is None else Path(path).expanduser()
    clear_memory_cache()


def clear_memory_cache() -> None:
    """Drop database tables cached in memory (files on disk are kept)."""
    _read_exfinder.cache_clear()
    _read_pathway_genes.cache_clear()


def _base_url() -> str:
    if ZENODO_RECORD is None:
        return ""
    return f"https://zenodo.org/records/{ZENODO_RECORD}/files/"


def _fetch(fname: str) -> Path:
    """Return a local path to ``fname``, downloading it if needed."""
    folder = data_dir()
    local = folder / fname
    if ZENODO_RECORD is None:
        # No public URL yet: the file must already be present locally.
        if not local.exists():
            raise FileNotFoundError(
                f"Database file {fname!r} not found in {folder}.\n"
                "The scRICH database has not been published online yet. Point scRICH to a local copy with\n"
                f"    export {_ENV_VAR}=/path/to/scrich_db_v1\n"
                "or, in Python,\n"
                "    scrich.db.set_data_dir('/path/to/scrich_db_v1')"
            )
        expected = REGISTRY[fname].split(":", 1)[1]
        if pooch.file_hash(str(local)) != expected:
            raise ValueError(f"Checksum mismatch for {local}. The file may be corrupted or from another version.")
        return local
    registry = pooch.create(path=folder, base_url=_base_url(), registry=REGISTRY)
    return Path(registry.fetch(fname, progressbar=False))


# --------------------------------------------------------------------------- #
# Bundled tables
# --------------------------------------------------------------------------- #
@cache
def _read_pathway_genes(kind: str, species: str) -> pd.DataFrame:
    ref = resources.files("scrich") / "data" / "pathway_database" / f"{kind}_{species}.csv"
    with resources.as_file(ref) as path:
        return pd.read_csv(path)


def load_pathway_genes(kind: Literal["ligand", "receptor"], species: str = "human") -> pd.DataFrame:
    """Load the bundled pathway -> ligand or pathway -> receptor table.

    Parameters
    ----------
    kind
        ``"ligand"`` or ``"receptor"``.
    species
        ``"human"`` or ``"mouse"``.

    Returns
    -------
    pandas.DataFrame
        Two columns: ``pathway`` and ``ligand`` (or ``receptor``). Receptor
        complexes are encoded with ``_`` (e.g. ``"TGFBR1_TGFBR2"``).
    """
    if kind not in ("ligand", "receptor"):
        raise ValueError(f"kind must be 'ligand' or 'receptor', got {kind!r}")
    return _read_pathway_genes(kind, check_species(species)).copy()


# --------------------------------------------------------------------------- #
# External tables
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def _read_exfinder(layer: str, species: str) -> pd.DataFrame:
    fname = f"{EXFINDER_LAYERS[layer]}_{species}.parquet"
    return pd.read_parquet(_fetch(fname))


def load_exfinder(layer: Literal["cellchat", "lr", "rtf", "tft"], species: str = "human") -> pd.DataFrame:
    """Load one of the external exFinder / CellChat database tables.

    The file is downloaded on first use (or read from :func:`data_dir`) and
    kept in memory for subsequent calls.

    Parameters
    ----------
    layer
        Which table to load:

        - ``"cellchat"``: CellChatDB interaction table (``pathway_name``, ``ligand``, ``receptor``, ...)
        - ``"lr"``: exFinder layer 1, ligand -> receptor
        - ``"rtf"``: exFinder layer 2, receptor -> transcription factor
        - ``"tft"``: exFinder layer 3, transcription factor -> target gene
    species
        ``"human"`` or ``"mouse"``.

    Returns
    -------
    pandas.DataFrame
        For the exFinder layers: columns ``from``, ``to`` and ``source``.

    Raises
    ------
    FileNotFoundError
        If the file is not available locally and cannot be downloaded.
    """
    if layer not in EXFINDER_LAYERS:
        raise ValueError(f"Unknown layer {layer!r}. Choose one of {list(EXFINDER_LAYERS)}.")
    return _read_exfinder(layer, check_species(species)).copy()


def fetch_all(species: str | None = None) -> list[Path]:
    """Download (or verify) all external database files in advance.

    Useful before working offline, e.g. on a compute node without internet.

    Parameters
    ----------
    species
        Only fetch files for this species. ``None`` fetches all.

    Returns
    -------
    list of pathlib.Path
        Local paths of the files.
    """
    sp = None if species is None else check_species(species)
    return [_fetch(f) for f in REGISTRY if sp is None or f.endswith(f"_{sp}.parquet")]
