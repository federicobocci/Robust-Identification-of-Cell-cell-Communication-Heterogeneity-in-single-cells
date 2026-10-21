# scRICH — Robust Identification of Cell-cell Communication Heterogeneity

[![Tests](https://github.com/federicobocci/Robust-Identification-of-Cell-cell-Communication-Heterogeneity-in-single-cells/actions/workflows/tests.yml/badge.svg)](https://github.com/federicobocci/Robust-Identification-of-Cell-cell-Communication-Heterogeneity-in-single-cells/actions/workflows/tests.yml)
[![Lint](https://github.com/federicobocci/Robust-Identification-of-Cell-cell-Communication-Heterogeneity-in-single-cells/actions/workflows/lint.yml/badge.svg)](https://github.com/federicobocci/Robust-Identification-of-Cell-cell-Communication-Heterogeneity-in-single-cells/actions/workflows/lint.yml)
[![Docs](https://github.com/federicobocci/Robust-Identification-of-Cell-cell-Communication-Heterogeneity-in-single-cells/actions/workflows/docs.yml/badge.svg)](https://github.com/federicobocci/Robust-Identification-of-Cell-cell-Communication-Heterogeneity-in-single-cells/actions/workflows/docs.yml)
[![Documentation](https://readthedocs.org/projects/scrich/badge/?version=latest)](https://scrich.readthedocs.io/en/latest/)

This repository hosts the code of the scRICH method, published in
[Cell Systems](https://www.cell.com/cell-systems/fulltext/S2405-4712(26)00185-7).
The method is developed by [Federico Bocci](https://boccilab.org)
at Radboud University.

![scRICH schematic](https://raw.githubusercontent.com/federicobocci/Robust-Identification-of-Cell-cell-Communication-Heterogeneity-in-single-cells/main/misc/GA.png)

> **Paper version.** scRICH 1.0 is an installable Python package with the same analysis methods
> as the paper. The exact code used in the paper is the GitHub release
> [`v0.1.0.0`](https://github.com/federicobocci/Robust-Identification-of-Cell-cell-Communication-Heterogeneity-in-single-cells/releases/tag/v0.1.0.0)
> (Zenodo DOI [10.5281/zenodo.20394264](https://doi.org/10.5281/zenodo.20394264)).
> See the [changelog](https://scrich.readthedocs.io/en/latest/changelog.html) for what changed.

## Installation

scRICH requires Python ≥ 3.10 (3.12 recommended). Install into a fresh environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install "scrich[tutorials]"     # or: pip install scrich   (core package only)
```

With [uv](https://docs.astral.sh/uv/): `uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install "scrich[tutorials]"`.

**From the source code (development)**

```bash
git clone https://github.com/federicobocci/Robust-Identification-of-Cell-cell-Communication-Heterogeneity-in-single-cells scrich
cd scrich
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[tutorials,dev]"      # or: pip install -e ".[tutorials,dev]"
```

With conda (from the source code): `conda env create -f environment.yml && conda activate scrich`

Optional extras: `velocity` (scvelo + seaborn, for RNA-velocity functions and plots), `alluvial` (plotly +
kaleido, for `scrich.pl.alluvial_*`), `chord` (d3blocks, for `scrich.pl.chord_diagram`), `tutorials` (all of
these + JupyterLab + LIANA), `dev` (pytest, ruff).
A plain `pip install scrich` installs only the core package.

## Databases

scRICH uses two kinds of database files (see `scrich.db`):

| Data | Size | Where it lives |
|---|---|---|
| Pathway → ligand / receptor tables (CellChatDB) | ~50 kB | Shipped inside the package |
| exFinder layers (L→R, R→TF, TF→target) + CellChat interactions, human & mouse | ~21 MB | [Zenodo](https://doi.org/10.5281/zenodo.23319307), downloaded on first use and cached |

The external files are only needed for `scrich.tl.import_database` (target genes of a pathway).
They are downloaded automatically from Zenodo (DOI
[10.5281/zenodo.23319307](https://doi.org/10.5281/zenodo.23319307)) the first time they are needed,
then cached. Nothing needs to be set up.

On a computer without internet (e.g. a cluster compute node), download the files once from a
machine that has internet and point scRICH to that folder:

```python
import scrich as ri
ri.db.set_data_dir("/path/to/scrich_db")   # or: export SCRICH_DATA_DIR=/path/to/scrich_db
ri.db.fetch_all()   # download (or verify) all files now
```

Files are verified with SHA256 checksums. `scripts/convert_exfinder_to_parquet.py` documents how
they were produced from the original exFinder data.

## Quick start

```python
import scrich as ri

ri.tl.pathways_overview(adata, species="mouse")
ri.tl.all_path_sim(adata, "celltype", pathways_to_use=10, species="mouse", neighbor_average=True)
ri.tl.permutation_testing(adata, key="celltype")
ri.pl.pathway_heterogeneity_summary(adata, key="celltype")
```

Plots are shown by default and never saved automatically. To keep a plot, return it and save it:

```python
ax = ri.pl.single_pathway_heterogeneity(adata, "IGF", "celltype", show=False)
ax.figure.savefig("IGF_heterogeneity.pdf")
```

Use `ri.settings.autoshow = False` in scripts, and `ri.set_verbosity("warning")` to reduce progress messages.

## Tutorials

The notebooks in `tutorials/` cover the main analysis (tutorial 1), modifying the ligand-receptor
set of a pathway (2), integrating LIANA (3), comparing RNA velocity methods (4) and spatial
transcriptomics (5). Datasets for tutorials 3–5 are on Zenodo, record
[19859013](https://zenodo.org/records/19859014).

## Documentation

The documentation is online at **https://scrich.readthedocs.io** (installation, analysis steps,
databases, plotting, tutorials and the API reference generated from the docstrings). Its sources
are in `docs/` and are built with Sphinx for Read the Docs (`.readthedocs.yaml`). To build it locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html      # then open docs/_build/html/index.html
```

## Development

```bash
uv pip install -e ".[dev]"
pre-commit install                       # run ruff and file checks before each commit
pytest              # all tests (the database tests download ~21 MB on first run)
pytest -m "not network"   # skip the tests that need internet
```

Continuous integration (GitHub Actions, `.github/workflows/`) runs on every push and pull request:
`tests.yml` (Linux and macOS, Python 3.10/3.12/3.13, with and without the optional extras; also weekly),
`lint.yml` (the pre-commit checks) and `docs.yml` (documentation build, warnings as errors).

## Contacts

For any question, get in touch with Federico Bocci (federico.bocci at ru.nl).
