# Installation

scRICH requires Python 3.10 or newer (3.12 recommended). Install it in a fresh environment.

## With pip

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install "scrich[tutorials] @ git+https://github.com/federicobocci/Robust-Identification-of-Cell-cell-Communication-Heterogeneity-in-single-cells"
```

Once scRICH is on PyPI this becomes `pip install "scrich[tutorials]"`.

## From a local copy (development)

```bash
git clone https://github.com/federicobocci/Robust-Identification-of-Cell-cell-Communication-Heterogeneity-in-single-cells scrich
cd scrich
python3.12 -m venv .venv && source .venv/bin/activate   # or: uv venv -p 3.12 .venv
pip install -e ".[tutorials,dev]"
```

With conda: `conda env create -f environment.yml && conda activate scrich`.

## Optional extras

`pip install scrich` installs the core package. Extras add optional features:

| Extra | Installs | Needed for |
|---|---|---|
| `velocity` | scvelo, seaborn | RNA-velocity functions and plots (`scrich.pl.plot_maps`, `expr_map`, ...) |
| `alluvial` | plotly, kaleido | alluvial plots (`scrich.pl.alluvial_*`); kaleido saves them as pdf/png |
| `chord` | d3blocks | {func}`scrich.pl.chord_diagram` |
| `tutorials` | all of the above, JupyterLab, LIANA | running the tutorials |
| `dev` | pytest, ruff, pre-commit | development |
| `docs` | Sphinx and extensions | building this documentation |

Functions that need an optional package raise an error explaining which extra to install.

```{note}
scvelo 0.3.4 does not work with NumPy 2.4 or newer; the `velocity` extra therefore installs `numpy<2.4`.
scvelo 0.3.4 also no longer log-transforms the data in `scv.pp.filter_and_normalize`; add
`sc.pp.log1p(adata)` after it to reproduce the tutorials.
```
