# Changelog

## Unreleased (branch `refactor/v1`)

### Packaging and environment (Phase 1)
- Installable package: `pyproject.toml` (hatchling), `src/` layout, version taken from git tags.
- Direct dependencies with minimum versions replace the full `pip freeze`
  (kept for reference as `dev/requirements_paper_py39_freeze.txt`).
- Optional extras: `velocity` (scvelo), `chord` (d3blocks), `tutorials`, `dev`, `docs`.
  `import scrich` no longer needs scvelo or d3blocks; they are loaded only by the functions that use them.
- Tested on Python 3.12 and 3.13 with current releases (numpy 2.5, pandas 3.0, scanpy 1.12,
  anndata 0.13, scikit-learn 1.9, numba 0.67).
- Explicit public API in `scrich.tl` and `scrich.pl` (no more `import *`). Internal helpers
  such as `extract_neighbors` / `neighbor_avg` are no longer exposed in `scrich.pl`.
- Removed committed `__pycache__` files and the duplicated `pathway_database/` folder.
- `cellflow_tools.py` / `cellflow_plots.py` (not integrated, depend on splicejac) moved to `dev/unintegrated/`.
- Tutorials moved to `tutorials/`.
- Basic test suite (`tests/`).

### Databases (Phase 2)
- New module `scrich.db`: `load_pathway_genes`, `load_exfinder`, `fetch_all`, `data_dir`, `set_data_dir`.
- Pathway → ligand/receptor tables are bundled in the package; they no longer depend on the
  current working directory.
- exFinder tables converted from CSV (597 MB) to Parquet (21 MB, same content), with SHA256
  checksums, fetched with pooch and cached. Local copies via `SCRICH_DATA_DIR`.
- Removed hard-coded relative paths (`../../exfinder_database/...`, `pathway_database/...`).

### Fixes
- `pl.pattern(..., human=True)` read the *receptor* table as ligands; now reads the ligand table.
- `tl.all_path_sim(..., target=True, human=False)` loaded the human TF database; it now uses mouse.
- Compatibility with current libraries: `matplotlib.cm.get_cmap` (removed) in `pl.redundancy`,
  and `AgglomerativeClustering(affinity=...)` (renamed `metric`) in `pl.pathway_hierarchy`.
- Plots no longer fail or silently drop categories when there are more signaling patterns, modes or
  cell types than colours in the palette (`pl.pattern_summary`, `pl.pattern`, `pl.alluvial_*`,
  `pl.scatter2D`, `pl.mode_violin`, `pl.mode_composition`, `pl.coarse_grained_map`). Palettes are now
  extended automatically (new helper `scrich.plots._colors.discrete_colors`).
- `tl.find_sign_patterns` prints how many patterns were found at the chosen resolution and the number
  of cells in each (new argument `print_info=True`).
- The `velocity` extra caps `numpy<2.4`: scvelo 0.3.4 fails in `scv.tl.velocity` with NumPy ≥ 2.4.
- `KMeans` now sets `n_init=10` explicitly, the default of scikit-learn < 1.4 used in the paper.
  (Newer scikit-learn defaults to a single initialisation, which changed the clustering.)
