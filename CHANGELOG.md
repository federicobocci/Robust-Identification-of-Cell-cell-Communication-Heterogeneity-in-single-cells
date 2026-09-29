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

### Code organisation (Phase 3)
- `scrich/tools` reorganised by task (no change in results or in the `scrich.tl` API):
  `pathways.py` (pathway selection, ligand/receptor lookup), `modes.py` (similarity matrix, number of modes,
  consensus clustering), `heterogeneity.py` (fraction matrix, heterogeneity score, permutation test),
  `patterns.py` (signaling patterns, mutual information), `ccc.py` (CCC matrix), `targets.py` (downstream
  genes), `annotations.py` (renaming) and the private `_utils.py`. Replaces `mode_clustering_func.py`,
  `database_func.py`, `sign_patterns.py`, `similarity.py`, `misc_functions.py`, `signaling_func.py` and
  `tf_tar_functions.py`.
- `extract_neighbors` / `neighbor_avg` are defined once in `tools/_utils.py` (the copy in `plots/velocity.py`
  was removed).

### Robustness and consistency of `scrich.tl` (Phase 3)
- **Memory**: the cell-cell similarity matrix is no longer built as a dense cells x cells matrix in
  `tl.all_path_sim`. It is kept as its low-rank factor (cells x number of clusters, e.g. 90 columns with the
  defaults) and the spectral gap is computed exactly from the factor. For 50,000 cells this is ~36 MB instead
  of ~20 GB per matrix; eigenvalues agree with the dense computation to ~1e-15 and results are unchanged.
  `tl.single_path_sim` still returns the dense matrix unless `return_factor=True`.
- `tl.permutation_testing` has a `seed` argument (default 0) for reproducible p-values (the test itself is
  unchanged; `seed=None` uses NumPy's global random state as before).
- `species='human'|'mouse'` replaces `human=True/False` in `tl.pathways_overview`, `tl.find_genes`,
  `tl.select_pathways` and `tl.all_path_sim` (`human` still works, with a FutureWarning).
- Progress messages use Python logging instead of `print`; control them with `scrich.set_verbosity('warning'|
  'info'|'debug')`. Per-run clustering messages are now at debug level; the leftover debug print in
  `all_path_sim` was removed.
- Input checks raise `ValueError`/`KeyError`/`TypeError` with explicit messages instead of `assert`
  (e.g. invalid `model`, missing velocity layer, unknown pathways, `pathways_overview` not run).
- Global warning filters set on import (which hid all `FutureWarning`s in the user's session) were removed;
  only pandas' DataFrame-fragmentation warning is silenced locally where scRICH adds `adata.obs` columns.
- `tl.find_sign_patterns` passes `flavor='leidenalg'` to Leiden explicitly (scanpy's current default, same
  results) to avoid scanpy's warning about a future default change.
- `tl.pathways_overview` scans pathways in sorted order (deterministic across Python sessions).
- `tl.import_database`: `input_target` default changed from a mutable `{}` to `None`.

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
- **Cell-type order bug fixed**: passing `order=` to `tl.all_path_sim` / `tl.consensus_clustering` stored the
  fraction matrix and heterogeneity scores in that order, while plots and `tl.permutation_testing` assumed
  alphabetical order, so results could be shown under the wrong cell type. Results are now always stored in
  sorted order, with labels in `adata.uns['pathways'][pathway]['cell_types']` and `['modes']`; `order` there
  is deprecated (FutureWarning) and only the plotting functions' `order` changes the display.
  `pl.heatmap_one_pathway` and `pl.mode_composition` reorder columns accordingly; `mode_composition` also
  accepts a mode name, and the heatmap shows mode names.
- `tl.import_database(unspliced=False)`: gene averages were a single number (missing `axis`), so the top
  downstream genes were chosen arbitrarily. Now per-gene means (sparse or dense `X`).
- `tl.all_path_sim(target=True)` crashed (`KeyError: 'targets'`). It now uses the downstream genes selected by
  `import_database` (`adata.uns['TF']`), passes `unspliced=moments` to it, applies neighbour averaging to them
  like ligands/receptors, and falls back to ligands/receptors (with a warning) for pathways with no expressed
  downstream genes.
- `target` now defaults to **False** in `tl.all_path_sim` and `tl.single_path_sim` (was True). With
  `target=True`, `tl.consensus_clustering` (new argument `target`) now also clusters on the downstream genes,
  so they are used consistently for both the number of modes and the mode assignment. Pathways without
  expressed downstream genes use ligands/receptors only (`adata.uns['pathways'][p]['uses_targets']`).
- `tl.consensus_clustering` has a new `neighbor_average` argument (default False) to assign modes from
  neighbour-averaged expression; `tl.all_path_sim` exposes it as `consensus_neighbor_average` (default False).
  Defaults keep the v0.1.0 behaviour: the similarity matrix follows `neighbor_average`, while the final mode
  assignment uses unsmoothed expression.
- Removed unused code: `cellchat_DB`, `layer_3_DB`, `get_targets`, and the unused kinetic functions
  (`K1`, `K1_RNAVel`, `K2`, `beta_RNAVel`, `gamma`, `P_pair_LRU_RNAVel`, `P_pair_LRD`, `P_pair_all`).
  `tl.pathway_strength` and `tl.compute_ccc_matrix_CellchatBenchmark` were removed from the package and kept
  for reference in `dev/unintegrated/signaling_extras.py`.
- **Cell-type heterogeneity score corrected** (`tl.consensus_clustering` / `cci_heterogeneity`): the score
  was inverted. It is now `1 - mean normalized deviation`: close to 1 when a cell type contains the
  communication modes in the same proportions as the whole population, close to 0 when its composition
  deviates maximally. Values from v0.1.0 correspond to `1 - score`.
- `tl.rename_modes` stores modes as an ordered categorical that keeps the original mode order. Modes not in
  the dictionary keep their label (as strings if any new name is a string), and several modes can be merged
  by giving them the same name. Renaming again with the new names works.
- `tl.find_sign_patterns` works after `rename_modes` (and with any mode labels); one-to-one renaming gives
  exactly the same patterns as before renaming.
- `tl.find_sign_patterns`: the number of neighbours is now an argument, `n_neighbors`, with default **30**
  (previously fixed at 10). With 10 neighbours the cell graph often splits into many disconnected pieces
  because many cells share identical mode combinations, so the number of patterns depended on library
  versions and random seeds rather than on `res`. New argument `random_state` (neighbour search, UMAP,
  Leiden); the function now reports the number of graph pieces, warns when there is more than one, and
  stores its parameters in `adata.uns['sign_pattern_params']`. **This changes results compared to v0.1.0**;
  use `n_neighbors=10` to reproduce the previous setting.
- The `velocity` extra caps `numpy<2.4`: scvelo 0.3.4 fails in `scv.tl.velocity` with NumPy ≥ 2.4.
- `KMeans` now sets `n_init=10` explicitly, the default of scikit-learn < 1.4 used in the paper.
  (Newer scikit-learn defaults to a single initialisation, which changed the clustering.)
