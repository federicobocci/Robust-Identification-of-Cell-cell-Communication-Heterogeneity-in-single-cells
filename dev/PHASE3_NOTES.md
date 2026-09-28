# Notes for Phase 3 (refactor) — issues found during Phases 1–2

These were **not** changed in Phases 1–2 because fixing them changes scientific output or
behaviour. Each needs a decision from the author and a regression test.

## Results depend on library versions

Regression check (`dev/regression/`): same pipeline on the same synthetic dataset,
old code/Python 3.9/paper environment vs new code/Python 3.12/current libraries.

- New code in the **old** environment: outputs are identical to the old code (the data layer
  and packaging changes did not change results).
- New code in the **new** environment: the optimal number of modes is the same for every
  pathway and 5 of 6 pathways give identical mode partitions (ARI = 1.0); one gives ARI = 0.89.
  Eigenvalues differ by ~1%. Cause: scikit-learn's KMeans implementation changed between 1.2
  and 1.9 (`n_init` was already pinned to 10 to match the paper).
- **Mode label numbers can be permuted** between library versions (the same modes may be
  called 0/1/2 in a different order). The tutorials hard-code names such as
  `rename_modes(adata, 'IGF', {0: 'Receiver', 1: 'Inactive', 2: 'Sender/Receiver'})`, which
  would then assign the wrong names. Proposal: order modes deterministically, e.g. by mean
  ligand + receptor expression.

## Bugs

1. `tools/tf_tar_functions.get_counts(unspliced=False)`: `np.mean(adata.X.toarray())` has no
   `axis`, so every gene gets the same average and the "top TFs" are an arbitrary subset. It
   also fails on dense `X`. Should be `np.asarray(adata.X.mean(axis=0)).ravel()`. Tutorials use
   `unspliced=True`, which is not affected.
2. `tl.all_path_sim(target=True)` calls `import_database`, which fills `adata.uns['TF']`,
   but `single_path_sim` reads `adata.uns['targets']` → `KeyError`.
3. `pl.heatmap_one_pathway(xlim_right=...)`: uses undefined `lim_right` → `NameError`
   (probably meant `plt.xlim([0, xlim_right])`).
4. `pl.pathway_hierarchy(savefig='True')`: default is the *string* `'True'`, so it saves a file
   by default.
5. `tl.permutation_testing` uses the global NumPy random state with no `seed` argument, so
   p-values are not reproducible.
6. `spectral_analysis_opt`: `eigsh(..., which='SM')` without a fixed `v0` uses a random start
   vector; results can vary slightly between runs.
7. `compute_ccc_matrix_CellchatBenchmark` checks `pathway + 'tar'` instead of `pathway + '_tar'`
   and never stores its result.

## Scalability

- `sim_matrix` builds a dense `n_cells × n_cells` matrix (plus one per KMeans run): ~20 GB at
  50,000 cells. Consider a co-association matrix restricted to kNN pairs, or subsampling.
- `compute_sim` is an O(n²) Python/numba loop; `labels[:, None] == labels[None, :]` is equivalent.

## Code quality

- `print` statements everywhere (e.g. the leftover `print(adata.uns.keys(), '4')` in
  `all_path_sim`) → use `logging` with a verbosity setting.
- `assert` used for input validation → raise `ValueError` / `KeyError` with clear messages.
- `human=True/False` in some functions and `species='human'/'mouse'` in others → use `species`
  everywhere (keep `human` as a deprecated alias for one release).
- Duplicated helpers: `extract_neighbors` / `neighbor_avg` exist in both
  `tools/mode_clustering_func.py` and `plots/velocity.py`.
- Global `warnings.simplefilter('ignore', FutureWarning)` in `mode_clustering_func.py` hides
  warnings for the user's whole session.
- `ruff check src` reports ~120 issues in legacy modules (mostly unused imports/variables).
