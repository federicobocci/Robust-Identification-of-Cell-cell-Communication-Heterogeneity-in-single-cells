# Getting started

A scRICH analysis starts from an {class}`~anndata.AnnData` object with log-normalized expression in
`adata.X`, cell-type annotations in `adata.obs`, and a nearest-neighbour graph and embedding (e.g. from
{func}`scanpy.pp.neighbors` and {func}`scanpy.tl.umap`, or {func}`scvelo.pp.moments`).

## Analysis steps

| Step | Function | Main results |
|---|---|---|
| 1. Pathways detected in the data | {func}`scrich.tl.pathways_overview` | `adata.uns['pathways']`, `adata.uns['paths_overview']` |
| 2. Communication modes of each pathway | {func}`scrich.tl.all_path_sim` | `adata.obs['<pathway>_modes']`, `adata.uns['fraction_mat']`, `adata.uns['SymNMF']` |
| 3. Heterogeneity of cell types | computed in step 2; tested with {func}`scrich.tl.permutation_testing` | `adata.uns['pathways'][p]['cci_heterogeneity']`, `adata.uns['permutation_test']` |
| 4. Similarity between pathways | {func}`scrich.tl.pairwise_MI` | `adata.uns['CCC_similarity_map']` |
| 5. Signaling patterns across pathways | {func}`scrich.tl.find_sign_patterns` | `adata.obs['sign_pattern']` |
| 6. CCC matrix between cell states | {func}`scrich.tl.compute_ccc_matrix` | `adata.uns['ccc_mat']` |

Downstream genes of a pathway (optional, for `target=True` / `include_target=True`) are selected with
{func}`scrich.tl.import_database`, which uses the exFinder database (see {doc}`databases`).

## Example

```python
import scanpy as sc
import scvelo as scv
import scrich as ri

adata = scv.datasets.gastrulation_erythroid()
scv.pp.filter_and_normalize(adata, min_shared_counts=20)
sc.pp.log1p(adata)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

# 1-3: pathways, communication modes, heterogeneity
ri.tl.pathways_overview(adata, species="mouse")
ri.pl.pathways_overview(adata, top=10)
ri.tl.all_path_sim(adata, "celltype", pathways_to_use=10, species="mouse", neighbor_average=True)
ri.tl.permutation_testing(adata, key="celltype")
ri.pl.pathway_heterogeneity_summary(adata, key="celltype")

# name the modes of a pathway after inspecting them
ri.pl.mode_violin(adata, "IGF_modes")
ri.tl.rename_modes(adata, "IGF", {0: "Receiver", 1: "Inactive", 2: "Sender/Receiver"})

# 4-5: redundancy between pathways and signaling patterns
ri.tl.pairwise_MI(adata)
ri.pl.redundancy(adata)
ri.tl.find_sign_patterns(adata, res=0.05)
ri.pl.pattern_summary(adata, "celltype")

# 6: CCC matrix of one pathway
ri.tl.compute_ccc_matrix(adata, "IGF", key="celltype", min_cells=10)
```

## Reproducibility

All random steps take a seed (`seed`, `random_state`), so repeated runs give the same results. Results can
differ slightly between versions of scikit-learn and scanpy; the versions used in the paper are listed in
`dev/requirements_paper_py39_freeze.txt` of the repository.

Progress messages are controlled with {func}`scrich.set_verbosity`.
