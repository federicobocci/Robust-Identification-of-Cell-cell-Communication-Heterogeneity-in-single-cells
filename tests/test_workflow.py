"""End-to-end smoke test of the main scRICH workflow on a small synthetic dataset."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import scrich as ri


def _run(adata):
    ri.tl.pathways_overview(adata, species="mouse")
    ri.tl.all_path_sim(adata, "celltype", pathways_to_use=2, target=False, species="mouse", neighbor_average=True)
    np.random.seed(0)
    ri.tl.permutation_testing(adata, key="celltype")
    return adata.uns["selected_pathways"]


def test_mode_clustering(adata_mouse):
    paths = _run(adata_mouse)
    assert len(paths) == 2
    for p in paths:
        n_opt = adata_mouse.uns["SymNMF"][p]["optimal"]
        assert adata_mouse.obs[p + "_modes"].nunique() == n_opt
        frac = adata_mouse.uns["fraction_mat"][p]
        assert frac.sum() == adata_mouse.n_obs
        assert adata_mouse.uns["pathways"][p]["cci_heterogeneity"].shape == (3,)


def test_downstream_and_plots(adata_mouse):
    paths = _run(adata_mouse)
    ri.tl.pairwise_MI(adata_mouse)
    assert adata_mouse.uns["CCC_similarity_map"].shape == (2, 2)
    ri.tl.compute_ccc_matrix(adata_mouse, paths[0], key="celltype", min_cells=5)
    mat = adata_mouse.uns["ccc_mat"][paths[0]]["mat"]
    assert np.isclose(mat.sum(), 1.0)
    for f in [
        lambda: ri.pl.plot_mode_gap(adata_mouse, paths[0], show=False),
        lambda: ri.pl.mode_violin(adata_mouse, paths[0] + "_modes", show=False),
        lambda: ri.pl.heatmap_one_pathway(adata_mouse, paths[0], "celltype", show=False),
        lambda: ri.pl.redundancy(adata_mouse, show=False),
        lambda: ri.pl.pathway_hierarchy(adata_mouse, show=False),
    ]:
        f()
        plt.close("all")


@pytest.mark.network
def test_import_database(adata_mouse, exfinder_dir):
    paths = _run(adata_mouse)
    ri.tl.import_database(adata_mouse, [paths[0]], species="mouse", unspliced=True)
    assert paths[0] in adata_mouse.uns["TF"]
