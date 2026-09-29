import importlib.util
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import scrich as ri


@pytest.fixture
def analysed(adata_mouse):
    ri.tl.pathways_overview(adata_mouse, species="mouse")
    ri.tl.all_path_sim(adata_mouse, "celltype", pathways_to_use=3, species="mouse", neighbor_average=True)
    ri.tl.permutation_testing(adata_mouse, key="celltype")
    ri.tl.pairwise_MI(adata_mouse)
    adata_mouse.obsm["X_umap"] = np.random.default_rng(0).normal(size=(adata_mouse.n_obs, 2))
    return adata_mouse


def test_heatmap_xlim_right(analysed):
    p = analysed.uns["selected_pathways"][0]
    ri.pl.heatmap_one_pathway(analysed, p, "celltype", xlim_right=0.8, savefig=False)
    ax = plt.gcf().axes[-1]
    assert ax.get_xlim() == (0.0, 0.8)
    plt.close("all")


def test_saved_before_shown(analysed, tmp_path, monkeypatch):
    """show() may clear the figure (scripts, some backends); the file must be written before that."""
    monkeypatch.setattr(plt, "show", lambda *a, **k: plt.close("all"))
    p = analysed.uns["selected_pathways"][0]
    f = tmp_path / "violin.png"
    ri.pl.mode_violin(analysed, p + "_modes", showfig=True, savefig=True, figname=str(f), format="png")
    assert f.stat().st_size > 5000  # a real plot, not an empty canvas


def test_no_saving_by_default_for_former_string_defaults(analysed, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ri.pl.redundancy(analysed)
    ri.pl.pathway_hierarchy(analysed)
    p = analysed.uns["selected_pathways"][0]
    ri.pl.alluvial_onepath(analysed, p, "celltype", showfig=False)
    plt.close("all")
    assert os.listdir(tmp_path) == []


@pytest.mark.skipif(importlib.util.find_spec("kaleido") is not None, reason="kaleido installed")
def test_alluvial_static_export_needs_kaleido(analysed, tmp_path):
    p = analysed.uns["selected_pathways"][0]
    with pytest.raises(ImportError, match="kaleido"):
        ri.pl.alluvial_onepath(analysed, p, "celltype", savefig=True, showfig=False, figname=str(tmp_path / "a.pdf"))


def test_mode_order_follows_renamed_modes(analysed):
    p = analysed.uns["selected_pathways"][0]
    ri.tl.rename_modes(analysed, p, {0: "Zeta", 1: "Alpha", 2: "Mid"})
    ri.pl.scatter2D(analysed, p + "_modes", savefig=False)
    labels = [t.get_text() for t in plt.gca().get_legend().get_texts()]
    assert labels == list(analysed.obs[p + "_modes"].cat.categories)  # original mode order, not alphabetical
    plt.close("all")


def test_pattern_probabilities_numeric_order(analysed):
    from scrich.plots.velocity import compute_pattern_prob

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 12, analysed.n_obs).astype(str)
    analysed.obs["sign_pattern"] = pd.Categorical(labels, categories=[str(i) for i in range(12)])
    order, prob = compute_pattern_prob(analysed)
    assert list(order) == [str(i) for i in range(12)]
    assert np.allclose(prob.sum(axis=1), 1)
