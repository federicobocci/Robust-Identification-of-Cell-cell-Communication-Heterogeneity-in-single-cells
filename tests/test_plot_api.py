"""Every scrich.pl function: runs, returns axes/figures with show=False, supports ax= where single-panel,
handles the deprecated savefig/showfig/figname arguments, and follows scrich.settings.autoshow."""

import warnings

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import scrich as ri


@pytest.fixture(scope="module")
def full():
    """Synthetic mouse dataset with modes, patterns, CCC matrix and (if scvelo is installed) RNA velocity."""
    import sys

    sys.path.insert(0, "tests")
    import conftest

    rng = np.random.default_rng(0)
    a = conftest.adata_mouse.__wrapped__()
    counts = np.round(np.expm1(a.X.toarray())).astype(np.float32)
    a.layers["spliced"] = sp.csr_matrix(counts)
    a.layers["unspliced"] = sp.csr_matrix(rng.poisson(counts * 0.3 + 0.2).astype(np.float32))
    a.obsm["X_umap"] = rng.normal(size=(a.n_obs, 2)) + np.repeat(np.eye(3, 2) * 4, a.n_obs // 3, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ri.set_verbosity("warning")
        try:  # neighbours and moments as in the tutorials, when scvelo is available
            import scanpy as sc
            import scvelo as scv

            scv.settings.verbosity = 0
            sc.pp.pca(a, n_comps=20)
            sc.pp.neighbors(a, n_neighbors=15)
            scv.pp.moments(a, n_neighbors=15)
        except ImportError:
            pass
        ri.tl.pathways_overview(a, species="mouse")
        ri.tl.all_path_sim(a, "celltype", pathways_to_use=3, species="mouse", neighbor_average=True)
        ri.tl.permutation_testing(a, key="celltype")
        ri.tl.pairwise_MI(a)
        ri.tl.find_sign_patterns(a, res=0.5, print_info=False)
        p = a.uns["selected_pathways"][0]
        ri.tl.compute_ccc_matrix(a, p, key="celltype", min_cells=5)
        a.obs["velocity_pseudotime"] = rng.random(a.n_obs)
        try:
            import scvelo as scv

            scv.tl.velocity(a, mode="stochastic")
            scv.tl.velocity_graph(a)
            a.uns["has_velocity"] = True
        except Exception:  # scvelo missing or failed
            a.uns["has_velocity"] = False
        ri.set_verbosity("info")
    return a


def _p(a, i=0):
    return a.uns["selected_pathways"][i]


MATPLOTLIB_PLOTS = {
    "pathways_overview": lambda a, **k: ri.pl.pathways_overview(a, top=5, **k),
    "plot_mode_gap": lambda a, **k: ri.pl.plot_mode_gap(a, _p(a), **k),
    "scatter2D": lambda a, **k: ri.pl.scatter2D(a, _p(a) + "_modes", **k),
    "mode_violin": lambda a, **k: ri.pl.mode_violin(a, _p(a) + "_modes", **k),
    "pathway_umap": lambda a, **k: ri.pl.pathway_umap(a, _p(a), key="celltype", **k),
    "feature_plot": lambda a, **k: ri.pl.feature_plot(a, _p(a), **k),
    "heatmap_one_pathway": lambda a, **k: ri.pl.heatmap_one_pathway(a, _p(a), "celltype", **k),
    "single_pathway_heterogeneity": lambda a, **k: ri.pl.single_pathway_heterogeneity(a, _p(a), "celltype", **k),
    "pathway_heterogeneity_summary": lambda a, **k: ri.pl.pathway_heterogeneity_summary(a, key="celltype", **k),
    "state_heterogeneity_summary": lambda a, **k: ri.pl.state_heterogeneity_summary(a, "celltype", **k),
    "heterogeneity_heatmap": lambda a, **k: ri.pl.heterogeneity_heatmap(a, "celltype", **k),
    "mode_composition": lambda a, **k: ri.pl.mode_composition(a, "celltype", _p(a), 0, **k),
    "redundancy": lambda a, **k: ri.pl.redundancy(a, **k),
    "pathway_hierarchy": lambda a, **k: ri.pl.pathway_hierarchy(a, **k),
    "pattern_summary": lambda a, **k: ri.pl.pattern_summary(a, "celltype", **k),
    "pattern": lambda a, **k: ri.pl.pattern(a, 0, key="celltype", species="mouse", ntop=3, **k),
    "state_dist_pst": lambda a, **k: ri.pl.state_dist_pst(a, key="celltype", **k),
}
VELOCITY_PLOTS = {
    "plot_maps": lambda a, **k: ri.pl.plot_maps(a, _p(a), **k),
    "pattern_plot": lambda a, **k: ri.pl.pattern_plot(a, pattern=0, **k),
    "coarse_grained_map": lambda a, **k: ri.pl.coarse_grained_map(a, _p(a), key="celltype", **k),
    "expr_map": lambda a, **k: ri.pl.expr_map(a, _p(a), **k),
    "top_players_map": lambda a, **k: ri.pl.top_players_map(a, _p(a), top=2, **k),
    "sign_prob_plot": lambda a, **k: ri.pl.sign_prob_plot(a, _p(a), **k),
    "pattern_prob_plot": lambda a, **k: ri.pl.pattern_prob_plot(a, 0, **k),
}
SINGLE_PANEL = ["plot_mode_gap", "scatter2D", "single_pathway_heterogeneity", "pathway_heterogeneity_summary",
                "state_heterogeneity_summary", "heterogeneity_heatmap", "mode_composition", "redundancy",
                "pathway_hierarchy", "state_dist_pst", "pattern_plot", "coarse_grained_map", "sign_prob_plot",
                "pattern_prob_plot"]


def _plots(full):
    plots = dict(MATPLOTLIB_PLOTS)
    if full.uns["has_velocity"]:
        plots.update(VELOCITY_PLOTS)
    return plots


def test_all_public_plots_covered():
    covered = set(MATPLOTLIB_PLOTS) | set(VELOCITY_PLOTS) | {"alluvial_onepath", "alluvial_twopath",
                                                             "alluvial_pattern", "chord_diagram"}
    assert covered == set(ri.pl.__all__)


def test_show_false_returns_axes_and_return_fig(full):
    for name, f in _plots(full).items():
        out = f(full, show=False)
        assert isinstance(out, plt.Axes) or (isinstance(out, list) and all(isinstance(x, plt.Axes) for x in out)), name
        fig = f(full, show=False, return_fig=True)
        assert isinstance(fig, plt.Figure), name
        plt.close("all")


def test_ax_argument(full):
    for name in SINGLE_PANEL:
        if name not in _plots(full):
            continue
        fig, axs = plt.subplots(1, 2)
        out = _plots(full)[name](full, ax=axs[1], show=False)
        assert out is axs[1] or (isinstance(out, list) and axs[1] in out), name
        assert len(axs[1].get_children()) > len(axs[0].get_children()), name  # drawn into the given axes
        plt.close("all")


def test_autoshow_and_show(full, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(1))
    ri.settings.autoshow = True
    assert ri.pl.plot_mode_gap(full, _p(full)) is None and len(shown) == 1
    ri.settings.autoshow = False
    ri.pl.plot_mode_gap(full, _p(full))
    assert len(shown) == 1
    ri.pl.plot_mode_gap(full, _p(full), show=True)
    assert len(shown) == 2


def test_legacy_arguments(full, tmp_path):
    f = tmp_path / "gap.png"
    with pytest.warns(FutureWarning, match="deprecated"):
        ri.pl.plot_mode_gap(full, _p(full), savefig=True, showfig=False, figname=str(f), format="png")
    assert f.stat().st_size > 1000
    # figname without extension gets the format appended (e.g. plot_maps default 'velocity_plot')
    with pytest.warns(FutureWarning):
        ri.pl.redundancy(full, savefig=True, figname=str(tmp_path / "red"), format="png")
    assert (tmp_path / "red.png").exists()


def test_renamed_arguments(full):
    with pytest.warns(FutureWarning, match="annotation"):
        ri.pl.heatmap_one_pathway(full, _p(full), annotation="celltype", show=False)
    with pytest.warns(FutureWarning, match="legens_pos"):
        ri.pl.scatter2D(full, _p(full) + "_modes", legens_pos=(0.5, 1.1), show=False)
    with pytest.warns(FutureWarning, match="species"):
        ri.pl.pattern(full, 0, key="celltype", human=False, ntop=3, show=False)


def test_basis_argument(full):
    full.obsm["X_other"] = full.obsm["X_umap"] * 10
    ax = ri.pl.scatter2D(full, _p(full) + "_modes", basis="other", show=False)
    assert ax.get_xlim()[1] > 10


def test_no_global_style_change(full):
    before = plt.rcParams["axes.prop_cycle"]
    ri.pl.heatmap_one_pathway(full, _p(full), "celltype", show=False)
    assert plt.rcParams["axes.prop_cycle"] == before


def test_alluvial_returns_plotly_figure(full):
    pytest.importorskip("plotly")
    p1, p2 = _p(full, 0), _p(full, 1)
    for fig in [ri.pl.alluvial_onepath(full, p1, "celltype", show=False),
                ri.pl.alluvial_twopath(full, p1, p2, key="celltype", show=False),
                ri.pl.alluvial_pattern(full, "celltype", show=False)]:
        assert type(fig).__module__.startswith("plotly")


def test_velocity_plots_many_modes(full):
    if not full.uns["has_velocity"]:
        pytest.skip("scvelo not installed")
    b = full.copy()
    rng = np.random.default_rng(1)
    b.obs[_p(b) + "_modes"] = rng.integers(0, 7, b.n_obs)  # more modes than the former colour lists
    ri.pl.plot_maps(b, _p(b), show=False)
    ri.pl.sign_prob_plot(b, _p(b), show=False)
    with pytest.raises(ValueError, match="mode_to_plot"):
        ri.pl.plot_maps(b, _p(b), mode_to_plot=99, show=False)
    ax = ri.pl.plot_maps(b, _p(b), mode_to_plot=3, show=False)
    assert isinstance(ax, plt.Axes)
