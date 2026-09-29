import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import scrich as ri
from scrich.tools.targets import get_counts


def _modes(adata, **kw):
    ri.tl.pathways_overview(adata, species="mouse")
    ri.tl.all_path_sim(adata, "celltype", pathways_to_use=2, target=False, species="mouse", neighbor_average=True, **kw)
    return adata.uns["selected_pathways"]


def test_order_does_not_change_stored_results(adata_mouse):
    a, b = adata_mouse.copy(), adata_mouse.copy()
    paths = _modes(a)
    with pytest.warns(FutureWarning, match="order"):
        _modes(b, order=["T2", "T1", "T0"])
    for p in paths:
        assert a.uns["pathways"][p]["cell_types"] == ["T0", "T1", "T2"]
        np.testing.assert_array_equal(a.uns["fraction_mat"][p], b.uns["fraction_mat"][p])
        np.testing.assert_allclose(a.uns["pathways"][p]["cci_heterogeneity"], b.uns["pathways"][p]["cci_heterogeneity"])
        # fraction matrix columns match the stored cell types
        col = list(a.obs["celltype"]).count("T1")
        assert a.uns["fraction_mat"][p][:, 1].sum() == col


def test_plot_order_reorders_columns(adata_mouse):
    from scrich.plots._utils import _fraction_matrix

    p = _modes(adata_mouse)[0]
    frac, modes, types = _fraction_matrix(adata_mouse, p, "celltype", order=["T2", "T0", "T1"])
    full = adata_mouse.uns["fraction_mat"][p]
    np.testing.assert_array_equal(frac, full[:, [2, 0, 1]])
    assert types == ["T2", "T0", "T1"]
    with pytest.raises(ValueError, match="unknown cell types"):
        _fraction_matrix(adata_mouse, p, "celltype", order=["T9"])


@pytest.mark.parametrize("sparse", [True, False])
def test_get_counts_per_gene(adata_mouse, sparse):
    a = adata_mouse
    X = a.X.toarray()
    a.X = sp.csr_matrix(X) if sparse else X
    df = get_counts(a)
    np.testing.assert_allclose(df["avg_count"].to_numpy(), X.mean(axis=0), rtol=1e-5)
    assert df["avg_count"].nunique() > 1
    df_u = get_counts(a, unspliced=True)
    np.testing.assert_allclose(df_u["avg_count"].to_numpy(), (a.layers["Mu"] + a.layers["Ms"]).mean(axis=0), rtol=1e-5)


def test_target_uses_tf_genes(adata_mouse):
    ri.tl.pathways_overview(adata_mouse, species="mouse")
    p = "IGF" if "IGF" in adata_mouse.uns["pathways"] else list(adata_mouse.uns["pathways"])[0]
    with pytest.raises(ValueError, match="import_database"):
        ri.tl.single_path_sim(adata_mouse, p, target=True, print_info=False)
    adata_mouse.uns["TF"] = {p: list(adata_mouse.var_names[:5])}
    sim = ri.tl.single_path_sim(adata_mouse, p, target=True, neighbor_average=True, print_info=False)
    assert sim.shape == (adata_mouse.n_obs, adata_mouse.n_obs)
    assert p + "_tar" in adata_mouse.obs


@pytest.mark.network
def test_all_path_sim_with_targets(adata_mouse, exfinder_dir):
    import anndata as ad

    # add expressed downstream TFs of the mouse receptors to the synthetic dataset
    rec = ri.db.load_pathway_genes("receptor", "mouse")["receptor"].str.split("_").explode()
    rtf = ri.db.load_exfinder("rtf", "mouse")
    tfs = [g for g in rtf.loc[rtf["from"].isin(set(rec)), "to"].astype(str).unique() if g not in adata_mouse.var_names][:200]
    rng = np.random.default_rng(0)
    extra = rng.poisson(2.0, (adata_mouse.n_obs, len(tfs))).astype(np.float32)
    a = ad.concat([adata_mouse, ad.AnnData(sp.csr_matrix(extra), obs=adata_mouse.obs[[]],
                  var=__import__("pandas").DataFrame(index=tfs),
                  layers={"Ms": extra, "Mu": extra * 0.5})], axis=1, merge="same")
    a.obs = adata_mouse.obs.copy()
    a.obsp = adata_mouse.obsp.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ri.tl.pathways_overview(a, species="mouse")
        ri.tl.all_path_sim(a, "celltype", pathways_to_use=3, target=True, species="mouse", top_target=5)
    paths = a.uns["selected_pathways"]
    assert any(len(a.uns["TF"][p]) > 0 for p in paths)
    for p in paths:
        assert p + "_modes" in a.obs
        if len(a.uns["TF"][p]) > 0:
            assert p + "_tar" in a.obs


def test_consensus_clustering_uses_targets(adata_mouse):
    from scrich.tools.modes import consensus_clustering

    ri.tl.pathways_overview(adata_mouse, species="mouse")
    p = list(adata_mouse.uns["pathways"])[0]
    ri.tl.single_path_sim(adata_mouse, p, print_info=False)
    # a target signal with a clear two-group split that dominates ligand/receptor expression
    split = (np.arange(adata_mouse.n_obs) % 2).astype(bool)
    adata_mouse.obs[p + "_tar"] = np.where(split, 100.0, 0.0)
    consensus_clustering(adata_mouse, p, "celltype", n_cluster=2, target=True)
    modes = np.asarray(adata_mouse.obs[p + "_modes"])
    assert len(set(modes[split])) == 1 and len(set(modes[~split])) == 1

    del adata_mouse.obs[p + "_tar"]
    with pytest.raises(ValueError, match="_tar"):
        consensus_clustering(adata_mouse, p, "celltype", n_cluster=2, target=True)


def test_target_default_is_false():
    import inspect

    from scrich.tools.modes import consensus_clustering

    for f in (ri.tl.all_path_sim, ri.tl.single_path_sim, consensus_clustering):
        assert inspect.signature(f).parameters["target"].default is False


def test_consensus_clustering_neighbor_average(adata_mouse):
    from scrich.tools._utils import extract_neighbors, neighbor_avg
    from scrich.tools.modes import consensus_clustering

    ri.tl.pathways_overview(adata_mouse, species="mouse")
    p = list(adata_mouse.uns["pathways"])[0]
    ri.tl.single_path_sim(adata_mouse, p, print_info=False)
    rec_before = np.asarray(adata_mouse.obs[p + "_rec"]).copy()
    # smoothed ligand signal with a clear split that dominates; unsmoothed it is noise around it
    neigh = extract_neighbors(adata_mouse)
    rng = np.random.default_rng(0)
    group = np.asarray(adata_mouse.obs["celltype"]) == "T0"
    adata_mouse.obs[p + "_lig"] = np.where(group, 50.0, 0.0) + rng.normal(0, 200, adata_mouse.n_obs)
    adata_mouse.obs[p + "_rec"] = 0.0
    consensus_clustering(adata_mouse, p, "celltype", n_cluster=2, neighbor_average=True)
    modes = np.asarray(adata_mouse.obs[p + "_modes"])
    smoothed = neighbor_avg(np.asarray(adata_mouse.obs[p + "_lig"]), neigh)
    # modes follow the smoothed signal
    assert abs(np.corrcoef(modes, smoothed > np.median(smoothed))[0, 1]) > 0.9
    assert adata_mouse.uns["pathways"][p]["neighbor_average"] is True
    assert np.all(np.asarray(adata_mouse.obs[p + "_rec"]) == 0.0)  # obs values not overwritten
