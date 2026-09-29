import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import scrich as ri


@pytest.fixture
def adata_modes():
    """Cells with discrete modes in 6 pathways; many cells share identical mode combinations."""
    rng = np.random.default_rng(0)
    ncell, npath = 1200, 6
    base = rng.integers(0, 3, (6, npath))
    M = base[rng.integers(0, 6, ncell)].copy()
    mut = rng.random(ncell) < 0.2
    M[mut, rng.integers(0, npath, mut.sum())] = rng.integers(0, 3, mut.sum())
    adata = ad.AnnData(X=rng.poisson(1.0, (ncell, 20)).astype(float),
                       obs=pd.DataFrame(index=[f"c{i}" for i in range(ncell)]))
    paths = [f"P{i}" for i in range(npath)]
    for i, p in enumerate(paths):
        adata.obs[p + "_modes"] = M[:, i]
    adata.uns["selected_pathways"] = paths
    return adata


def test_params_stored_and_reproducible(adata_modes):
    ri.tl.find_sign_patterns(adata_modes, res=0.1, print_info=False)
    params = adata_modes.uns["sign_pattern_params"]
    assert params["n_neighbors"] == 30 and params["res"] == 0.1 and params["random_state"] == 0
    first = np.asarray(adata_modes.obs["sign_pattern"]).copy()
    ri.tl.find_sign_patterns(adata_modes, res=0.1, print_info=False)
    assert (np.asarray(adata_modes.obs["sign_pattern"]) == first).all()


def test_warns_when_graph_is_disconnected(adata_modes):
    with pytest.warns(UserWarning, match="disconnected pieces"):
        ri.tl.find_sign_patterns(adata_modes, res=0.1, n_neighbors=3, print_info=False)
    n_pieces = adata_modes.uns["sign_pattern_params"]["n_graph_components"]
    assert n_pieces > 1
    assert adata_modes.obs["sign_pattern"].nunique() >= n_pieces


def test_more_neighbours_fewer_pieces(adata_modes):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ri.tl.find_sign_patterns(adata_modes, res=0.1, n_neighbors=3, print_info=False)
        small = adata_modes.uns["sign_pattern_params"]["n_graph_components"]
        ri.tl.find_sign_patterns(adata_modes, res=0.1, n_neighbors=100, print_info=False)
        large = adata_modes.uns["sign_pattern_params"]["n_graph_components"]
    assert large < small
