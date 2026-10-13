"""Seeds, logging, species handling, input checks and the memory-efficient similarity matrix."""


import anndata as ad
import numpy as np
import pytest

import scrich as ri
from scrich.tools.modes import _coassociation_factor, spectral_analysis_opt


def _run(adata):
    ri.tl.pathways_overview(adata, species="mouse")
    ri.tl.all_path_sim(adata, "celltype", pathways_to_use=2, species="mouse", neighbor_average=True)


def test_permutation_test_seeded(adata_mouse):
    _run(adata_mouse)
    ri.tl.permutation_testing(adata_mouse, key="celltype", seed=3)
    first = dict(adata_mouse.uns["permutation_test"][1])
    np.random.seed(123)  # global state must not matter
    ri.tl.permutation_testing(adata_mouse, key="celltype", seed=3)
    for p, v in first.items():
        assert adata_mouse.uns["permutation_test"][1][p] == v
    # seed=None keeps the old behaviour (global NumPy random state)
    np.random.seed(3)
    ri.tl.permutation_testing(adata_mouse, key="celltype", seed=None)
    for p, v in first.items():
        assert adata_mouse.uns["permutation_test"][1][p] == v


def test_human_argument_deprecated_but_working(adata_mouse):
    a, b = adata_mouse.copy(), adata_mouse.copy()
    with pytest.warns(FutureWarning, match="species"):
        ri.tl.pathways_overview(a, human=False)
    ri.tl.pathways_overview(b, species="mouse")
    assert a.uns["paths_overview"]["gene_number"] == b.uns["paths_overview"]["gene_number"]
    with pytest.warns(FutureWarning):
        ri.tl.find_genes(a, "IGF", False, print_info=False)  # old positional human flag
    with pytest.raises(ValueError, match="Unsupported species"):
        ri.tl.pathways_overview(a, species="fish")


def test_factorized_similarity_matches_dense():
    rng = np.random.default_rng(0)
    dat = np.c_[rng.gamma(2, 1, 600) + np.repeat([0, 3, 6], 200), rng.gamma(2, 1, 600)]
    F = _coassociation_factor(dat, print_info=False)
    assert F.shape == (600, 5 * (3 + 4 + 5 + 6))
    a = ad.AnnData(np.zeros((600, 1)))
    spectral_analysis_opt(a, F @ F.T, "P", print_info=False)
    dense = a.uns["SymNMF"]["P"]
    spectral_analysis_opt(a, F, "P", print_info=False, is_factor=True)
    fac = a.uns["SymNMF"]["P"]
    np.testing.assert_allclose(np.sort(dense["eigenvalues"]), fac["eigenvalues"], atol=1e-10)
    assert dense["optimal"] == fac["optimal"]


def test_logging_verbosity(adata_mouse, capsys):
    ri.set_verbosity("warning")
    try:
        ri.tl.pathways_overview(adata_mouse, species="mouse")
        assert "Scanning" not in capsys.readouterr().out
    finally:
        ri.set_verbosity("info")
    ri.tl.pathways_overview(adata_mouse, species="mouse")
    assert "Scanning" in capsys.readouterr().out
    with pytest.raises(ValueError):
        ri.set_verbosity("loud")


def test_no_global_warning_filters():
    import inspect

    import scrich.tools

    for mod in (scrich.tools.modes, scrich.tools.patterns, scrich.tools.heterogeneity, scrich.tools.pathways):
        assert "warnings.simplefilter(" not in inspect.getsource(mod)


def test_input_errors(adata_mouse):
    _run(adata_mouse)
    p = adata_mouse.uns["selected_pathways"][0]
    with pytest.raises(ValueError, match="model"):
        ri.tl.compute_ccc_matrix(adata_mouse, p, key="celltype", model="foo")
    with pytest.raises(KeyError, match="velocity"):
        ri.tl.compute_ccc_matrix(adata_mouse, p, key="celltype", use_velocity=True)
    with pytest.raises(ValueError, match="method"):
        ri.tl.select_pathways(adata_mouse, method="foo")
    with pytest.raises(KeyError, match="pathways_overview"):
        ri.tl.import_database(adata_mouse, ["NOT_A_PATHWAY"], species="mouse")
