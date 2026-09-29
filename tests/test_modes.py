import anndata as ad
import numpy as np
import pandas as pd

import scrich as ri
from scrich.tools.mode_clustering_func import cci_heterogeneity
from scrich.tools.sign_patterns import _mode_codes


def _with_fraction(frac):
    a = ad.AnnData(np.zeros((1, 1)))
    a.uns["fraction_mat"] = {"P": np.asarray(frac, dtype=float)}
    a.uns["pathways"] = {"P": {}}
    cci_heterogeneity(a, "P")
    return a.uns["pathways"]["P"]["cci_heterogeneity"]


def test_heterogeneity_limits():
    # rows = modes, columns = cell types
    assert np.allclose(_with_fraction([[10, 10], [10, 10]]), 1.0)  # same composition as population
    assert np.allclose(_with_fraction([[20, 0], [0, 20]]), 0.0)  # each type entirely in one mode
    v = _with_fraction([[15, 5], [5, 15]])
    assert np.all((v > 0) & (v < 1))


def _obs(values):
    return pd.DataFrame({"P_modes": values}, index=[f"c{i}" for i in range(len(values))])


def test_rename_modes_keeps_codes():
    a = ad.AnnData(np.zeros((6, 1)), obs=_obs([0, 1, 2, 2, 1, 0]))
    before = _mode_codes(a.obs["P_modes"])
    ri.tl.rename_modes(a, "P", {0: "Receiver", 1: "Inactive", 2: "Sender/Receiver"})
    assert list(a.obs["P_modes"]) == ["Receiver", "Inactive", "Sender/Receiver", "Sender/Receiver", "Inactive", "Receiver"]
    assert (_mode_codes(a.obs["P_modes"]) == before).all()
    ri.tl.rename_modes(a, "P", {"Inactive": "Off"})  # rename again
    assert (_mode_codes(a.obs["P_modes"]) == before).all()


def test_rename_modes_partial_and_merge():
    a = ad.AnnData(np.zeros((4, 1)), obs=_obs([0, 1, 2, 1]))
    ri.tl.rename_modes(a, "P", {0: "Weak"})
    assert list(a.obs["P_modes"]) == ["Weak", "1", "2", "1"]  # no mixed int/str labels
    ri.tl.rename_modes(a, "P", {"1": "Active", "2": "Active"})
    assert list(a.obs["P_modes"].cat.categories) == ["Weak", "Active"]


def test_mode_codes_plain_strings():
    assert list(_mode_codes(pd.Series(["b", "a", "b"]))) == [1.0, 0.0, 1.0]
