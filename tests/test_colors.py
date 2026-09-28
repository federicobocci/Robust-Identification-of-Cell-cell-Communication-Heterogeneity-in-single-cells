import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import scrich as ri
from scrich.plots._colors import discrete_colors


@pytest.mark.parametrize("cmap", [plt.cm.Set2, plt.cm.tab20, "Set3", "viridis", ["red", "blue"]])
@pytest.mark.parametrize("n", [1, 5, 25, 80])
def test_discrete_colors_length_and_distinct(cmap, n):
    colors = discrete_colors(cmap, n)
    assert len(colors) == n
    assert all(len(c) == 3 for c in colors)
    assert len(set(colors)) == n


def test_palette_order_is_kept():
    assert discrete_colors(plt.cm.Set2, 3) == [tuple(c) for c in plt.cm.Set2.colors[:3]]


def test_plots_with_many_patterns(adata_mouse):
    """Plots must not fail (or drop categories) with more patterns/cell types than palette colours."""
    rng = np.random.default_rng(1)
    adata = adata_mouse
    adata.obs["sign_pattern"] = pd.Categorical(rng.integers(0, 30, adata.n_obs).astype(str))
    adata.obs["many_modes"] = pd.Categorical(rng.integers(0, 12, adata.n_obs).astype(str))
    adata.obsm["X_umap"] = rng.normal(size=(adata.n_obs, 2))
    ri.pl.pattern_summary(adata, "celltype", panel_per_row=5, savefig=False, showfig=False)
    ri.pl.scatter2D(adata, "many_modes", savefig=False, showfig=False)
    plt.close("all")
