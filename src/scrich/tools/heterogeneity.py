"""Cell-type heterogeneity of communication modes and permutation testing."""

import warnings

import numpy as np

from ._utils import _mode_order


def state_fractions(adata, pathway, annotations, order=None):
    r"""Number of cells in each communication mode (rows) and cell type (columns) of a pathway.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with ``adata.obs[pathway + '_modes']``.
    pathway : str
        Pathway of interest.
    annotations : str
        Key in ``adata.obs`` with cell-type annotations.
    order : list, optional
        Deprecated, has no effect (cell types are always stored in sorted order).

    Returns
    -------
    numpy.ndarray
        Modes x cell types matrix of cell counts. Its row and column labels are stored in
        ``adata.uns['pathways'][pathway]['modes']`` and ``['cell_types']``.
    """
    if order is not None:
        warnings.warn("The 'order' argument no longer changes how results are stored: cell types are always "
                      "stored in sorted order (see adata.uns['pathways'][pathway]['cell_types']). "
                      "Use the 'order' argument of the plotting functions to change the display order.",
                      FutureWarning, stacklevel=3)
    labs = np.asarray(list(adata.obs[pathway + '_modes']))
    lab_set = _mode_order(adata.obs[pathway + '_modes'])
    clusters = np.asarray(list(adata.obs[annotations]))
    clst_set = sorted(list(set(clusters)))

    ntypes, nclst = len(lab_set), len(clst_set)
    frac = np.zeros((ntypes, nclst))

    for i in range(ntypes):
        type1 = clusters[labs == lab_set[i]]
        for j in range(nclst):
            frac[i][j] = len(type1[type1 == clst_set[j]])

    # record the row (mode) and column (cell type) labels of the fraction matrix
    adata.uns['pathways'].setdefault(pathway, {})
    adata.uns['pathways'][pathway]['modes'] = list(lab_set)
    adata.uns['pathways'][pathway]['cell_types'] = list(clst_set)
    return frac


def cci_heterogeneity(adata, pathway):
    r"""Heterogeneity score of every cell type for a pathway.

    See the Notes of :func:`scrich.tl.consensus_clustering` for the definition.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with ``adata.uns['fraction_mat'][pathway]``.
    pathway : str
        Pathway of interest.

    Returns
    -------
    None
        Sets ``adata.uns['pathways'][pathway]['cci_heterogeneity']`` (one value per cell type, in the order of
        ``adata.uns['pathways'][pathway]['cell_types']``).
    """
    frac = adata.uns['fraction_mat'][pathway]
    n_mode, n_clst = frac.shape
    mode_density = np.sum(frac, axis=1) / np.sum(frac)

    v = np.zeros(n_clst)

    for j in range(n_clst):
        rho = frac[:, j] / np.sum(frac[:, j])
        for i in range(rho.size):
            if rho[i] - mode_density[i] > 0:
                s = 1 - mode_density[i]
            else:
                s = mode_density[i]
            v[j] = v[j] + np.abs(rho[i] - mode_density[i]) / s
    v = 1 - v / mode_density.size

    adata.uns['pathways'][pathway]['cci_heterogeneity'] = v


def permutation_testing(adata, key='seurat_clusters', n_sample = 10000, perm_threshold=0.05, seed=0):
    r"""Test which cell types are significantly heterogeneous in their communication modes.

    For every selected pathway and cell type, the number of cells in the cell type's most frequent mode,
    :math:`n_{max}`, is compared with the cell type's size :math:`n`: the p-value is the fraction of
    ``n_sample`` Poisson draws with mean :math:`n_{max}` that exceed :math:`n`.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.all_path_sim`.
    key : str, default 'seurat_clusters'
        Key in ``adata.obs`` with cell-type annotations (the same as used in :func:`scrich.tl.all_path_sim`).
    n_sample : int, default 10000
        Number of random draws.
    perm_threshold : float, default 0.05
        Significance threshold.
    seed : int or None, default 0
        Seed of the random number generator; ``None`` uses NumPy's global random state (scRICH 0.1 behaviour).

    Returns
    -------
    None
        Sets ``adata.uns['permutation_test'] = [cell_types, p_values, significant_pathways]``, where
        ``p_values`` maps each pathway to one p-value per cell type (in the order of ``cell_types``) and
        ``significant_pathways`` lists the pathways with at least one cell type below ``perm_threshold``.

    See Also
    --------
    scrich.pl.single_pathway_heterogeneity : Heterogeneity scores with significant cell types marked.
    """
    rng = np.random if seed is None else np.random.RandomState(seed)
    perm_dict = {}
    pathways = adata.uns['selected_pathways']
    types = sorted(list(set(adata.obs[key])))

    pos_pathways = []

    for p in pathways:
        frac = adata.uns['fraction_mat'][p]

        ncells_per_type = np.sum(frac, axis=0)
        mu, ratio = [], []

        for i in range(len(types)):
            ncell = ncells_per_type[i]
            major_mode = np.amax(frac[:, i])
            test = rng.poisson(major_mode, n_sample)
            ratio.append(float(len(test[test > ncell])) / len(test))

            mu.append(major_mode / ncell)
        perm_dict[p] = ratio

        if np.any(np.asarray(ratio)<perm_threshold):
            pos_pathways.append(p)

    adata.uns['permutation_test'] = [types, perm_dict, pos_pathways]
