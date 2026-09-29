"""Cell-type heterogeneity of communication modes and permutation testing."""

import warnings

import numpy as np

from ._utils import _mode_order


def state_fractions(adata, pathway, annotations, order=None):
    '''Compute a matrix of cell fractions between cell annotations and communication modes of a pathway

    Parameters
    ----------
    adata: scRNA-seq data object
    pathway: the pathway to select communication modes
    annotations: keys for cell annotations in adata.obs
    order: deprecated, has no effect (cell types are always stored in sorted order)

    Returns
    -------
    frac: the cell fraction matrix (modes x cell types). Row and column labels are stored in
        adata.uns['pathways'][pathway]['modes'] and ['cell_types'].

    '''
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
    '''Compute the ccc heterogeneity score for a given pathway, saved in adata.uns['pathways'][pathway]['cci_heterogeneity']

    For each cell type, the distribution of its cells across communication modes is compared with the
    distribution of all cells. The score is 1 minus the average normalized deviation between the two:
    it is close to 1 when the cell type contains the modes in the same proportions as the whole
    population (heterogeneous), and close to 0 when its mode composition deviates maximally from it
    (e.g. all cells of the type in a single, otherwise rare, mode).
    Versions up to 0.1.0 stored the deviation itself (i.e. 1 - score).

    Parameters
    ----------
    adata: scRNA-seq data object
    pathway: CCC pathway of interest

    Returns
    -------
    None

    '''
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
    '''Compute permutation testing to identify statistically significant heterogeneous cell types.
    Results are saved in adata.uns['permutation_test']

    Parameters
    ----------
    adata: scRNA-seq data object
    key: adata.obs key for cell annotations
    n_sample: number of permutation tests
    perm_threshold: p-value threshold for statistical significance
    seed: seed of the random number generator, for reproducible p-values (default=0).
        None uses NumPy's global random state (behaviour of versions up to 0.1.0).

    Returns
    -------
    None

    '''
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
