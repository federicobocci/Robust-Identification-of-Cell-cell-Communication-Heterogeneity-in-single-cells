"""Internal helpers shared by the analysis tools and plots."""

import contextlib
import warnings

import numpy as np
import pandas as pd


def list_intersect(geneset,
                   lst
                   ):
    r"""Unique elements of ``lst`` that are also in ``geneset``.

    Parameters
    ----------
    geneset : list of str
        Genes in the dataset.
    lst : list of str
        Genes of interest.

    Returns
    -------
    list of str
        Common genes.
    """
    if not (isinstance(geneset, list) and isinstance(lst, list)):
        raise TypeError('geneset and lst must be Python lists')

    gene_list = []
    for l in lst:
        if l in geneset:
            gene_list.append(l)
    return list(set(gene_list))


def get_list(df, pathway):
    geneset = list(df[ df['pathway']==pathway ].iloc[:,1])
    # print(geneset)
    new_set = []
    for g in geneset:
        if '_' in g:
            i = g.index('_')
            a, b = g[0:i], g[i + 1:]
            new_set = new_set + [a, b]
            # print( g, g[0:i], g[i+1:] )
        else:
            new_set.append(g)
    return new_set


def extract_neighbors(adata, key='distances'):
    r"""Neighbours of every cell from a neighbour graph.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with ``adata.obsp[key]``.
    key : {'distances', 'connectivities'}, default 'distances'
        Neighbour graph.

    Returns
    -------
    list of numpy.ndarray
        Indices of the neighbours of each cell.
    """
    if key not in ('distances', 'connectivities'):
        raise ValueError(f"key must be 'distances' or 'connectivities', got {key!r}")
    dist = adata.obsp[key]
    neighbors = [ dist[i].indices for i in range(adata.shape[0]) ]
    return neighbors


def neighbor_avg(v, neigh):
    r"""Average of a per-cell value over each cell's neighbours.

    Parameters
    ----------
    v : numpy.ndarray
        Value of every cell.
    neigh : list of numpy.ndarray
        Neighbours of every cell (:func:`extract_neighbors`).

    Returns
    -------
    numpy.ndarray
        Neighbour average of every cell.
    """
    v_avg = np.zeros(v.size)
    for i in range(v_avg.size):
        neigh_v = v[neigh[i]]
        v_avg[i] = np.mean(neigh_v)
    return v_avg


def _mode_order(col):
    """Modes of a pathway in a stable order: category order for renamed modes, otherwise sorted."""
    if isinstance(col.dtype, pd.CategoricalDtype):
        return [c for c in col.cat.categories if (col == c).any()]
    return sorted(set(np.asarray(col)))


def _mode_codes(col):
    """Integer codes of the communication modes of one pathway.

    Integer labels (as produced by the clustering) are used as they are. Modes renamed with
    rename_modes() are categorical and keep their original order, so they give the same codes
    as before renaming. Other labels (e.g. strings) are numbered in sorted order.
    """
    if isinstance(col.dtype, pd.CategoricalDtype):
        return np.asarray(col.cat.codes, dtype=float)
    if pd.api.types.is_numeric_dtype(col):
        return np.asarray(col, dtype=float)
    return pd.Categorical(col, categories=sorted(pd.unique(col))).codes.astype(float)


def _resolve_species(species=None, human=None):
    """Species name from the new ``species`` argument or the deprecated ``human`` flag.

    ``species`` given as True/False (e.g. an old positional ``human`` argument) is treated as ``human``.
    """
    from ..db import check_species

    if isinstance(species, (bool, np.bool_)):
        human, species = bool(species), None
    if human is not None:
        warnings.warn("The 'human' argument is deprecated; use species='human' or species='mouse' instead.",
                      FutureWarning, stacklevel=3)
        if species is None:
            species = 'human' if human else 'mouse'
    return check_species(species if species is not None else 'human')


@contextlib.contextmanager
def _quiet_pandas():
    """Silence pandas' DataFrame-fragmentation PerformanceWarning (caused by adding many adata.obs columns)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=pd.errors.PerformanceWarning)
        yield


def compute_sign_prob(adata, pathway, key='distances'):
    r"""Fraction of each cell's neighbours in each communication mode of a pathway.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with ``adata.obs[pathway + '_modes']``.
    pathway : str
        Pathway of interest.
    key : {'distances', 'connectivities'}, default 'distances'
        Neighbour graph in ``adata.obsp``.

    Returns
    -------
    tuple
        Modes (in stored order) and a cells x modes array of fractions.
    """
    neighbors = extract_neighbors(adata, key=key)
    modes = np.asarray(adata.obs[pathway + '_modes'])

    unique_modes = _mode_order(adata.obs[pathway + '_modes'])
    sign_prob = np.zeros((modes.size, len(unique_modes)))
    for i in range(modes.size):
        neigh_modes = modes[neighbors[i]]
        counts = np.asarray([neigh_modes[neigh_modes == m].size for m in unique_modes])
        counts = counts / np.sum(counts)
        sign_prob[i] = counts
    return unique_modes, sign_prob


def compute_pattern_prob(adata, key='distances'):
    r"""Fraction of each cell's neighbours in each signaling pattern.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with ``adata.obs['sign_pattern']``.
    key : {'distances', 'connectivities'}, default 'distances'
        Neighbour graph in ``adata.obsp``.

    Returns
    -------
    tuple
        Patterns (in numerical order) and a cells x patterns array of fractions.
    """
    neighbors = extract_neighbors(adata, key=key)
    modes = np.asarray(adata.obs['sign_pattern'])

    unique_modes = _mode_order(adata.obs['sign_pattern'])  # numeric pattern order (not '0', '1', '10', '2', ...)
    sign_prob = np.zeros((modes.size, len(unique_modes)))
    for i in range(modes.size):
        neigh_modes = modes[neighbors[i]]
        counts = np.asarray([neigh_modes[neigh_modes == m].size for m in unique_modes])
        counts = counts / np.sum(counts)
        sign_prob[i] = counts
    return unique_modes, sign_prob
