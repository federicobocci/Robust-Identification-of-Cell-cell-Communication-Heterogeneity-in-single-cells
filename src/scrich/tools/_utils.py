"""Internal helpers shared by the analysis tools and plots."""

import contextlib
import warnings

import numpy as np
import pandas as pd


def list_intersect(geneset,
                   lst
                   ):
    '''selects the elements in common between the two lists

    Parameters
    ----------
    geneset: list of genes in dataset
    lst: list of genes of interested

    Returns
    -------
    gene_list: list of common genes between the two lists

    '''
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
    '''
    return the list of cell neighbors for all cells in adata

    Parameters
    ----------
    adata: anndata object of gene counts
    key: distance metric (Please choose between key=='distances' or key=='connectivities')

    Returns
    -------
    neighbors: the list of neighbors per cell

    '''
    if key not in ('distances', 'connectivities'):
        raise ValueError(f"key must be 'distances' or 'connectivities', got {key!r}")
    dist = adata.obsp[key]
    neighbors = [ dist[i].indices for i in range(adata.shape[0]) ]
    return neighbors


def neighbor_avg(v, neigh):
    '''
    Compute average of array v over cell neighbors

    Parameters
    ----------
    v: array to be averaged
    neigh: list of neighbors

    Returns
    -------
    v_avg: the averge value

    '''
    v_avg = np.zeros(v.size)
    for i in range(v_avg.size):
        neigh_v = v[neigh[i]]
        v_avg[i] = np.mean(neigh_v)
    return v_avg


def _mode_order(col):
    '''Modes of a pathway in a stable order: category order for renamed modes, otherwise sorted.'''
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
