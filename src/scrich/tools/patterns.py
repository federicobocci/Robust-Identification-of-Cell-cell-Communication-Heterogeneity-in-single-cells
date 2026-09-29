"""Signaling patterns across pathways and similarity (mutual information) between pathways."""

import warnings

import anndata
import numpy as np
import scanpy as sc
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import mutual_info_score

from .._logging import logger
from ._utils import _mode_codes, _quiet_pandas


def find_sign_patterns(adata, paths='all', res=1, n_neighbors=30, random_state=0, seed=42, print_info=True):
    '''Identify CCC patterns across several pathways by clustering cells based on the CCC modes

    Cells are described by their communication mode in each pathway. A k-nearest-neighbour
    graph is built on these profiles (after PCA) and clustered with Leiden.

    Because many cells share exactly the same combination of modes, a small ``n_neighbors``
    can split the graph into disconnected pieces. Leiden cannot merge disconnected pieces,
    so their number is a lower bound on the number of patterns, whatever the resolution.
    The number of pieces is reported and a warning is raised when there is more than one.

    Parameters
    ----------
    adata: scRNA-seq data object
    paths: list of pathways to use in the clustering (default: 'all', i.e. adata.uns['selected_pathways'])
    res: Leiden clustering resolution (lower values give fewer, larger patterns)
    n_neighbors: number of nearest neighbours in the cell graph (default=30). Increase it if the graph
        is split into several pieces. Versions up to 0.1.0 used a fixed value of 10.
    random_state: seed for the neighbour search, UMAP and Leiden, for reproducible results (default=0)
    seed: seed for NumPy's global random generator (kept for backward compatibility)
    print_info: if True, print the number of patterns found and the number of cells in each

    Returns
    -------
    None. Patterns are stored in adata.obs['sign_pattern'], marker genes in adata.uns['rank_genes_groups']
    and the parameters used in adata.uns['sign_pattern_params'].

    '''
    np.random.seed(seed)
    if paths=='all':
        paths = adata.uns['selected_pathways']
    ncell, npath = adata.shape[0], len(paths)

    sel_data = np.zeros((npath, ncell))
    for i in range(len(paths)):
        sel_data[i] = _mode_codes(adata.obs[paths[i] + '_modes'])

    path_adata = anndata.AnnData(X=sel_data.transpose())
    path_adata.obs_names = adata.obs_names
    path_adata.var_names = paths

    sc.tl.pca(path_adata, svd_solver='arpack', random_state=random_state)
    sc.pp.neighbors(path_adata, n_neighbors=n_neighbors, n_pcs=min(len(paths), path_adata.obsm['X_pca'].shape[1]),
                    random_state=random_state)
    sc.tl.umap(path_adata, random_state=random_state)
    # flavor='leidenalg' is scanpy's current default (same results as before); passing it explicitly avoids
    # scanpy's FutureWarning about a future change of default, and its hint about the igraph flavor is silenced
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*igraph.*', category=UserWarning)
        sc.tl.leiden(path_adata, resolution=res, random_state=random_state, flavor='leidenalg')

    with _quiet_pandas():
        adata.obs['sign_pattern'] = path_adata.obs['leiden']

    # disconnected pieces of the graph set a minimum number of patterns
    n_pieces = connected_components(path_adata.obsp['connectivities'], directed=False)[0]
    adata.uns['sign_pattern_params'] = {'paths': list(paths), 'res': res, 'n_neighbors': n_neighbors,
                                        'random_state': random_state, 'n_graph_components': int(n_pieces)}

    if print_info:
        counts = adata.obs['sign_pattern'].value_counts()
        counts = counts.reindex(sorted(counts.index, key=int))
        logger.info(f"Identified {counts.size} signaling patterns at resolution res={res} "
                    f"(clustering {len(paths)} pathways, n_neighbors={n_neighbors}).")
        logger.info('Cells per pattern: ' + ', '.join(f'{k}: {v}' for k, v in counts.items()))
        if counts.size > 20 and n_pieces <= 1:
            logger.info('Many patterns were found; a lower resolution (res) gives fewer, larger patterns.')

    if n_pieces > 1:
        warnings.warn(
            f"The cell graph used to find signaling patterns is split into {n_pieces} disconnected pieces, "
            f"so there will be at least {n_pieces} patterns regardless of the resolution. "
            f"Consider increasing n_neighbors (currently {n_neighbors}).",
            UserWarning,
            stacklevel=2,
        )

    with _quiet_pandas():
        sc.tl.rank_genes_groups(adata, 'sign_pattern')


def twopath_mutual_info(adata, path1, path2):
    '''Compute mutual information between CCC modes of two pathways

    Parameters
    ----------
    adata: scRNA-seq data object
    path1: first pathway
    path2: second pathway

    Returns
    -------
    mi_skl: mutual information

    '''
    lab1 = np.asarray(adata.obs[path1 + '_modes'])
    lab2 = np.asarray(adata.obs[path2 + '_modes'])
    mi_skl = mutual_info_score(lab1, lab2)
    return mi_skl


def pairwise_MI(adata):
    '''Compute pairwise mutual information between all pairs of CCC pathways.
    Results are saved as 2D array in adata.uns['CCC_similarity_map']

    Parameters
    ----------
    adata: scRNA-seq data object

    Returns
    -------
    None

    '''
    pathways = adata.uns['selected_pathways']

    sim_mat = np.zeros((len(pathways), len(pathways)))

    for i in range(len(pathways)):
        for j in range(len(pathways)):
            sim_mat[i][j] = twopath_mutual_info(adata, pathways[i], pathways[j])
    adata.uns['CCC_similarity_map'] = sim_mat
