import warnings

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse.csgraph import connected_components


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
    sc.tl.leiden(path_adata, resolution=res, random_state=random_state)

    adata.obs['sign_pattern'] = path_adata.obs['leiden']

    # disconnected pieces of the graph set a minimum number of patterns
    n_pieces = connected_components(path_adata.obsp['connectivities'], directed=False)[0]
    adata.uns['sign_pattern_params'] = {'paths': list(paths), 'res': res, 'n_neighbors': n_neighbors,
                                        'random_state': random_state, 'n_graph_components': int(n_pieces)}

    if print_info:
        counts = adata.obs['sign_pattern'].value_counts()
        counts = counts.reindex(sorted(counts.index, key=int))
        print(f"Identified {counts.size} signaling patterns at resolution res={res} "
              f"(clustering {len(paths)} pathways, n_neighbors={n_neighbors}).")
        print('Cells per pattern: ' + ', '.join(f'{k}: {v}' for k, v in counts.items()))
        if counts.size > 20 and n_pieces <= 1:
            print('Many patterns were found; a lower resolution (res) gives fewer, larger patterns.')

    if n_pieces > 1:
        warnings.warn(
            f"The cell graph used to find signaling patterns is split into {n_pieces} disconnected pieces, "
            f"so there will be at least {n_pieces} patterns regardless of the resolution. "
            f"Consider increasing n_neighbors (currently {n_neighbors}).",
            UserWarning,
            stacklevel=2,
        )

    sc.tl.rank_genes_groups(adata, 'sign_pattern')
