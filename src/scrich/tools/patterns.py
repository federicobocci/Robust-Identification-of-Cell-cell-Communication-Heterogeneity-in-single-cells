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
    r"""Identify signaling patterns: groups of cells with similar communication modes across pathways.

    Each cell is described by its communication mode in every pathway. A nearest-neighbour graph is built on
    these profiles (after PCA) and clustered with Leiden. Marker genes of each pattern are then computed with
    :func:`scanpy.tl.rank_genes_groups`.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.all_path_sim`.
    paths : list of str or 'all', default 'all'
        Pathways to use; ``'all'`` uses ``adata.uns['selected_pathways']``.
    res : float, default 1
        Leiden resolution; lower values give fewer, larger patterns.
    n_neighbors : int, default 30
        Number of neighbours of the cell graph. Many cells share identical mode combinations, so small values
        can split the graph into disconnected pieces, each giving at least one pattern whatever the
        resolution (a warning is raised). scRICH 0.1 used 10.
    random_state : int, default 0
        Seed of PCA, neighbour search, UMAP and Leiden.
    seed : int, default 42
        Seed of NumPy's global random generator (kept for backward compatibility).
    print_info : bool, default True
        Report the number of patterns and the number of cells in each.

    Returns
    -------
    None
        Sets the following fields:

        - ``adata.obs['sign_pattern']``: pattern of each cell.
        - ``adata.uns['rank_genes_groups']``: marker genes of each pattern.
        - ``adata.uns['sign_pattern_params']``: parameters used and number of disconnected graph pieces.

    Notes
    -----
    Modes renamed with :func:`scrich.tl.rename_modes` keep their original order, so renaming does not change
    the patterns.

    See Also
    --------
    scrich.pl.pattern_summary : Pattern composition of every cell type.
    scrich.pl.pattern : Details of one pattern.

    Examples
    --------
    >>> ri.tl.find_sign_patterns(adata, res=0.05)  # doctest: +SKIP
    """
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
    r"""Mutual information between the communication modes of two pathways.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with ``adata.obs[path1 + '_modes']`` and ``adata.obs[path2 + '_modes']``.
    path1, path2 : str
        Pathways to compare.

    Returns
    -------
    float
        Mutual information (in nats, :func:`sklearn.metrics.mutual_info_score`).
    """
    lab1 = np.asarray(adata.obs[path1 + '_modes'])
    lab2 = np.asarray(adata.obs[path2 + '_modes'])
    mi_skl = mutual_info_score(lab1, lab2)
    return mi_skl


def pairwise_MI(adata):
    r"""Mutual information between the communication modes of all pairs of selected pathways.

    High values indicate redundant pathways, i.e. pathways that divide cells into modes in a similar way.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.all_path_sim`.

    Returns
    -------
    None
        Sets ``adata.uns['CCC_similarity_map']``: pathways x pathways matrix of mutual information, in the
        order of ``adata.uns['selected_pathways']``.

    See Also
    --------
    scrich.pl.redundancy : Plot the matrix.
    scrich.pl.pathway_hierarchy : Hierarchical clustering of pathways.
    """
    pathways = adata.uns['selected_pathways']

    sim_mat = np.zeros((len(pathways), len(pathways)))

    for i in range(len(pathways)):
        for j in range(len(pathways)):
            sim_mat[i][j] = twopath_mutual_info(adata, pathways[i], pathways[j])
    adata.uns['CCC_similarity_map'] = sim_mat
