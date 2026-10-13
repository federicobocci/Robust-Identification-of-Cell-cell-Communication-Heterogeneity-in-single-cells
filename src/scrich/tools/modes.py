"""Communication modes: cell-cell similarity per pathway, optimal number of modes and consensus clustering."""

import warnings

import numpy as np
from sklearn.cluster import KMeans

from .._logging import logger
from ._utils import _quiet_pandas, _resolve_species, extract_neighbors, neighbor_avg
from .heterogeneity import cci_heterogeneity, state_fractions
from .pathways import select_pathways
from .targets import import_database


def _coassociation_factor(mat, nsim=5, kmin=3, kmax=6, print_info=True):
    """Factor F of the average co-clustering (similarity) matrix, S = F @ F.T.

    Each of the nsim * (kmax - kmin + 1) k-means runs contributes a one-hot cluster-membership block;
    S[i, j] is the fraction of runs in which cells i and j fall in the same cluster. Storing F
    (cells x total number of clusters) instead of S (cells x cells) keeps memory linear in the number of cells.
    """
    if not isinstance(mat, np.ndarray):
        raise TypeError('The expression matrix must be a numpy.ndarray.')
    if not (isinstance(kmin, (int, np.integer)) and isinstance(kmax, (int, np.integer))) or kmin >= kmax:
        raise ValueError(f'kmin and kmax must be integers with kmin < kmax (got kmin={kmin}, kmax={kmax}).')

    if print_info:
        logger.info('Running clustering with nsim=%d, kmin=%d, kmax=%d', nsim, kmin, kmax)

    n_clst = list(range(kmin, kmax + 1))
    ncell = mat.shape[0]
    blocks = []
    for k in n_clst:
        for m in range(nsim):
            logger.debug('clustering #%d with k=%d', m + 1, k)
            labels = KMeans(n_clusters=k, n_init=10, random_state=m).fit(mat).labels_
            onehot = np.zeros((ncell, k))
            onehot[np.arange(ncell), labels] = 1.0
            blocks.append(onehot)
    return np.hstack(blocks) / np.sqrt(nsim * len(n_clst))


def sim_matrix(mat,
               nsim=5,
               kmin=3,
               kmax=6,
               print_info=True
               ):
    r"""Consensus (co-clustering) similarity matrix of cells from repeated k-means clustering.

    Parameters
    ----------
    mat : numpy.ndarray
        Cells x features matrix (e.g. average receptor and ligand expression per cell).
    nsim : int, default 5
        Number of k-means runs (different random seeds) for each number of clusters.
    kmin : int, default 3
        Minimum number of clusters.
    kmax : int, default 6
        Maximum number of clusters.
    print_info : bool, default True
        Report progress.

    Returns
    -------
    numpy.ndarray
        Cells x cells matrix; entry ``(i, j)`` is the fraction of runs in which cells ``i`` and ``j`` are in the
        same cluster. :func:`scrich.tl.all_path_sim` uses a memory-efficient factorized form instead.
    """
    F = _coassociation_factor(mat, nsim=nsim, kmin=kmin, kmax=kmax, print_info=print_info)
    return F @ F.T


def spectral_analysis_opt(adata,
                      A,
                      pathway,
                      print_info=True,
                      is_factor=False
                      ):
    r"""Optimal number of communication modes from the eigenvalue gap of the similarity matrix.

    The 10 smallest eigenvalues of the normalized Laplacian :math:`I - D^{-1/2} S D^{-1/2}` of the similarity
    matrix :math:`S` are computed; the optimal number of modes is the position of the largest gap between
    consecutive eigenvalues.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    A : numpy.ndarray
        Similarity matrix (cells x cells), or its factor ``F`` with ``S = F @ F.T`` if ``is_factor=True``.
    pathway : str
        Pathway name, used as key for the results.
    print_info : bool, default True
        Report the optimal number of modes.
    is_factor : bool, default False
        ``A`` is the low-rank factor of the similarity matrix (exact, memory-efficient).

    Returns
    -------
    None
        Sets ``adata.uns['SymNMF'][pathway] = {'eigenvalues', 'gap', 'optimal'}``.
    """
    n_eig = 10
    if is_factor:
        F = np.asarray(A)
        d = F @ F.sum(axis=0)                       # row sums of F @ F.T
        d_inv_sqrt = np.zeros_like(d)
        mask = d > 0
        d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
        B = d_inv_sqrt[:, None] * F                 # D^-1/2 A D^-1/2 = B @ B.T
        # the non-zero eigenvalues of B @ B.T equal those of the small matrix B.T @ B
        lam = np.linalg.eigvalsh(B.T @ B)[::-1]
        lam = np.concatenate([lam, np.zeros(max(0, n_eig - lam.size))])[:n_eig]
        w = np.sort(1.0 - lam)
    else:
        from scipy.sparse.linalg import eigsh
        n = A.shape[0]
        d = A.sum(axis=1)
        d_inv_sqrt = np.zeros_like(d)
        mask = d > 0
        d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
        P = (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]
        w, _ = eigsh(np.eye(n) - P, k=n_eig, which='SM')
        w = np.real(w)

    srt = np.sort(w)
    diff = srt[1:] - srt[0:-1]
    opt = np.argmax(diff)+1
    if print_info:
        logger.info('The optimal number of clusters is %d', opt)

    if 'SymNMF' not in adata.uns.keys():
        adata.uns['SymNMF'] = {}
    adata.uns['SymNMF'][pathway] = {'eigenvalues': w, 'gap': diff, 'optimal': opt}


def single_path_sim(adata,
                    pathway,
                    nsim=5,
                    kmin=3,
                    kmax=6,
                    target=False,
                    human=None,
                    neighbor_average=False,
                    moments=False,
                    print_info=True,
                    return_factor=False
                    ):
    r"""Cell-cell similarity matrix for one pathway.

    Computes the average receptor and ligand expression of the pathway in every cell (and, with
    ``target=True``, of its downstream genes) and builds a consensus similarity matrix by repeated k-means
    clustering of these profiles (see ``sim_matrix``). Usually called through
    :func:`scrich.tl.all_path_sim`.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.pathways_overview` or :func:`scrich.tl.find_genes`.
    pathway : str
        Pathway of interest.
    nsim : int, default 5
        Number of k-means runs for each number of clusters.
    kmin : int, default 3
        Minimum number of clusters used to build the similarity matrix.
    kmax : int, default 6
        Maximum number of clusters used to build the similarity matrix.
    target : bool, default False
        Also use the average expression of the pathway's downstream genes (``adata.uns['TF'][pathway]``,
        from :func:`scrich.tl.import_database`). Pathways without expressed downstream genes fall back to
        receptors and ligands, with a warning.
    human : bool, optional
        Deprecated and unused.
    neighbor_average : bool, default False
        Average expression over each cell's neighbours (``adata.obsp['distances']``) before clustering.
    moments : bool, default False
        Use ``adata.layers['Mu'] + adata.layers['Ms']`` instead of ``adata.X``.
    print_info : bool, default True
        Report progress.
    return_factor : bool, default False
        Return the factor ``F`` of the similarity matrix (``S = F @ F.T``; cells x number of clusters)
        instead of the cells x cells matrix ``S``.

    Returns
    -------
    numpy.ndarray
        Similarity matrix, or its factor if ``return_factor=True``. Sets ``adata.obs[pathway + '_rec']``,
        ``adata.obs[pathway + '_lig']`` (and ``'_tar'``) with the per-cell averages (not neighbour-averaged) and
        ``adata.uns['pathways'][pathway]['uses_targets']``.
    """
    if human is not None:
        _resolve_species(human=human)  # deprecation warning only

    rec, lig = adata.uns['pathways'][pathway]['receptors'], adata.uns['pathways'][pathway]['ligands']
    if moments:
        rec_exp = adata[:, rec].layers['Mu'].mean(axis=1) + adata[:, rec].layers['Ms'].mean(axis=1)
        lig_exp = adata[:, lig].layers['Mu'].mean(axis=1) + adata[:, lig].layers['Ms'].mean(axis=1)
    else:
        rec_exp = adata[:, rec].X.toarray().mean(axis=1)
        lig_exp = adata[:, lig].X.toarray().mean(axis=1)
    with _quiet_pandas():
        adata.obs[pathway+'_rec'] = np.asarray(rec_exp)
        adata.obs[pathway+'_lig'] = np.asarray(lig_exp)

    if neighbor_average:
        neigh = extract_neighbors(adata)
        rec_exp, lig_exp = neighbor_avg(rec_exp, neigh), neighbor_avg(lig_exp, neigh)

    if target:
        # downstream genes of the pathway, selected by import_database() and stored in adata.uns['TF']
        if pathway not in adata.uns.get('TF', {}):
            raise ValueError(f"No downstream genes for pathway {pathway!r} in adata.uns['TF']. "
                             "Run scrich.tl.import_database() first, or use target=False.")
        tar = list(adata.uns['TF'][pathway])
        if len(tar) == 0:
            warnings.warn(f"No downstream genes of pathway {pathway!r} are expressed in the dataset; "
                          "clustering it on ligands and receptors only.", UserWarning, stacklevel=2)
            target = False
    adata.uns['pathways'][pathway]['uses_targets'] = bool(target)
    if target:
        if moments:
            tar_exp = adata[:, tar].layers['Mu'].mean(axis=1) + adata[:, tar].layers['Ms'].mean(axis=1)
        else:
            tar_exp = adata[:, tar].X.toarray().mean(axis=1)
        tar_exp = np.asarray(tar_exp).ravel()
        with _quiet_pandas():
            adata.obs[pathway+'_tar'] = tar_exp
        if neighbor_average:
            tar_exp = neighbor_avg(tar_exp, neigh)
        dat = np.array([rec_exp, lig_exp, tar_exp]).transpose()
    else:
        dat = np.array([rec_exp, lig_exp]).transpose()

    F = _coassociation_factor(dat, nsim=nsim, kmin=kmin, kmax=kmax, print_info=print_info)
    logger.debug('finished similarity matrix for %s', pathway)
    return F if return_factor else F @ F.T


def consensus_clustering(adata,
                         pathway,
                         annotations,
                         n_cluster='optimal',
                         order=None,
                         seed=100,
                         target=False,
                         neighbor_average=False
                         ):
    r"""Assign every cell to a communication mode of a pathway.

    Cells are clustered with k-means on their receptor and ligand expression (and downstream-gene
    expression with ``target=True``) into the optimal number of modes found by
    ``spectral_analysis_opt``, or ``n_cluster`` modes. The distribution of cell types across modes and
    the heterogeneity score of each cell type are then computed.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.single_path_sim` and ``spectral_analysis_opt``.
    pathway : str
        Pathway of interest.
    annotations : str
        Key in ``adata.obs`` with cell-type annotations.
    n_cluster : int or 'optimal', default 'optimal'
        Number of modes; ``'optimal'`` uses ``adata.uns['SymNMF'][pathway]['optimal']``.
    order : list, optional
        Deprecated, has no effect (cell types are stored in sorted order).
    seed : int, default 100
        Random seed of k-means.
    target : bool, default False
        Also cluster on ``adata.obs[pathway + '_tar']``.
    neighbor_average : bool, default False
        Average expression over each cell's neighbours before clustering.

    Returns
    -------
    None
        Sets the following fields:

        - ``adata.obs[pathway + '_modes']``: communication mode of each cell (integers ``0, 1, ...``).
        - ``adata.uns['fraction_mat'][pathway]``: number of cells per mode (rows) and cell type (columns).
        - ``adata.uns['pathways'][pathway]``: ``'modes'`` and ``'cell_types'`` (row and column labels of the
          fraction matrix), ``'cci_heterogeneity'`` (score per cell type, see Notes) and ``'neighbor_average'``.

    Notes
    -----
    The heterogeneity score of a cell type compares its distribution over modes, :math:`\rho_i`, with that of
    all cells, :math:`m_i`. It is :math:`1 - \frac{1}{M}\sum_i |\rho_i - m_i| / s_i`, with :math:`M` modes and
    :math:`s_i = 1 - m_i` if :math:`\rho_i > m_i`, else :math:`s_i = m_i`. It is close to 1 when the cell
    type contains the modes in the same proportions as the whole population (heterogeneous) and close to 0
    when its composition deviates maximally.
    """
    # get average ligand, receptor expression
    rec_exp, lig_exp = np.asarray(adata.obs[pathway + '_rec']), np.asarray(adata.obs[pathway + '_lig'])
    if target:
        if pathway + '_tar' not in adata.obs:
            raise ValueError(f"adata.obs[{pathway + '_tar'!r}] not found. Run single_path_sim(..., target=True) "
                             "(or all_path_sim(..., target=True)) first, or use target=False.")
        tar_exp = np.asarray(adata.obs[pathway + '_tar'])
        dat = np.array([rec_exp, lig_exp, tar_exp]).transpose()
    else:
        dat = np.array([rec_exp, lig_exp]).transpose()

    if neighbor_average:
        # same smoothing as used for the similarity matrix; adata.obs keeps the unsmoothed values
        neigh = extract_neighbors(adata)
        dat = np.array([neighbor_avg(col, neigh) for col in dat.T]).transpose()
    adata.uns['pathways'][pathway]['neighbor_average'] = bool(neighbor_average)

    # run clustering with chosen number of clusters
    if n_cluster=='optimal':
        n = adata.uns['SymNMF'][pathway]['optimal']
    else:
        n = n_cluster
    kmeans = KMeans(n_clusters=n, n_init=10, random_state=seed).fit(dat)
    labels = kmeans.labels_

    with _quiet_pandas():
        adata.obs[pathway + '_modes'] = labels

    if 'fraction_mat' not in adata.uns.keys():
        adata.uns['fraction_mat'] = {}
    adata.uns['fraction_mat'][pathway] = state_fractions(adata, pathway, annotations, order=order)
    cci_heterogeneity(adata, pathway)


def all_path_sim(adata, annotations, nsim=5, kmin=3, kmax=6, target=False, species=None, moments=False,
                 neighbor_average=False, method='expression', pathways_to_use=10, top_target=10, order=None, seed=100, print_info=False,
                 consensus_neighbor_average=False, human=None):
    r"""Identify the communication modes of several pathways.

    Runs the complete mode analysis for each pathway: similarity matrix (:func:`scrich.tl.single_path_sim`), optimal
    number of modes (``spectral_analysis_opt``) and assignment of cells to modes
    (:func:`scrich.tl.consensus_clustering`), including the heterogeneity score of every cell type.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.pathways_overview`.
    annotations : str
        Key in ``adata.obs`` with cell-type annotations.
    nsim : int, default 5
        Number of k-means runs for each number of clusters in the similarity matrix.
    kmin : int, default 3
        Minimum number of clusters used to build the similarity matrix.
    kmax : int, default 6
        Maximum number of clusters used to build the similarity matrix.
    target : bool, default False
        Also use the pathways' downstream genes (selected with :func:`scrich.tl.import_database`, which is
        called automatically) for the similarity matrix and the mode assignment.
    species : {'human', 'mouse'}, default 'human'
        Species of the databases.
    moments : bool, default False
        Use ``adata.layers['Mu'] + adata.layers['Ms']`` instead of ``adata.X``.
    neighbor_average : bool, default False
        Average expression over each cell's neighbours for the similarity matrix.
    method : {'expression', 'number'}, default 'expression'
        How to rank pathways when ``pathways_to_use`` is a number (see :func:`scrich.tl.select_pathways`).
    pathways_to_use : int or list of str, default 10
        Number of top pathways to analyse, or the list of pathways.
    top_target : int, default 10
        Number of downstream genes per pathway with ``target=True``.
    order : list, optional
        Deprecated, has no effect.
    seed : int, default 100
        Random seed of the final k-means clustering.
    print_info : bool, default False
        Report details for each pathway.
    consensus_neighbor_average : bool, default False
        Also use neighbour-averaged expression for the final mode assignment (by default only the similarity
        matrix uses it, as in scRICH 0.1).
    human : bool, optional
        Deprecated, use ``species``.

    Returns
    -------
    None
        Sets ``adata.uns['selected_pathways']`` and, for every pathway, the fields described in
        :func:`scrich.tl.single_path_sim`, ``spectral_analysis_opt`` and :func:`scrich.tl.consensus_clustering`
        (``adata.obs[pathway + '_modes']``, ``adata.uns['fraction_mat']``, ``adata.uns['SymNMF']``, ...).

    See Also
    --------
    scrich.tl.permutation_testing : Test which cell types are significantly heterogeneous.
    scrich.pl.mode_violin : Receptor and ligand expression in each mode.

    Examples
    --------
    >>> import scrich as ri
    >>> ri.tl.pathways_overview(adata, species="mouse")  # doctest: +SKIP
    >>> ri.tl.all_path_sim(adata, "celltype", pathways_to_use=10, species="mouse",
    ...                    neighbor_average=True)  # doctest: +SKIP
    """
    species = _resolve_species(species, human)
    if isinstance(pathways_to_use, int):
        select_pathways(adata, species=species, method=method, n=pathways_to_use)
        sel_path = adata.uns['selected_pathways']
    else:
        adata.uns['selected_pathways'] = pathways_to_use
        sel_path = adata.uns['selected_pathways']
    logger.info('Selected pathways: %s', ', '.join(map(str, sel_path)))

    if target:
        import_database(adata, sel_path, top=top_target, unspliced=moments, species=species)

    for p in sel_path:
        logger.info('Analyzing the %s pathway...', p)
        F = single_path_sim(adata, p, nsim=nsim, kmin=kmin, kmax=kmax, target=target, moments=moments,
                            neighbor_average=neighbor_average, print_info=print_info, return_factor=True)
        spectral_analysis_opt(adata, F, p, print_info=print_info, is_factor=True)
        del F
        # use targets only for pathways where single_path_sim could use them (some have no expressed downstream genes)
        uses_targets = target and adata.uns['pathways'][p].get('uses_targets', False)
        consensus_clustering(adata, p, annotations, n_cluster='optimal', order=order, seed=seed, target=uses_targets,
                             neighbor_average=consensus_neighbor_average)
