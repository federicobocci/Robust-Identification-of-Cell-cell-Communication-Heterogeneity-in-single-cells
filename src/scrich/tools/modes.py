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
    '''Factor F of the average co-clustering (similarity) matrix, S = F @ F.T.

    Each of the nsim * (kmax - kmin + 1) k-means runs contributes a one-hot cluster-membership block;
    S[i, j] is the fraction of runs in which cells i and j fall in the same cluster. Storing F
    (cells x total number of clusters) instead of S (cells x cells) keeps memory linear in the number of cells.
    '''
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
    '''compute an average similarity matrix based on k-means clustering on expression of ligands and receptors

    Note: the matrix has size cells x cells. all_path_sim() works with its low-rank factor instead
    (see _coassociation_factor), which needs far less memory.

    Parameters
    ----------
    mat: numpy array with average expression of ligand and receptors
    nsim: number of independent k-means clustering solutions for fixed k (default=5)
    kmin: minimum number of cluster (default=3)
    kmax: maximum number of cluster (default=6)

    Returns
    -------
    sim: similarity matrix

    '''
    F = _coassociation_factor(mat, nsim=nsim, kmin=kmin, kmax=kmax, print_info=print_info)
    return F @ F.T


def spectral_analysis_opt(adata,
                      A,
                      pathway,
                      print_info=True,
                      is_factor=False
                      ):
    '''Find the optimal number of communication modes from the spectrum of the similarity matrix
    Results for pathway are stored in adata.uns['SymNMF'][pathway]

    The 10 smallest eigenvalues of the normalized Laplacian I - D^-1/2 A D^-1/2 are computed and the
    optimal number of modes is given by the largest gap between consecutive eigenvalues.

    Parameters
    ----------
    adata: anndata object of gene counts
    A: similarity matrix (cells x cells), or its factor F with A = F @ F.T if is_factor=True
    pathway: pathway of interest
    print_info: if True, report the optimal number of modes
    is_factor: if True, A is the low-rank factor returned by _coassociation_factor (memory-efficient, same result)

    Returns
    -------
    None

    '''
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
    '''Compute a cell-cell similarity matrix for a given pathway
    Identified receptors and ligands for pathway are stored in adata.uns['pathways']
    Average expression of receptors and ligands for pathway are stored in adata.obs
    Similarity matrix for pathway is stored in adata.uns['similarity']

    Parameters
    ----------
    adata: anndata object with gene counts
    pathway: pathway of interest
    target: if True, also use the expression of the pathway's downstream genes (adata.uns['TF'][pathway],
        from import_database) (default=False)
    human: deprecated and unused (the pathway genes are taken from adata.uns['pathways'])
    print_info: print information while executing (default=True)
    return_factor: if True, return the factor F of the similarity matrix (S = F @ F.T, cells x number of
        clusters) instead of S itself (cells x cells), which needs much less memory (default=False)

    Returns
    -------
    sim_mat: the similarity matrix (or its factor if return_factor=True)

    '''
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
    '''Computes consensus clustering based on ligand/receptor (and optionally downstream gene) expression of a pathway
    Results are stored in adata.obs

    Parameters
    ----------
    adata: anndata object with gene counts
    pathway: pathway of interest
    n_cluster: number of clusters; can be user input, default is optimal number found by symmetric non-negative matrix factorization
    seed: seed for random umber generator (default=100)
    target: if True, also cluster on the expression of the pathway's downstream genes (adata.obs[pathway + '_tar'],
        computed by single_path_sim with target=True) (default=False)
    neighbor_average: if True, average expression over each cell's neighbours (adata.obsp['distances']) before
        clustering, as done for the similarity matrix in single_path_sim (default=False)

    Returns
    -------
    None

    '''
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
    '''Loop the clustering based on cell-cell communication to all pathways identified in the dataset

    Parameters
    ----------
    adata: anndata object with gene counts
    annotations: adata.obs key with cell type annotations
    target: if True, also use the expression of each pathway's downstream genes (selected with import_database)
        both for the similarity matrix and for the final consensus clustering of modes (default=False)
    top_target: number of downstream genes per pathway used when target=True (default=10)
    species: 'human' (default) or 'mouse'
    human: deprecated, use species
    neighbor_average: if True, average expression over each cell's neighbours for the similarity matrix
    consensus_neighbor_average: if True, also use neighbour-averaged expression for the final consensus clustering
        of modes (default=False, i.e. modes are assigned from unsmoothed expression as in v0.1.0)
    order: deprecated, has no effect
    seed: random seed for the consensus clustering

    Returns
    -------
    None

    '''
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
