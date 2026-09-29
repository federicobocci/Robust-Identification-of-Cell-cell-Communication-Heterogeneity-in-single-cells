"""Communication modes: cell-cell similarity per pathway, optimal number of modes and consensus clustering."""

import warnings

import numpy as np
import pandas as pd
from numba import jit
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from ._utils import extract_neighbors, neighbor_avg
from .heterogeneity import cci_heterogeneity, state_fractions
from .pathways import select_pathways
from .targets import import_database

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


@jit(nopython=True)
def compute_sim(clst_labels):
    '''computes a similarity matrix for clustering given by clst_labels

    Parameters
    ----------
    clst_labels: list of cluster labels

    Returns
    -------
    sim: a similarity matrix with ones if cells belong to same cluster and zeros otherwise

    '''
    ncell = clst_labels.size
    sim = np.zeros((ncell, ncell))
    for i in range(ncell):
        for j in range(ncell):
            if clst_labels[i]==clst_labels[j]:
                sim[i][j]=1.
    return sim


def sim_matrix(mat,
               nsim=5,
               kmin=3,
               kmax=6,
               print_info=True
               ):
    '''compute an average similarity matrix based on k-means clustering on expression of ligands and receptors

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
    assert isinstance(mat, np.ndarray), 'expression matrix is not in numpy.ndarray format'
    assert (isinstance(kmin, int) and isinstance(kmax, int)) and kmin<kmax, 'kmin and kmax typing or values are not compatible'

    if print_info:
        print('Running clustering with nsim=%d, kmin=%d, kmax=%d' % (nsim, kmin, kmax))

    n_clst = [i for i in range(kmin, kmax+1)]
    ncell = mat.shape[0]

    sim = np.zeros((ncell, ncell))
    for k in n_clst:
        for m in range(nsim):
            if print_info:
                print('clustering #' + str(m + 1) + ' with k=' + str(k))
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=m).fit(mat)
            labels = kmeans.labels_
            sim_mat = compute_sim(labels)
            sim = sim + sim_mat/(float(nsim * len(n_clst)))

    return sim


def spectral_analysis_opt(adata,
                      A,
                      pathway,
                      print_info=True
                      ):
    '''Decompose the similarity matrix via symmetric non-negative matrix factorization
    Results for pathway are stored in adata.uns['SymNMF'][pathway]

    Parameters
    ----------
    adata: anndata object of gene counts

    Returns
    -------
    None

    '''
    # symmetric non-negative matrix factorization
    n = A.shape[0]

    # Row sums
    d = A.sum(axis=1)

    # Inverse sqrt safely
    d_inv_sqrt = np.zeros_like(d)
    mask = d > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])

    # Normalize without forming diagonal matrices
    # Equivalent to C @ A @ C
    P = (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]

    # Construct matrix
    Pwr = np.eye(n) - P

    # Faster eigen decomposition for symmetric matrices
    w, v = eigsh(Pwr, k=10, which='SM')
    w = np.real(w)

    srt = np.sort(w)
    diff = srt[1:] - srt[0:-1]
    opt = np.argmax(diff)+1
    if print_info:
        print('The optimal number of clusters is %d' % opt)

    if 'SymNMF' not in adata.uns.keys():
        adata.uns['SymNMF'] = {}
    # adata.uns['SymNMF'][pathway] = {'eigenvalues': w, 'eigenvectors': v, 'gap':diff, 'optimal':opt}
    adata.uns['SymNMF'][pathway] = {'eigenvalues': w, 'gap': diff, 'optimal': opt}


def single_path_sim(adata,
                    pathway,
                    nsim=5,
                    kmin=3,
                    kmax=6,
                    target=False,
                    human=True,
                    neighbor_average=False,
                    moments=False,
                    print_info=True
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
    human: if True, use the human database, otherwise use mouse (default=True)
    print_info: print information while executing (default=True)

    Returns
    -------
    None

    '''
    # find_genes(adata, pathway, human=human, print_info=print_info)

    rec, lig = adata.uns['pathways'][pathway]['receptors'], adata.uns['pathways'][pathway]['ligands']
    if moments:
        rec_exp = adata[:, rec].layers['Mu'].mean(axis=1) + adata[:, rec].layers['Ms'].mean(axis=1)
        lig_exp = adata[:, lig].layers['Mu'].mean(axis=1) + adata[:, lig].layers['Ms'].mean(axis=1)
    else:
        rec_exp = adata[:, rec].X.toarray().mean(axis=1)
        lig_exp = adata[:, lig].X.toarray().mean(axis=1)
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
        adata.obs[pathway+'_tar'] = tar_exp
        if neighbor_average:
            tar_exp = neighbor_avg(tar_exp, neigh)
        dat = np.array([rec_exp, lig_exp, tar_exp]).transpose()
    else:
        dat = np.array([rec_exp, lig_exp]).transpose()

    sim_mat = sim_matrix(dat, nsim=nsim, kmin=kmin, kmax=kmax, print_info=print_info)
    print('finished sim_mat for ' + pathway + ', starting spectral analysis')

    return sim_mat


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

    adata.obs[pathway + '_modes'] = labels

    if 'fraction_mat' not in adata.uns.keys():
        adata.uns['fraction_mat'] = {}
    adata.uns['fraction_mat'][pathway] = state_fractions(adata, pathway, annotations, order=order)
    cci_heterogeneity(adata, pathway)


def all_path_sim(adata, annotations, nsim=5, kmin=3, kmax=6, target=False, human=True, moments=False,
                 neighbor_average=False, method='expression', pathways_to_use=10, top_target=10, order=None, seed=100, print_info=False,
                 consensus_neighbor_average=False):
    '''Loop the clustering based on cell-cell communication to all pathways identified in the dataset

    Parameters
    ----------
    adata: anndata object with gene counts
    annotations: adata.obs key with cell type annotations
    target: if True, also use the expression of each pathway's downstream genes (selected with import_database)
        both for the similarity matrix and for the final consensus clustering of modes (default=False)
    top_target: number of downstream genes per pathway used when target=True (default=10)
    human: if True use the human database, otherwise mouse
    neighbor_average: if True, average expression over each cell's neighbours for the similarity matrix
    consensus_neighbor_average: if True, also use neighbour-averaged expression for the final consensus clustering
        of modes (default=False, i.e. modes are assigned from unsmoothed expression as in v0.1.0)
    order: deprecated, has no effect
    seed: random seed for the consensus clustering

    Returns
    -------
    None

    '''
    if isinstance(pathways_to_use, int):
        select_pathways(adata, human=human, method=method, n=pathways_to_use)
        sel_path = adata.uns['selected_pathways']
    else:
        adata.uns['selected_pathways'] = pathways_to_use
        sel_path = adata.uns['selected_pathways']
    print(sel_path)

    if target:
        import_database(adata, sel_path, top=top_target, unspliced=moments, species='human' if human else 'mouse')

    for p in sel_path:
        print('Analyzing the ' + p + ' pathway...')
        print('running clustering')
        sim_mat = single_path_sim(adata, p, nsim=nsim, kmin=kmin, kmax=kmax, target=target, human=human, moments=moments, neighbor_average=neighbor_average, print_info=print_info)
        print('')

        spectral_analysis_opt(adata, sim_mat, p, print_info=print_info)

        print('running consensus')
        # use targets only for pathways where single_path_sim could use them (some have no expressed downstream genes)
        uses_targets = target and adata.uns['pathways'][p].get('uses_targets', False)
        consensus_clustering(adata, p, annotations, n_cluster='optimal', order=order, seed=seed, target=uses_targets,
                             neighbor_average=consensus_neighbor_average)
    print(adata.uns.keys(), '4')
