'''
Functions to select the top signaling pathways and supervised clustering of signaling modes
'''
import numpy as np
from numba import jit
from sklearn.cluster import KMeans
import pandas as pd
from scipy.sparse.linalg import eigsh

from .. import db
from .database_func import get_list
from .tf_tar_functions import import_database

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


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
    assert isinstance(geneset, list) and isinstance(lst, list), 'Please convert the gene lists to Python list format'

    gene_list = []
    for l in lst:
        if l in geneset:
            gene_list.append(l)
    return list(set(gene_list))


def find_genes(adata,
               pathway,
               human=True,
               save_genes=True,
               return_lists=False,
               print_info=True
               ):
    '''Identify receptors and ligands of a given pathway that are detected in adata
    Identified receptors and ligands for pathway are stored in adata.uns['pathways']

    Parameters
    ----------
    adata: anndata object of gene counts
    pathway: pathway of interest
    human: if True, use the human database, otherwise use mouse (default=True)
    save_genes: if True, save identified genes in adata.uns['pathways'] (default=True)
    return_lists: if True, returns lists of common receptors and ligands (default=False)
    print_info: if True, print information (default=True)

    Returns
    -------
    None

    '''
    # select human or mouse CellChat database
    species = 'human' if human else 'mouse'
    lig = db.load_pathway_genes('ligand', species=species)
    rec = db.load_pathway_genes('receptor', species=species)

    # select ligands/receptor for pathway of interest
    rec_list = get_list(rec, pathway)
    lig_list = get_list(lig, pathway)

    geneset = list(adata.var_names)
    rec_common = list_intersect(geneset, rec_list)
    lig_common = list_intersect(geneset, lig_list)

    if save_genes:
        if 'pathways' not in adata.uns.keys():
            adata.uns['pathways'] = {}
        adata.uns['pathways'][pathway] = {'receptors': rec_common, 'ligands': lig_common}

    if return_lists:
        return rec_common, lig_common

    if print_info:
        print('Identified ' + str(len(rec_common)) + ' receptors in the ' + pathway + ' pathway:')
        print(*rec_common)
        print('Identified ' + str(len(lig_common)) + ' ligands in the ' + pathway + ' pathway:')
        print(*lig_common)


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
    assert key=='distances' or key=='connectivities', "Please choose between key=='distances' or key=='connectivities'"
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


def pathways_overview(adata, human=True, moments=False):
    '''Overview of pathways in the database with both Receptors and Ligands detected in adata
    Information on identified pathways is printed

    Parameters
    ----------
    adata: anndata object of gene counts
    human: if True, use the human database, otherwise use mouse (default=True)
    min_count: minimum number of Receptors and Ligands that must be detected in adata

    Returns
    -------
    None

    '''
    species = 'human' if human else 'mouse'
    lig = db.load_pathway_genes('ligand', species=species)
    rec = db.load_pathway_genes('receptor', species=species)
    path_list = list(set(list(lig['pathway'])))

    overview_genes = {}
    overview_expression = {}

    count = 0
    print('Scanning for cell-cell communication pathways...')
    for p in path_list:
        rec_list = get_list(rec, p)
        lig_list = get_list(lig, p)
        if len(rec_list)==0 or len(lig_list)==0:
            continue
        else:
            rec_common, lig_common = find_genes(adata, p, human=human, save_genes=True, print_info=False, return_lists=True)
            if len(rec_common)>0 and len(lig_common)>0:
                count = count + 1
                overview_genes[p] = [len(rec_common), len(lig_common)]

                if moments:
                    rec_exp = adata[:, rec_common].layers['Mu'].mean(axis=1)+adata[:, rec_common].layers['Ms'].mean(axis=1)
                    lig_exp = adata[:, lig_common].layers['Mu'].mean(axis=1)+adata[:, lig_common].layers['Ms'].mean(axis=1)
                else:
                    rec_exp = adata[:, rec_common].X.toarray().mean(axis=1)
                    lig_exp = adata[:, lig_common].X.toarray().mean(axis=1)
                overview_expression[p] = [np.mean(rec_exp), np.mean(lig_exp)]

    print('Identified ' + str(count) + ' pathways with at least one Receptor and one Ligand detected')
    adata.uns['paths_overview'] = {'gene_number': overview_genes, 'gene_expression': overview_expression}



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


def _mode_order(col):
    '''Modes of a pathway in a stable order: category order for renamed modes, otherwise sorted.'''
    if isinstance(col.dtype, pd.CategoricalDtype):
        return [c for c in col.cat.categories if (col == c).any()]
    return sorted(set(np.asarray(col)))

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


def select_pathways(adata, human=True, method='expression', n=10):
    '''Select the top CCC pathways on which to run the scRICH analysis.
    The list of selected pathways is stored in adata.uns['selected_pathways']

    Parameters
    ----------
    adata: scRNA-seq data object
    human: if True, use human database, otherise mouse.
    method: 'expression' (total gene expression of rec+lig genes) of 'number' (number of detected rec+lig genes)
    n: number of pathway to select

    Returns
    -------
    None

    '''
    species = 'human' if human else 'mouse'
    lig = db.load_pathway_genes('ligand', species=species)
    rec = db.load_pathway_genes('receptor', species=species)
    path_list = list(set(list(lig['pathway'])))

    if method=='expression':
        expr_dct = adata.uns['paths_overview']['gene_expression']
        path, expr = [], []
        for p in expr_dct.keys():
            path.append(p)
            expr.append(expr_dct[p])
        expr = np.mean(np.asarray(expr), axis=1)
        ind = np.flip(np.argsort(expr))

    elif method=='number':
        number_dct = adata.uns['paths_overview']['gene_number']
        path, num = [], []
        for p in number_dct.keys():
            path.append(p)
            num.append(number_dct[p])
        rec_num, lig_num = np.asarray(num)[:, 0], np.asarray(num)[:, 1]
        ind = np.flip(np.argsort(rec_num + lig_num))

    sel_path = [path[i] for i in ind[0:n]]
    adata.uns['selected_pathways'] = sel_path

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

def permutation_testing(adata, key='seurat_clusters', n_sample = 10000, perm_threshold=0.05):
    '''Compute permutation testing to identify statistically significant heterogeneous cell types.
    Results are saved in adata.uns['permutation_test']

    Parameters
    ----------
    adata: scRNA-seq data object
    key: adata.obs key for cell annotations
    n_sample: number of permutation tests
    perm_threshold: p-value threshold for statistical significance

    Returns
    -------
    None

    '''
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
            test = np.random.poisson(major_mode, n_sample)
            ratio.append(float(len(test[test > ncell])) / len(test))

            mu.append(major_mode / ncell)
        perm_dict[p] = ratio

        if np.any(np.asarray(ratio)<perm_threshold):
            pos_pathways.append(p)

    adata.uns['permutation_test'] = [types, perm_dict, pos_pathways]






