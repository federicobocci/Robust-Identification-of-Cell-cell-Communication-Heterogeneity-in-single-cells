'''
Functions to select the top signaling pathways and supervised clustering of signaling modes
'''
import numpy as np
from numba import jit
from sklearn.cluster import KMeans
import pandas as pd

from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

from .database_func import loadDB, get_list
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
    # assert len(geneset)>0 and len(lst)>0, 'One or more inputs is a list with no elements'

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

    '''
    # select human or mouse CellChat database
    if human:
        lig = loadDB('pathway_database/ligand_human.csv')
        rec = loadDB('pathway_database/receptor_human.csv')
    else:
        lig = loadDB('pathway_database/ligand_mouse.csv')
        rec = loadDB('pathway_database/receptor_mouse.csv')

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
    assert key=='distances' or key=='connectivities', "Please choose between key=='distances' or key=='connectivities'"
    dist = adata.obsp[key]
    neighbors = [ dist[i].indices for i in range(adata.shape[0]) ]
    return neighbors


def neighbor_avg(v, neigh):
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
    if human:
        lig = loadDB('pathway_database/ligand_human.csv')
        rec = loadDB('pathway_database/receptor_human.csv')
    else:
        lig = loadDB('pathway_database/ligand_mouse.csv')
        rec = loadDB('pathway_database/receptor_mouse.csv')
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
            kmeans = KMeans(n_clusters=k, random_state=m).fit(mat)
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
    # A = adata.uns['similarity'][pathway]

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
    # w, v = np.linalg.eigh(Pwr)
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



def spectral_analysis_old(adata,
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
    # A = adata.uns['similarity'][pathway]

    # symmetric non-negative matrix factorization
    print('start')
    n = A.shape[0]
    D = np.diag(np.matmul(A, np.ones(n)))
    print('matmul 1 done')
    C = D ** (-1. / 2)
    C[C == np.inf] = 0
    print('start matmul 2')
    Pwr = np.identity(n) - np.matmul(C, np.matmul(A, C))
    print('matmul 2 done')
    w, v = np.linalg.eig(Pwr)
    print('eigenvalue done')
    w = np.real(w)
    print('stop')

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
                    target=True,
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
        tar = adata.uns['targets'][pathway]
        tar_exp = np.asarray(adata[:, tar].X).mean(axis=1)
        adata.obs[pathway+'_tar'] = tar_exp
        dat = np.array([rec_exp, lig_exp, tar_exp]).transpose()
    else:
        dat = np.array([rec_exp, lig_exp]).transpose()

    sim_mat = sim_matrix(dat, nsim=nsim, kmin=kmin, kmax=kmax, print_info=print_info)
    print('finished sim_mat for ' + pathway + ', starting spectral analysis')

    return sim_mat


def state_fractions(adata, pathway, annotations, order=None):
    labs = np.asarray(list(adata.obs[pathway + '_modes']))
    lab_set = list(set(labs))
    clusters = np.asarray(list(adata.obs[annotations]))
    if order == None:
        clst_set = sorted(list(set(clusters)))
    else:
        clst_set = list(np.asarray(order))

    ntypes, nclst = len(lab_set), len(clst_set)
    frac = np.zeros((ntypes, nclst))

    for i in range(ntypes):
        type1 = clusters[labs == lab_set[i]]
        for j in range(nclst):
            frac[i][j] = len(type1[type1 == clst_set[j]])
    return frac

def cci_heterogeneity(adata, pathway):
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
    v = v / mode_density.size

    # v = 1 - v #  THIS NEEDS TO BE UNCOMMENTED IN THE END!

    # for j in range(n_clst):
    #     rho = frac[:, j] / np.sum(frac[:, j])
    #     v[j] = np.sum(np.abs(rho - mode_density))
    adata.uns['pathways'][pathway]['cci_heterogeneity'] = v


def consensus_clustering(adata,
                         pathway,
                         annotations,
                         n_cluster='optimal',
                         order=None,
                         seed=100
                         ):
    '''Computes consensus clustering based on ligand/receptor expression of a given pathway
    Results are stored in adata.obs

    Parameters
    ----------
    adata: anndata object with gene counts
    pathway: pathway of interest
    n_cluster: number of clusters; can be user input, default is optimal number found by symmetric non-negative matrix factorization
    seed: seed for random umber generator (default=100)

    Returns
    -------
    None

    '''
    # get average ligand, receptor expression
    rec_exp, lig_exp = np.asarray(adata.obs[pathway + '_rec']), np.asarray(adata.obs[pathway + '_lig'])
    dat = np.array([rec_exp, lig_exp]).transpose()

    # run clustering with chosen number of clusters
    if n_cluster=='optimal':
        n = adata.uns['SymNMF'][pathway]['optimal']
    else:
        n = n_cluster
    kmeans = KMeans(n_clusters=n, random_state=seed).fit(dat)
    labels = kmeans.labels_

    adata.obs[pathway + '_modes'] = labels

    if 'fraction_mat' not in adata.uns.keys():
        adata.uns['fraction_mat'] = {}
    adata.uns['fraction_mat'][pathway] = state_fractions(adata, pathway, annotations, order=order)
    cci_heterogeneity(adata, pathway)


def select_pathways(adata, human=True, method='expression', n=10):
    if human:
        lig = loadDB('pathway_database/ligand_human.csv')
        rec = loadDB('pathway_database/receptor_human.csv')
    else:
        lig = loadDB('pathway_database/ligand_mouse.csv')
        rec = loadDB('pathway_database/receptor_mouse.csv')
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

def all_path_sim(adata, annotations, nsim=5, kmin=3, kmax=6, target=True, human=True, moments=False,
                 neighbor_average=False, method='expression', pathways_to_use=10, top_target=10, order=None, seed=100, print_info=False):
    '''Loop the clustering based on cell-cell communication to all pathways identified in the dataset

    Parameters
    ----------
    adata
    annotations
    human
    min_count
    order
    seed

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
        import_database(adata, sel_path, top=top_target)

    for p in sel_path:
        print('Analyzing the ' + p + ' pathway...')
        print('running clustering')
        sim_mat = single_path_sim(adata, p, nsim=nsim, kmin=kmin, kmax=kmax, target=target, human=human, moments=moments, neighbor_average=neighbor_average, print_info=print_info)
        print('')

        # spectral_analysis(adata, sim_mat, p, kmax=kmax, print_info=print_info)
        spectral_analysis_opt(adata, sim_mat, p, print_info=print_info)

        print('running consensus')
        consensus_clustering(adata, p, annotations, n_cluster='optimal', order=order, seed=seed)
    print(adata.uns.keys(), '4')

def permutation_testing(adata, key='seurat_clusters', n_sample = 10000, perm_threshold=0.05):
    perm_dict = {}
    # pathways = list(adata.uns['pathways'].keys())
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

        # perhaps remove other pathways instead of saving them to the anndata?
        if np.any(np.asarray(ratio)<perm_threshold):
            pos_pathways.append(p)

    adata.uns['permutation_test'] = [types, perm_dict, pos_pathways]






