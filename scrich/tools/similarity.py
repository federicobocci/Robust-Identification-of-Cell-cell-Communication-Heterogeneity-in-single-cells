import numpy as np
from sklearn.metrics import mutual_info_score

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


