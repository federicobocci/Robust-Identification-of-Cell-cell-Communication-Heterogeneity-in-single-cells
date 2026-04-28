import numpy as np
from sklearn.metrics import mutual_info_score

def twopath_mutual_info(adata, path1, path2):
    lab1 = np.asarray(adata.obs[path1 + '_modes'], dtype='int')
    lab2 = np.asarray(adata.obs[path2 + '_modes'], dtype='int')
    mi_skl = mutual_info_score(lab1, lab2)
    return mi_skl

def pairwise_MI(adata):
    # pathways = list(adata.uns['pathways'].keys())
    pathways = adata.uns['selected_pathways']

    sim_mat = np.zeros((len(pathways), len(pathways)))

    for i in range(len(pathways)):
        for j in range(len(pathways)):
            sim_mat[i][j] = twopath_mutual_info(adata, pathways[i], pathways[j])
    adata.uns['CCC_similarity_map'] = sim_mat


