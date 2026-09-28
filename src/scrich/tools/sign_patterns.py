import numpy as np
import anndata
import scanpy as sc

def find_sign_patterns(adata, paths='all', res=1, seed=42):
    '''Identify CCC patterns across several pathways by clustering cells based on the CCC modes

    Parameters
    ----------
    adata: scRNA-seq data object
    paths: list of pathways to use in the clustering
    res: clustering resolution
    seed: random generator seed for replicability

    Returns
    -------

    '''
    np.random.seed(seed)
    if paths=='all':
        paths = adata.uns['selected_pathways']
    ncell, npath = adata.shape[0], len(paths)

    sel_data = np.zeros((npath, ncell))
    for i in range(len(paths)):
        sel_data[i] = adata.obs[paths[i]+'_modes']

    path_adata = anndata.AnnData(X=sel_data.transpose())
    path_adata.obs_names = adata.obs_names
    path_adata.var_names = paths

    sc.tl.pca(path_adata, svd_solver='arpack')
    sc.pp.neighbors(path_adata, n_neighbors=10, n_pcs=len(paths))
    sc.tl.umap(path_adata)
    sc.tl.leiden(path_adata, resolution=res)

    adata.obs['sign_pattern'] = path_adata.obs['leiden']

    sc.tl.rank_genes_groups(adata, 'sign_pattern')
