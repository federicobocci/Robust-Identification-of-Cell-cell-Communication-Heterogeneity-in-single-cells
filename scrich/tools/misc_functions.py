'''
miscellanea functions for the package
'''
import numpy as np

def rename_modes(adata, pathway, mode_dict):
    '''Rename CCC modes of a pathway

    Parameters
    ----------
    adata: scRNA-seq data object
    pathway: pathways whose mode are renamed
    mode_dict: dictionary with new CCC mode names

    Returns
    -------
    None

    '''
    old_mode = np.asarray(adata.obs[pathway + '_modes'])
    adata.obs[pathway + '_modes'] = [mode_dict[o] for o in old_mode]

def rename_cells(adata, cell_dict, key='clusters'):
    '''Rename cell annotations

    Parameters
    ----------
    adata: scRNA-seq data object
    cell_dict: dictionary with new cell type names
    key: adata.obs key that stores cell annotations

    Returns
    -------

    '''
    old_cell = np.asarray(adata.obs[key])
    new_cell = old_cell.copy()
    for i in range(new_cell.size):
        if new_cell[i] in cell_dict.keys():
            new_cell[i] = cell_dict[new_cell[i]]
    adata.obs[key] = new_cell