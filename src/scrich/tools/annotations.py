"""Renaming of communication modes and cell annotations."""

import numpy as np
import pandas as pd


def rename_modes(adata, pathway, mode_dict):
    '''Rename CCC modes of a pathway

    Parameters
    ----------
    adata: scRNA-seq data object
    pathway: pathways whose mode are renamed
    mode_dict: dictionary mapping old mode labels to new names, e.g. {0: 'Receiver', 1: 'Inactive'}.
        Modes not in the dictionary keep their label. Several modes can be given the same name to merge them.

    Returns
    -------
    None

    '''
    col = adata.obs[pathway + '_modes']
    if isinstance(col.dtype, pd.CategoricalDtype):
        old_order = list(col.cat.categories)
    else:
        old_order = sorted(pd.unique(col))
    # modes missing from mode_dict keep their name; the original mode order is preserved, so the
    # underlying integer codes (used e.g. by find_sign_patterns) do not change when modes are renamed
    def new_name(m):
        return mode_dict.get(m, mode_dict.get(str(m), m))

    mapping = {m: new_name(m) for m in old_order}
    # avoid mixing numbers and strings (e.g. only some modes renamed), which breaks sorting and plotting
    if any(isinstance(v, str) for v in mapping.values()):
        mapping = {m: str(v) for m, v in mapping.items()}
    new_order = list(dict.fromkeys(mapping.values()))
    new_values = [mapping[m] for m in np.asarray(col)]
    adata.obs[pathway + '_modes'] = pd.Categorical(new_values, categories=new_order)


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
