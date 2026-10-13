"""Renaming of communication modes and cell annotations."""

import numpy as np
import pandas as pd


def rename_modes(adata, pathway, mode_dict):
    r"""Rename the communication modes of a pathway.

    Modes are stored as an ordered categorical that keeps the original mode order, so results that depend on
    the modes (e.g. :func:`scrich.tl.find_sign_patterns`) are unchanged by one-to-one renaming.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with ``adata.obs[pathway + '_modes']``.
    pathway : str
        Pathway whose modes are renamed.
    mode_dict : dict
        Old mode label -> new name, e.g. ``{0: 'Receiver', 1: 'Inactive', 2: 'Sender/Receiver'}``. Modes not in
        the dictionary keep their label (converted to text if any new name is text). Several modes can be
        given the same name to merge them.

    Returns
    -------
    None
        Updates ``adata.obs[pathway + '_modes']``.

    Examples
    --------
    >>> ri.tl.rename_modes(adata, "IGF", {0: "Receiver", 1: "Inactive", 2: "Sender/Receiver"})  # doctest: +SKIP
    """
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
    r"""Rename cell annotations.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    cell_dict : dict
        Old label -> new label; labels not in the dictionary are unchanged.
    key : str, default 'clusters'
        Key in ``adata.obs`` with the annotations.

    Returns
    -------
    None
        Updates ``adata.obs[key]``.
    """
    old_cell = np.asarray(adata.obs[key])
    new_cell = old_cell.copy()
    for i in range(new_cell.size):
        if new_cell[i] in cell_dict.keys():
            new_cell[i] = cell_dict[new_cell[i]]
    adata.obs[key] = new_cell
