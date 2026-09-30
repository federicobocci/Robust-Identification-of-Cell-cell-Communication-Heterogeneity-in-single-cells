"""Downstream genes (transcription factors) of CCC pathways from the exFinder database."""

import numpy as np
import pandas as pd

from .. import db


def layer_2_DB(species='human'):
    r"""Receptor -> transcription factor table of the exFinder database.

    Parameters
    ----------
    species : {'human', 'mouse'}, default 'human'
        Species.

    Returns
    -------
    pandas.DataFrame
        Columns ``from`` (receptor), ``to`` (transcription factor) and ``source``.
    """
    TDB = db.load_exfinder('rtf', species=species)
    return TDB


def get_TF(rec_dct, tf_df, pathway):
    r"""Transcription factors downstream of the receptors of a pathway.

    Parameters
    ----------
    rec_dct : dict
        Pathway -> list of receptors.
    tf_df : pandas.DataFrame
        Receptor -> transcription factor table (:func:`layer_2_DB`).
    pathway : str
        Pathway of interest.

    Returns
    -------
    list of str
        Transcription factors (in no particular order).
    """
    start_list = rec_dct[pathway]
    tf_list = list(set(list(tf_df[tf_df['from'].isin(start_list)]['to'])))
    return tf_list


def get_counts(adata, unspliced=False):
    r"""Average expression of every gene across cells.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    unspliced : bool, default False
        Use ``adata.layers['Mu'] + adata.layers['Ms']`` instead of ``adata.X``.

    Returns
    -------
    pandas.DataFrame
        Columns ``gene`` and ``avg_count``, one row per gene.
    """
    genes = list(adata.var_names)

    if unspliced:
        counts = np.asarray(adata.layers['Mu'].mean(axis=0) + adata.layers['Ms'].mean(axis=0)).ravel()
    else:
        # per-gene mean; works for sparse and dense X
        counts = np.asarray(adata.X.mean(axis=0)).ravel()

    count_df = pd.DataFrame.from_dict({'gene':genes, 'avg_count':counts})
    return count_df


def select_top_tf(count_df, tf_list, n=10):
    r"""Most expressed genes of a list.

    Parameters
    ----------
    count_df : pandas.DataFrame
        Average expression per gene (:func:`get_counts`).
    tf_list : list of str
        Candidate genes; genes not in the data are ignored.
    n : int, default 10
        Number of genes to keep.

    Returns
    -------
    list of str
        Up to ``n`` genes, in the order of ``adata.var_names``.
    """
    geneset = list(count_df['gene'])
    intersect = [t for t in tf_list if t in geneset]

    sel_df = count_df[count_df['gene'].isin(intersect)]

    if n<sel_df.shape[0]:
        sel_df = sel_df.nlargest(n, 'avg_count')
    keep = list(sel_df['gene'])
    return keep


def import_database(adata, pathways, top=10, unspliced=False, species='human', input_target=None):
    r"""Select the downstream genes (transcription factors) of each pathway.

    For every pathway, the transcription factors regulated by its receptors are taken from layer 2
    (receptor -> transcription factor) of the exFinder database, or from ``input_target``, and the ``top``
    most expressed ones in the data are kept. The database is downloaded on first use, see
    :mod:`scrich.db`.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.pathways_overview` or :func:`scrich.tl.find_genes`.
    pathways : list of str
        Pathways of interest.
    top : int, default 10
        Number of downstream genes to keep per pathway.
    unspliced : bool, default False
        Rank genes by ``adata.layers['Mu'] + adata.layers['Ms']`` instead of ``adata.X``.
    species : {'human', 'mouse'}, default 'human'
        Species of the database.
    input_target : dict, optional
        Pathway -> list of downstream genes provided by the user; overrides the database for those pathways.

    Returns
    -------
    None
        Sets ``adata.uns['TF']``: pathway -> list of selected downstream genes.

    Examples
    --------
    >>> ri.tl.import_database(adata, ["IGF"], species="mouse", unspliced=True)  # doctest: +SKIP
    """
    input_target = {} if input_target is None else input_target
    missing = [p for p in pathways if p not in adata.uns.get('pathways', {})]
    if missing:
        raise KeyError(f"Pathways {missing} not found in adata.uns['pathways']. Run pathways_overview() or find_genes() first.")
    rec_dct = {}
    for p in pathways:
        rec_dct[p] = adata.uns['pathways'][p]['receptors']

    tf_df = layer_2_DB(species=species)

    # compute average gene count to select
    count_df = get_counts(adata, unspliced=unspliced)

    tf_dict = {}
    for p in pathways:
        if p in input_target.keys():
            tf_list = input_target[p]
        else:
            tf_list = get_TF(rec_dct, tf_df, p)
        tf_dict[p] = select_top_tf(count_df, tf_list, n=top)

    adata.uns['TF'] = tf_dict
