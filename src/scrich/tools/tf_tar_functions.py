'''
functions to parse through the exfinder database
'''

import numpy as np
import pandas as pd

from .. import db


def layer_2_DB(species='human'):
    '''
    Load the Receptor-transcription factor database and return it as a dataframe

    Parameters
    ----------
    species: human or mouse (default: human)

    Returns
    -------
    TDB: pandas dataframe of receptor-TF interactions

    '''
    TDB = db.load_exfinder('rtf', species=species)
    return TDB


def get_TF(rec_dct, tf_df, pathway):
    '''
    Extract the transcription factors of a given pathway

    Parameters
    ----------
    rec_dct: dictionary of receptors per pathway
    tf_df: dataframe of transcription factor (output of layer_2_DB)
    pathway: pathway of interest

    Returns
    -------
    tf_list: list of TFs for the pathway

    '''
    start_list = rec_dct[pathway]
    tf_list = list(set(list(tf_df[tf_df['from'].isin(start_list)]['to'])))
    return tf_list


def get_counts(adata, unspliced=False):
    '''Average expression of every gene across cells

    Parameters
    ----------
    adata: anndata object
    unspliced: if True, use the imputed unspliced + spliced layers (Mu + Ms), otherwise adata.X

    Returns
    -------
    count_df: dataframe with columns 'gene' and 'avg_count' (one row per gene)

    '''
    genes = list(adata.var_names)

    if unspliced:
        counts = np.asarray(adata.layers['Mu'].mean(axis=0) + adata.layers['Ms'].mean(axis=0)).ravel()
    else:
        # per-gene mean; works for sparse and dense X
        counts = np.asarray(adata.X.mean(axis=0)).ravel()

    count_df = pd.DataFrame.from_dict({'gene':genes, 'avg_count':counts})
    return count_df

def select_top_tf(count_df, tf_list, n=10):
    '''
    Select the top transcription factors of a pathway that are highly expressed in the dataset

    Parameters
    ----------
    count_df: dataframe with geneset and average expression in the dataset
    tf_list: list of transcription factors in the pathway
    n: number of top TF to keep (default=10)

    Returns
    -------
    keep: list of top TF

    '''
    geneset = list(count_df['gene'])
    intersect = [t for t in tf_list if t in geneset]

    sel_df = count_df[count_df['gene'].isin(intersect)]

    if n<sel_df.shape[0]:
        sel_df = sel_df.nlargest(n, 'avg_count')
    keep = list(sel_df['gene'])
    return keep


def import_database(adata, pathways, top=10, unspliced=False, species='human', input_target={}):
    '''Find target genes of CCC pathways, either using the built-in database or passed by user via input_target.
    results are stored in adata.uns['TF'].

    Parameters
    ----------
    adata: scRNA-seq data object
    pathways: list of pathways for whom to identify targets
    top: number of top pathways to consider
    unspliced: if True, use unspliced counts to estimate top expressed CCC pathways
    species: human (default) or mouse
    input_target: a dictionary linking CCC pathways to lists of target genes (if the target genes are provided by the user)

    Returns
    -------
    None

    '''
    # load the database for ligand, receptor, and TF-receptor:
    rec_dct = {}
    for p in pathways:
        rec_dct[p] = adata.uns['pathways'][p]['receptors']

    tf_df = layer_2_DB(species=species)

    # compute average gene count to select
    count_df = get_counts(adata, unspliced=unspliced)

    tf_dict, tar_dict = {}, {}
    for p in pathways:
        if p in input_target.keys():
            tf_list = input_target[p]
        else:
            tf_list = get_TF(rec_dct, tf_df, p)
        tf_dict[p] = select_top_tf(count_df, tf_list, n=top)

    adata.uns['TF'] = tf_dict



