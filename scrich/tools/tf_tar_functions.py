'''
functions to parse through the exfinder database
'''

import numpy as np
import pandas as pd
from pathlib import Path

script_dir = Path(__file__).parent

def cellchat_DB(species='human'):
    '''
    Load the CellChat database and return as dictionaries of ligands and receptors

    Parameters
    ----------
    species: human or mouse (default: human)

    Returns
    -------
    lig_dct: dictionaru of ligands
    rec_dct: dictionary of receptors

    '''
    assert species=='human' or species=='mouse', "Choose between species=='human' or species=='mouse'"
    # CB = pd.read_csv('/Users/federicobocci/Desktop/commflow_project/ccc_project/exfinder_database/data/interaction_input_CellChatDB_' + species + '.csv')
    file_path = script_dir / f"../../exfinder_database/data/interaction_input_CellChatDB_{species}.csv"
    CB = pd.read_csv(file_path)

    lig_dct, rec_dct = {}, {}

    pathways = list(set(list(CB['pathway_name'])))
    for p in pathways:
        lig_dct[p] = list(set(list(CB[CB['pathway_name'] == p]['ligand'])))
        rec_dct[p] = list(set(list(CB[CB['pathway_name'] == p]['receptor'])))
    return lig_dct, rec_dct

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
    assert species == 'human' or species == 'mouse', "Choose between species=='human' or species=='mouse'"
    # TDB = pd.read_csv('/Users/federicobocci/Desktop/commflow_project/ccc_project/exfinder_database/data/RTF_layer2_' + species + '.csv')
    file_path = script_dir / f"../../exfinder_database/data/RTF_layer2_{species}.csv"
    TDB = pd.read_csv(file_path)
    return TDB

def layer_3_DB(species='human'):
    '''
    Load the TF-target database and return it as a dataframe

    Parameters
    ----------
    species: human or mouse (default: human)

    Returns
    -------
    DB: pandas dataframe of TF-target interactions

    '''
    file_path = script_dir / f"../../exfinder_database/data/TFT_layer3_{species}.csv"
    # DB = pd.read_csv('../../exfinder_database/data/TFT_layer3_' + species + '.csv')
    DB = pd.read_csv(file_path)
    return DB

def get_TF(rec_dct, tf_df, pathway):
    '''
    Extract the transcription factors of a given pathway

    Parameters
    ----------
    rec_dct: dictionary of receptors (output of cellchat_DB)
    lig_dct: dictionary of ligands (output of cellchat_DB) - necessary if no receptor is in the database
    tf_df: dataframe of transcription factor (output of layer_2_DB)
    pathway: pathway of interest

    Returns
    -------
    tf_list: list of TFs for the pathway

    '''
    start_list = rec_dct[pathway]
    tf_list = list(set(list(tf_df[tf_df['from'].isin(start_list)]['to'])))
    return tf_list

def get_targets(tf_list, tar_df):
    tar_list = list(set(list(tar_df[tar_df['from'].isin(tf_list)]['to'])))
    return tar_list

def get_counts(adata, unspliced=False):
    '''
    EXtract gene names and average counts (unspliced and spliced, imputated)

    Parameters
    ----------
    adata: anndata object

    Returns
    -------
    genes: list of gene names
    counts: numpy array of average counts

    '''
    genes = list(adata.var_names)

    if unspliced:
        counts = np.mean(adata.layers['Mu'] + adata.layers['Ms'], axis=0)
    else:
        counts = np.mean(adata.X.toarray())

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
    # return the top TF and DT for all the pathways in 'pathways' from adata
    # load the database for ligand, receptor, and TF-receptor:

    # lig_dct, rec_dct = cellchat_DB(species=species)

    rec_dct = {}
    for p in pathways:
        rec_dct[p] = adata.uns['pathways'][p]['receptors']

    tf_df = layer_2_DB(species=species)
    # target_df = layer_3_DB()

    # compute average gene count to select
    count_df = get_counts(adata, unspliced=unspliced)

    tf_dict, tar_dict = {}, {}
    for p in pathways:
        if p in input_target.keys():
            tf_list = input_target[p]
        else:
            tf_list = get_TF(rec_dct, tf_df, p)
        tf_dict[p] = select_top_tf(count_df, tf_list, n=top)

        # tar_list = get_targets(tf_dict[p], target_df)
        # tar_dict[p] = select_top_tf(count_df, tar_list)

    adata.uns['TF'] = tf_dict
    # adata.uns['targets'] = tar_dict


