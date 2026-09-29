"""Pathway selection: find ligands/receptors of CCC pathways detected in the data and rank pathways."""

import numpy as np

from .. import db
from .._logging import logger
from ._utils import _resolve_species, get_list, list_intersect


def find_genes(adata,
               pathway,
               species=None,
               save_genes=True,
               return_lists=False,
               print_info=True,
               human=None
               ):
    '''Identify receptors and ligands of a given pathway that are detected in adata
    Identified receptors and ligands for pathway are stored in adata.uns['pathways']

    Parameters
    ----------
    adata: anndata object of gene counts
    pathway: pathway of interest
    species: 'human' (default) or 'mouse'
    save_genes: if True, save identified genes in adata.uns['pathways'] (default=True)
    return_lists: if True, returns lists of common receptors and ligands (default=False)
    print_info: if True, print information (default=True)
    human: deprecated, use species

    Returns
    -------
    None

    '''
    species = _resolve_species(species, human)
    lig = db.load_pathway_genes('ligand', species=species)
    rec = db.load_pathway_genes('receptor', species=species)

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
        logger.info('Identified %d receptors in the %s pathway:', len(rec_common), pathway)
        logger.info(' '.join(rec_common))
        logger.info('Identified %d ligands in the %s pathway:', len(lig_common), pathway)
        logger.info(' '.join(lig_common))


def pathways_overview(adata, species=None, moments=False, human=None):
    '''Overview of pathways in the database with both Receptors and Ligands detected in adata
    Information on identified pathways is printed

    Parameters
    ----------
    adata: anndata object of gene counts
    species: 'human' (default) or 'mouse'
    moments: if True, use the imputed layers Mu + Ms for expression, otherwise adata.X
    human: deprecated, use species

    Returns
    -------
    None

    '''
    species = _resolve_species(species, human)
    lig = db.load_pathway_genes('ligand', species=species)
    rec = db.load_pathway_genes('receptor', species=species)
    path_list = sorted(set(lig['pathway']))

    overview_genes = {}
    overview_expression = {}

    count = 0
    logger.info('Scanning for cell-cell communication pathways...')
    for p in path_list:
        rec_list = get_list(rec, p)
        lig_list = get_list(lig, p)
        if len(rec_list)==0 or len(lig_list)==0:
            continue
        else:
            rec_common, lig_common = find_genes(adata, p, species=species, save_genes=True, print_info=False, return_lists=True)
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

    logger.info('Identified %d pathways with at least one Receptor and one Ligand detected', count)
    adata.uns['paths_overview'] = {'gene_number': overview_genes, 'gene_expression': overview_expression}


def select_pathways(adata, species=None, method='expression', n=10, human=None):
    '''Select the top CCC pathways on which to run the scRICH analysis.
    The list of selected pathways is stored in adata.uns['selected_pathways']

    Parameters
    ----------
    adata: scRNA-seq data object
    species: unused, kept for backward compatibility (pathways come from adata.uns['paths_overview'])
    method: 'expression' (total gene expression of rec+lig genes) of 'number' (number of detected rec+lig genes)
    n: number of pathway to select

    Returns
    -------
    None

    '''
    if human is not None:
        _resolve_species(species, human)  # deprecation warning only
    if 'paths_overview' not in adata.uns:
        raise ValueError("Run scrich.tl.pathways_overview() before selecting pathways.")
    if method not in ('expression', 'number'):
        raise ValueError(f"method must be 'expression' or 'number', got {method!r}")

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
