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
    r"""Find the receptors and ligands of a pathway that are detected in the data.

    The pathway's genes are taken from the bundled CellChatDB tables
    (:func:`scrich.db.load_pathway_genes`); receptor complexes such as ``TGFBR1_TGFBR2`` are split into
    their subunits. A gene counts as detected if it is in ``adata.var_names``.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix (cells x genes).
    pathway : str
        Name of the CCC pathway, e.g. ``'TGFb'`` or ``'IGF'``.
    species : {'human', 'mouse'}, default 'human'
        Species of the ligand-receptor database.
    save_genes : bool, default True
        Store the detected genes in ``adata.uns['pathways'][pathway]``.
    return_lists : bool, default False
        Return the detected receptors and ligands.
    print_info : bool, default True
        Report the detected genes (see :func:`scrich.set_verbosity`).
    human : bool, optional
        Deprecated, use ``species``.

    Returns
    -------
    tuple of list or None
        ``(receptors, ligands)`` if ``return_lists=True``, otherwise ``None``.
        With ``save_genes=True`` sets ``adata.uns['pathways'][pathway] = {'receptors': [...], 'ligands': [...]}``.

    Examples
    --------
    >>> import scrich as ri
    >>> ri.tl.find_genes(adata, "IGF", species="mouse")  # doctest: +SKIP
    """
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
    r"""Scan all pathways of the database for detected receptors and ligands.

    For every pathway with at least one receptor and one ligand detected in the data, records the number
    of detected genes and their average expression. This is usually the first step of a scRICH analysis;
    the pathways to analyse are then chosen with :func:`scrich.tl.select_pathways` (called by
    :func:`scrich.tl.all_path_sim`) or by hand.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix (cells x genes), with log-normalized expression in ``adata.X``.
    species : {'human', 'mouse'}, default 'human'
        Species of the ligand-receptor database.
    moments : bool, default False
        Use the imputed layers ``adata.layers['Mu'] + adata.layers['Ms']`` (e.g. from
        :func:`scvelo.pp.moments`) instead of ``adata.X``.
    human : bool, optional
        Deprecated, use ``species``.

    Returns
    -------
    None
        Sets the following fields:

        - ``adata.uns['pathways'][pathway]``: detected ``'receptors'`` and ``'ligands'`` of each pathway.
        - ``adata.uns['paths_overview']``: ``{'gene_number': {pathway: [n_receptors, n_ligands]},
          'gene_expression': {pathway: [mean receptor expression, mean ligand expression]}}``.

    See Also
    --------
    scrich.pl.pathways_overview : Plot the detected pathways.

    Examples
    --------
    >>> import scrich as ri
    >>> ri.tl.pathways_overview(adata, species="mouse")  # doctest: +SKIP
    >>> ri.pl.pathways_overview(adata, top=10)  # doctest: +SKIP
    """
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
    r"""Select the top pathways for the analysis.

    Pathways are ranked using the summary computed by :func:`scrich.tl.pathways_overview`.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.pathways_overview`.
    species : {'human', 'mouse'}, optional
        Not used; kept for backward compatibility.
    method : {'expression', 'number'}, default 'expression'
        Rank pathways by the average expression of their receptors and ligands (``'expression'``) or by the
        number of detected receptors and ligands (``'number'``).
    n : int, default 10
        Number of pathways to select.
    human : bool, optional
        Deprecated, use ``species``.

    Returns
    -------
    None
        Sets ``adata.uns['selected_pathways']``: list of the ``n`` top pathways.
    """
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
