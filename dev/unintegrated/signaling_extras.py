"""Functions removed from the scRICH package (kept for reference).

- ``pathway_strength``: overall incoming/outgoing strength from the CCC matrix (was marked "currently not used").
- ``compute_ccc_matrix_CellchatBenchmark``: CCC matrix on cell annotations only, used for benchmarking
  against CellChat. Note: it computes ``ccc_mat`` but never stores it.

They depend on ``sign_combs`` from ``scrich.tools.signaling_func``.
"""
import numpy as np
import pandas as pd

from scrich.tools.signaling_func import sign_combs

def pathway_strength(adata, pathway, key='clusters'):
    '''Computes the overall incoming and outgoing pathway strength from the CCC matrix (currently not used).
    result is stored as a dataframe in adata.uns['sign_strength'][pathway].

    Parameters
    ----------
    adata: scRNA-seq data object
    pathway: pathway of interest
    key: adata.obs key for cell type annotations

    Returns
    -------

    '''
    combs = adata.uns['ccc_mat'][pathway]['states']
    ccc_mat = adata.uns['ccc_mat'][pathway]['mat']
    outward, inward = np.zeros(len(combs)), np.zeros(len(combs))
    for i in range(len(combs)):
        outward[i] =  np.sum(ccc_mat[i])
        inward[i] = np.sum(ccc_mat[:, i])

    n_modes = len(set(adata.obs[pathway + '_modes']))
    n_states = int(len(combs)/n_modes)
    incoming, outgoing = np.zeros(n_states), np.zeros(n_states)
    for i in range(n_states):
        incoming[i] = np.sum(inward[i*n_modes:(i+1)*n_modes])
        outgoing[i] = np.sum(outward[i * n_modes:(i + 1) * n_modes])

    df = pd.DataFrame(data=[incoming, outgoing], index=['incoming', 'outgoing'],
                      columns=sorted(list(set(adata.obs[key])))).transpose()

    if 'sign_strength' not in adata.uns.keys():
        adata.uns['sign_strength'] = {}
    adata.uns['sign_strength'][pathway] = df




######################
#
# this function runs the ccc matrix calculation using only the cell state annotations, not the CCC communication modes
# this function was used only for benchmarking purposes - it is not needed to run scRICH and could be moved to a separate benchmarking script in the future
#
#######################

def compute_ccc_matrix_CellchatBenchmark(adata, pathway, key='clusters', include_target=False, min_cells=5, model='mass_action', conversion=True, moments=False):
    assert model == 'mass_action' or model == 'diffusion', "Invalid model parameter, choose between model=='mass action' or model=='diffusion'"
    # generate list of cell labels (state + mode)
    # some state-mode combination might not exist, but still need to be defined to have a well-defined matrix

    labels = np.asarray(adata.obs[key])
    combs = sorted(list(set(labels)))

    lig, rec = np.asarray(adata.obs[pathway + '_lig']), np.asarray(adata.obs[pathway + '_rec'])
    if conversion:
        lig, rec = 10 ** lig, 10 ** rec
    lig, rec = lig / np.amax(lig), rec / np.amax(rec)

    if include_target:
        # compute the target array
        tar_list = adata.uns['TF'][pathway]
        if moments:
            t_act = adata[:, tar_list].layers['Mu'].mean(axis=1) + adata[:, tar_list].layers['Ms'].mean(axis=1)
        else:
            t_act = adata[:, tar_list].X.toarray().mean(axis=1)
        if pathway + 'tar' not in adata.obs.keys():
            adata.obs[pathway + '_tar'] = np.asarray(t_act)
        if conversion:
            t_act = 10 ** t_act
        t_act = t_act / np.amax(t_act)

        ccc_mat = sign_combs(labels, combs, lig, rec, t_act=t_act, min_cells=min_cells, include_target=include_target,
                             model=model, normalize=True)
    else:
        ccc_mat = sign_combs(labels, combs, lig, rec, t_act=None, min_cells=min_cells, include_target=include_target,
                             model=model, normalize=True)

    return combs, ccc_mat
