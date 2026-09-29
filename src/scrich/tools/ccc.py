"""Cell-cell communication (CCC) matrix between cell states and communication modes."""

import numpy as np
from numba import jit

############################################################################
#
#### basic functions to model CCI
#
############################################################################

@jit(nopython=True)
def mass_act(R, L, K=0.5):
    return (L*R)/(K + L*R)

@jit(nopython=True)
def alpha(R, L):
    if L==0. or R==0.:
        return 0
    else:
        return np.exp(-1 / (L * R))

@jit(nopython=True)
def beta(T_act):
    if T_act==0.:
        return 0
    else:
        return np.exp(-1/T_act)



############################################################################
#
#### functions to model individual cell CCI with/without the target genes
#
############################################################################
@jit(nopython=True)
def P_pair_LR(R, L, model='mass_action'):
    if model=='mass_action':
        return mass_act(R, L)
    else:
        return alpha(R, L)

@jit(nopython=True)
def P_pair_LRU(R, L, T_act, model='mass_action'):
    # return alpha(R, L)*K1(L, R, T_act)*beta(T_act)
    if model=='mass_action':
        return mass_act(R, L)*beta(T_act)
    else:
        return alpha(R, L)*beta(T_act)




def sign_combs(labels, combs, lig, rec, t_act=None, include_target=False, min_cells=5, model='mass_action', normalize=True):
    ccc_mat = np.zeros((len(combs), len(combs)))
    for i in range(len(combs)):
        lig_sel = lig[labels == combs[i]]
        for j in range(len(combs)):
            rec_sel = rec[labels == combs[j]]
            if include_target:
                tar_sel = t_act[labels == combs[j]]
            if lig_sel.size<min_cells or rec_sel.size<min_cells:
                continue
            else:
                l, r = np.mean(lig_sel), np.mean(rec_sel)
                if include_target:
                    t = np.mean(tar_sel)
                    ccc_mat[i][j] = P_pair_LRU(r, l, t, model=model)
                else:
                    ccc_mat[i][j] = P_pair_LR(r, l, model=model)
    if normalize:
        ccc_mat = ccc_mat/np.sum(ccc_mat)
    return ccc_mat


def compute_ccc_matrix(adata, pathway, key='clusters', include_target=False, use_velocity=False, perm_test=False,
                       min_cells=5, model='mass_action', conversion=True, moments=False):
    '''Compute the CCC matrix at meta cell resolution for a given pathway.
    the matrix and metacell states are stored in adata.uns['ccc_mat'][pathway].

    Parameters
    ----------
    adata: scRNA-seq data object
    pathway: the CCC pathway of interest
    key: adata.obs key for cell annotations
    include_target: if true, include downstream targets of the pathway
    use_velocity: if true, use RNA velocity to estimate target gene response
    perm_test: if TRue, split cell types into meta-cells only if the permutation test is significant
    min_cells: minimum number of cells to consider a metacell in the calculation
    model: mass_action or diffucion model of CCC
    conversion: if true, reverse log-scaling of gene expression data before using gene expression for CCC estimation
    moments: if true, use the imputed layer of adata instead or counts

    Returns
    -------
    None

    '''
    if model not in ('mass_action', 'diffusion'):
        raise ValueError(f"model must be 'mass_action' or 'diffusion', got {model!r}")
    if use_velocity and 'velocity' not in adata.layers:
        raise KeyError("use_velocity=True requires adata.layers['velocity'] (e.g. from scvelo.tl.velocity).")
    if pathway + '_modes' not in adata.obs:
        raise KeyError(f"adata.obs[{pathway + '_modes'!r}] not found. Run all_path_sim() for this pathway first.")
    if include_target and pathway not in adata.uns.get('TF', {}):
        raise KeyError(f"No downstream genes for {pathway!r} in adata.uns['TF']. Run import_database() first.")
    # generate list of cell labels (state + mode)
    # some state-mode combination might not exist, but still need to be defined to have a well-defined matrix
    states = list(adata.obs[key])
    modes = list(adata.obs[pathway + '_modes'])

    if perm_test:
        labels = []
        types, p_vals = adata.uns['permutation_test'][0], np.asarray(adata.uns['permutation_test'][1][pathway])
        perm_dict = dict(zip(types, p_vals))

        for s, m in zip(states, modes):
            if perm_dict[s] < 0.05:
                labels.append(s + '-' + str(m))
            else:
                labels.append(s)
        labels = np.asarray(labels)

        combs = sorted(list(set(labels)))
    else:
        labels = np.asarray([states[i] + '-' + str(modes[i]) for i in range(len(states))])

        # define all state-mode combinations:
        combs = []
        for s in sorted(set(states)):
            for m in sorted(set(modes)):
                combs.append(s + '-' + str(m))


    lig, rec = np.asarray(adata.obs[pathway + '_lig']), np.asarray(adata.obs[pathway + '_rec'])
    if conversion:
        lig, rec = 10 ** lig, 10 ** rec
    lig, rec = lig / np.amax(lig), rec / np.amax(rec)

    if include_target:
        tar_list = adata.uns['TF'][pathway]

        if use_velocity:
            # compute the target velocity array
            t_act = adata[:, tar_list].layers['velocity'].mean(axis=1)

            # are these two lines necessary (save the expression of targets in adata.obs)??
            if pathway + '_tar' not in adata.obs.keys():
                adata.obs[pathway + '_tar'] = np.asarray(t_act)
            if conversion:
                t_act = 10 ** t_act
            t_act = t_act / np.amax(t_act)

        else:
            # compute the target expression array
            if moments:
                t_act = adata[:, tar_list].layers['Mu'].mean(axis=1) + adata[:, tar_list].layers['Ms'].mean(axis=1)
            else:
                t_act = adata[:, tar_list].X.toarray().mean(axis=1)
            # are these two lines necessary (save the expression of targets in adata.obs)??
            if pathway + '_tar' not in adata.obs.keys():
                adata.obs[pathway + '_tar'] = np.asarray(t_act)
            if conversion:
                t_act = 10 ** t_act
            t_act = t_act / np.amax(t_act)

        ccc_mat = sign_combs(labels, combs, lig, rec, t_act=t_act, min_cells=min_cells, include_target=include_target,
                             model=model, normalize=True)
    else:
        ccc_mat = sign_combs(labels, combs, lig, rec, t_act=None, min_cells=min_cells, include_target=include_target,
                             model=model, normalize=True)

    if 'ccc_mat' not in adata.uns.keys():
        adata.uns['ccc_mat'] = {}
    adata.uns['ccc_mat'][pathway] = {'states': combs, 'mat': ccc_mat}
