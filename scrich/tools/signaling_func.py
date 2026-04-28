'''
functions to model cell-cell communication at the sub-cluster level
option to include downstream target genes for refined CCI rediction
'''

import numpy as np
import pandas as pd
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

@jit(nopython=True)
def beta_RNAVel(T_act):
    if T_act<=0.:
        return 0
    else:
        return np.exp(-1/T_act)

@jit(nopython=True)
def gamma(T_inh):
    return np.exp(-T_inh)

@jit(nopython=True)
def K1(L, R, T_act):
    if (L==0. and T_act==0.) or (R==0. and T_act==0.):
        return 0.
    else:
        return alpha(L, R)/( alpha(L, R) + beta(T_act) )

@jit(nopython=True)
def K1_RNAVel(L, R, V_act):
    # this line is not correct anymore for RNA velocity case:
    if (L==0. and V_act==0.) or (R==0. and V_act==0.):
        return 0.
    else:
        return alpha(L, R)/( alpha(L, R) + beta_RNAVel(V_act) )

@jit(nopython=True)
def K2(L, R, T_inh):
    return alpha(L, R)/( alpha(L, R) + gamma(T_inh) )

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

@jit(nopython=True)
def P_pair_LRU_RNAVel(R, L, V_act):
    return alpha(L, R)*K1_RNAVel(L, R, V_act)*beta_RNAVel(V_act)

@jit(nopython=True)
def P_pair_LRD(R, L, T_inh):
    return alpha(L, R)*K2(L, R, T_inh)*gamma(T_inh)

@jit(nopython=True)
def P_pair_all(R, L, T_act, T_inh):
    return alpha(L, R)*K1(L, R, T_act)*beta(T_act)*K2(L, R, T_inh)*gamma(T_inh)


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

'''
to include the possibility for RNA velocity instead of targets, remove the include_target argument and instead define:
downstream (list): None (default), target, velocity
it will need some checks (velocities exist...) to avoid errors 

automatically switch to no-target of there are no target for the pathway?
'''
def compute_ccc_matrix(adata, pathway, key='clusters', include_target=False, use_velocity=False, perm_test=False, min_cells=5, model='mass_action', conversion=True, moments=False):
    assert model=='mass_action' or model=='diffusion', "Invalid model parameter, choose between model=='mass action' or model=='diffusion'"
    if use_velocity==True:
        assert 'velocity' in adata.layers.keys(), "'velocity' key not found in adata.layers"
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


# run the ccc matrix calculation using only the cell state annotations
# this does not belong here but rather in a benchmarking notebook
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


def pathway_strength(adata, pathway, key='clusters'):
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
