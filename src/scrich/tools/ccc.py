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
    r"""Mass-action interaction :math:`LR / (K + LR)`.

    Parameters
    ----------
    R, L : float
        Receptor and ligand levels.
    K : float, default 0.5
        Half-saturation constant.

    Returns
    -------
    float
        Interaction strength.
    """
    return (L*R)/(K + L*R)

@jit(nopython=True)
def alpha(R, L):
    r"""Diffusion-model interaction :math:`\exp(-1/(LR))` (0 if :math:`L` or :math:`R` is 0).

    Parameters
    ----------
    R, L : float
        Receptor and ligand levels.

    Returns
    -------
    float
        Interaction strength.
    """
    if L==0. or R==0.:
        return 0
    else:
        return np.exp(-1 / (L * R))

@jit(nopython=True)
def beta(T_act):
    r"""Response of downstream genes :math:`\exp(-1/T)` (0 if :math:`T` is 0).

    Parameters
    ----------
    T_act : float
        Downstream-gene level.

    Returns
    -------
    float
        Response factor.
    """
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
    r"""Interaction between a sender and a receiver from ligand and receptor levels.

    Parameters
    ----------
    R, L : float
        Receptor and ligand levels.
    model : {'mass_action', 'diffusion'}, default 'mass_action'
        Interaction model.

    Returns
    -------
    float
        Interaction strength.
    """
    if model=='mass_action':
        return mass_act(R, L)
    else:
        return alpha(R, L)

@jit(nopython=True)
def P_pair_LRU(R, L, T_act, model='mass_action'):
    # return alpha(R, L)*K1(L, R, T_act)*beta(T_act)
    r"""Interaction weighted by the response of downstream genes in the receiver.

    Parameters
    ----------
    R, L : float
        Receptor and ligand levels.
    T_act : float
        Downstream-gene level of the receiver.
    model : {'mass_action', 'diffusion'}, default 'mass_action'
        Interaction model.

    Returns
    -------
    float
        Interaction strength.
    """
    if model=='mass_action':
        return mass_act(R, L)*beta(T_act)
    else:
        return alpha(R, L)*beta(T_act)




def sign_combs(labels, combs, lig, rec, t_act=None, include_target=False, min_cells=5, model='mass_action', normalize=True):
    r"""CCC matrix between groups of cells.

    Parameters
    ----------
    labels : numpy.ndarray
        Group label of every cell.
    combs : list of str
        Groups (rows and columns of the matrix).
    lig, rec : numpy.ndarray
        Scaled ligand and receptor expression of every cell.
    t_act : numpy.ndarray, optional
        Scaled downstream-gene expression of every cell (with ``include_target=True``).
    include_target : bool, default False
        Weight the interaction by the downstream-gene expression of the receiving group.
    min_cells : int, default 5
        Groups with fewer cells get zero interactions.
    model : {'mass_action', 'diffusion'}, default 'mass_action'
        Interaction model, see :func:`scrich.tl.compute_ccc_matrix`.
    normalize : bool, default True
        Divide the matrix by its sum.

    Returns
    -------
    numpy.ndarray
        Groups x groups matrix; entry ``(i, j)`` is the signal sent by group ``i`` (ligands) to group ``j``
        (receptors).
    """
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
    r"""Cell-cell communication matrix of a pathway between cell states.

    Cells are grouped into *states*, the combinations of cell type and communication mode (e.g. ``'Er-1-0'``
    for mode 0 of cell type ``'Er-1'``). The interaction from state :math:`i` to state :math:`j` is computed
    from the mean ligand expression :math:`L_i` of the sender and the mean receptor expression :math:`R_j`
    of the receiver:

    - ``model='mass_action'``: :math:`P_{ij} = L_i R_j / (K + L_i R_j)` with :math:`K = 0.5`;
    - ``model='diffusion'``: :math:`P_{ij} = \exp(-1 / (L_i R_j))`.

    With ``include_target=True`` the interaction is multiplied by :math:`\exp(-1/T_j)`, where :math:`T_j` is
    the mean expression (or RNA velocity) of the pathway's downstream genes in the receiver. Expression
    values are scaled to a maximum of 1 and the matrix is normalized to sum 1.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.all_path_sim`.
    pathway : str
        Pathway of interest.
    key : str, default 'clusters'
        Key in ``adata.obs`` with cell-type annotations.
    include_target : bool, default False
        Weight interactions by the downstream genes in ``adata.uns['TF'][pathway]``
        (see :func:`scrich.tl.import_database`).
    use_velocity : bool, default False
        Use the RNA velocity of the downstream genes (``adata.layers['velocity']``) instead of their expression.
    perm_test : bool, default False
        Split a cell type into its modes only if it is significantly heterogeneous according to
        :func:`scrich.tl.permutation_testing` (p < 0.05); otherwise the cell type is one state.
    min_cells : int, default 5
        States with fewer cells get zero interactions.
    model : {'mass_action', 'diffusion'}, default 'mass_action'
        Interaction model.
    conversion : bool, default True
        Transform expression values :math:`x` to :math:`10^x` before scaling.
    moments : bool, default False
        Use ``adata.layers['Mu'] + adata.layers['Ms']`` for downstream-gene expression instead of ``adata.X``.

    Returns
    -------
    None
        Sets ``adata.uns['ccc_mat'][pathway] = {'states': [...], 'mat': ...}``: the state labels and the
        states x states matrix (rows: senders, columns: receivers). With ``include_target=True`` also sets
        ``adata.obs[pathway + '_tar']`` if missing.

    See Also
    --------
    scrich.pl.chord_diagram : Chord diagram of the matrix.
    scrich.pl.coarse_grained_map : The matrix on the embedding.

    Examples
    --------
    >>> ri.tl.compute_ccc_matrix(adata, "IGF", key="celltype", min_cells=10)  # doctest: +SKIP
    """
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
