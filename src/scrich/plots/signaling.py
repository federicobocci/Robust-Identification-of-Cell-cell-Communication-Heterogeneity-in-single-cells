import numpy as np
import matplotlib.pyplot as plt

def single_path_roles(adata, pathway, verbose=False,
                      xlim=None, ylim=None, pos={}, show_top=10,
                      figsize=(4,4), showfig=False, savefig=True, figname='single_path_role.pdf', format='pdf'):
    sign_df = adata.uns['sign_strength'][pathway]
    states = list(sign_df.index)
    incoming, outgoing = np.asarray(sign_df['incoming']), np.asarray(sign_df['outgoing'])
    ind = np.flip(np.argsort(incoming+outgoing))[0:show_top]

    fig = plt.figure(figsize=figsize)

    plt.scatter(incoming, outgoing)
    plt.xlabel('Incoming')
    plt.ylabel('Outgoing')
    # for i in range(show_top):
    #     plt.text(incoming[ind[i]], outgoing[ind[i]], states[ind[i]])

    for i in range(show_top):
        if states[ind[i]] in pos.keys():
            x, y = pos[states[ind[i]]][0], pos[states[ind[i]]][1]
        else:
            x, y = incoming[ind[i]], outgoing[ind[i]]
        plt.text(x, y, states[ind[i]])
        if verbose:
            print(states[ind[i]], x, y)

    plt.title(pathway + ' pathway')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    plt.tight_layout()
    if showfig:
        plt.showfig()
    if savefig:
        plt.savefig(figname, format=format)

def single_state_roles(adata, state, key='clusters', verbose=False,
                       xlim=None, ylim=None, pos={}, show_top=10,
                       figsize=(4,4), showfig=False, savefig=True, figname='state_role.pdf', format='pdf'):
    states = sorted(list(set(list(adata.obs[key]))))
    ind = states.index(state)

    path, inc, out = [], [], []
    for k in adata.uns['sign_strength'].keys():
        sign_df = adata.uns['sign_strength'][k]
        incoming, outgoing = np.asarray(sign_df['incoming']), np.asarray(sign_df['outgoing'])
        path.append(k)
        inc.append(incoming[ind])
        out.append(outgoing[ind])
    inc, out = np.asarray(inc), np.asarray(out)

    fig = plt.figure(figsize=figsize)

    plt.scatter(inc, out)
    plt.xlabel('Incoming')
    plt.ylabel('Outgoing')

    ind = np.flip(np.argsort(inc+out))[0:show_top]

    for i in range(show_top):
        if path[ind[i]] in pos.keys():
            x, y = pos[path[ind[i]]][0], pos[path[ind[i]]][1]
        else:
            x, y = inc[ind[i]], out[ind[i]]
        plt.text(x, y, path[ind[i]])
        if verbose:
            print(path[ind[i]], x, y)
    plt.title(state)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    plt.tight_layout()
    if showfig:
        plt.showfig()
    if savefig:
        plt.savefig(figname, format=format)