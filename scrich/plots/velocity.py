'''
plotting functions for overlaying CCC and RNA velocity
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import pandas as pd
# import scvelo as scv
try:
    import scvelo as scv
except AttributeError:
    print('No such attribute')

from seaborn import kdeplot


def extract_neighbors(adata, key='distances'):
    assert key == 'distances' or key == 'connectivities', "Please choose between key=='distances' or key=='connectivities'"
    dist = adata.obsp[key]
    neighbors = [dist[i].indices for i in range(adata.shape[0])]
    return neighbors


def compute_sign_prob(adata, pathway, key='distances'):
    neighbors = extract_neighbors(adata, key=key)
    modes = np.asarray(adata.obs[pathway + '_modes'])

    unique_modes = sorted(set(modes))
    sign_prob = np.zeros((modes.size, len(unique_modes)))
    for i in range(modes.size):
        neigh_modes = modes[neighbors[i]]
        counts = np.asarray([neigh_modes[neigh_modes == m].size for m in unique_modes])
        counts = counts / np.sum(counts)
        sign_prob[i] = counts
    return unique_modes, sign_prob

def compute_pattern_prob(adata, key='distances'):
    neighbors = extract_neighbors(adata, key=key)
    modes = np.asarray(adata.obs['sign_pattern'])

    unique_modes = sorted(set(modes))
    sign_prob = np.zeros((modes.size, len(unique_modes)))
    for i in range(modes.size):
        neigh_modes = modes[neighbors[i]]
        counts = np.asarray([neigh_modes[neigh_modes == m].size for m in unique_modes])
        counts = counts / np.sum(counts)
        sign_prob[i] = counts
    return unique_modes, sign_prob

def neighbor_avg(v, neigh):
    v_avg = np.zeros(v.size)
    for i in range(v_avg.size):
        neigh_v = v[neigh[i]]
        v_avg[i] = np.mean(neigh_v)
    return v_avg


###################################
#
# RNA velocity-related functions

def plot_maps(adata, pathway, key='distances', basis='umap', panel_height = 4, panel_length = 4,
              linewidth=0.5, arrow_style='->', mode_to_plot='all', fontsize=7,
              showfig=False, savefig=True, figname='velocity_plot', format='png'):

    # assert plot_style=='together' or plot_style=='single', "Choose between plot_style=='together' or plot_style=='single'"

    mode_name, sign_prob = compute_sign_prob(adata, pathway, key=key)
    n_modes = sign_prob.shape[1]

    modes = np.asarray(adata.obs[pathway + '_modes'])

    # signaling state needs a better definition...
    sign_state = np.zeros(modes.size)
    for i in range(sign_state.size):
        sign_state[i] = np.argmax(sign_prob[i])

    coords = np.asarray(adata.obsm['X_'+basis])
    x, y = coords[:, 0], coords[:, 1]

    colors = ['Greens', 'Oranges', 'Blues', 'Reds', 'Purples']

    if mode_to_plot=='all':
        fig = plt.figure(figsize=(n_modes * panel_length, panel_height))

    for i in range(n_modes):

        if mode_to_plot==mode_name[i]:
            fig = plt.figure(figsize=(panel_length, panel_height))
            ax = plt.subplot(111)
        else:
            ax = plt.subplot2grid((1, n_modes), (0, i), rowspan=1, colspan=1)

        ax.axis('off')
        df = pd.DataFrame.from_dict({'x': x, 'y': y, 'prob': sign_prob[:, i]})
        kdeplot(ax=ax, data=df, x='x', y='y', weights='prob', fill=True, cmap=colors[i], levels=20, cbar=False)
        scv.pl.velocity_embedding_stream(adata, basis=basis, size=0, linewidth=linewidth, arrow_style=arrow_style,
                                         arrow_size=0.5, ax=ax, show=False, save=False)
        plt.title(mode_name[i], fontsize=fontsize)

        if mode_to_plot==mode_name[i]:
            plt.tight_layout()
            if showfig:
                plt.show()
            if savefig:
                plt.savefig(figname + '_' + mode_name[i] + '.' + format, format=format, dpi=500)

    if mode_to_plot=='all':
        plt.tight_layout()
        if showfig:
            plt.show()
        if savefig:
            plt.savefig(figname + '.' + format, format=format, dpi=500)


def pattern_plot(adata, key='distances', basis='umap', pattern=0, colormap='Reds',
                 showfig=False, savefig=True, figname='pattern_plot.png', format='png'):
    mode_name, sign_prob = compute_pattern_prob(adata, key=key)
    n_modes = sign_prob.shape[1]

    coords = np.asarray(adata.obsm['X_' + basis])
    x, y = coords[:, 0], coords[:, 1]

    # figure parameters
    fig = plt.figure(figsize=(4,4))

    ax = plt.subplot(111)
    ax.axis('off')
    df = pd.DataFrame.from_dict({'x': x, 'y': y, 'prob': sign_prob[:, pattern]})
    kdeplot(ax=ax, data=df, x='x', y='y', weights='prob', fill=True, cmap=colormap, levels=20, cbar=False)
    scv.pl.velocity_embedding_stream(adata, basis=basis, size=0, linewidth=0.5, ax=ax, show=False, save=False)
    plt.title('Pattern ' + str(pattern))

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=500)



def enforce_spacing(x, y, dx, thr):
    x_new, y_new = x.copy(), y.copy()
    dist_count = 0
    for i in range(x.size):
        for j in range(x.size):
            if i != j:
                dist = np.sqrt((y[j] - y[i]) ** 2 + (x[j] - x[i]) ** 2)
                if dist < thr:
                    dist_count = dist_count + 1
                    if x[j] > x[i]:
                        a = (y[j] - y[i]) / (x[j] - x[i])
                        x_new[i], x_new[j] = x[i] - dx, x[j] + dx
                        y_new[i], y_new[j] = x[i] - a * dx, x[j] + a * dx
                    else:
                        a = (y[j] - y[i]) / (x[i] - x[j])
                        x_new[i], x_new[j] = x[i] + dx, x[j] - dx
                        y_new[i], y_new[j] = x[i] - a * dx, x[j] + a * dx
    return x_new, y_new, dist_count


def draw_self_loop(ax, center, radius, facecolor='#2693de', edgecolor='#000000', theta1=-30, theta2=180):
    # Add the ring
    rwidth = 0.02
    ring = patches.Wedge(center, radius, theta1, theta2, width=rwidth)
    # Triangle edges
    offset = 0.02
    xcent = center[0] - radius + (rwidth / 2)
    left = [xcent - offset, center[1]]
    right = [xcent + offset, center[1]]
    bottom = [(left[0] + right[0]) / 2., center[1] - 0.05]
    arrow = plt.Polygon([left, right, bottom, left])
    p = PatchCollection(
        [ring, arrow],
        edgecolor=edgecolor,
        facecolor=facecolor
    )
    ax.add_collection(p)


def coarse_grained_map(adata, pathway, key='clusters', thr=0.01, fontsize=10, update_connection=None,
                       update_horiz_align=None, update_vert_align=None, linewidth=0.25, arrow_style='->',
                       single_source=None, single_target=None, figsize=(4, 4), strip_text=False,
                       showfig=False, savefig=True, figname='coarse_grained.png', format='png'):
    mat = adata.uns['ccc_mat'][pathway]['mat']
    states = sorted(list(set(adata.obs[key])))
    modes = sorted(set(list(adata.obs[pathway + '_modes'])))

    x, y = np.zeros(len(modes) * len(states)), np.zeros(len(modes) * len(states))

    cmap = 'tab20'
    cmap = mpl.color_sequences[cmap]

    x_umap, y_umap = adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1]
    x_scale = np.amax(x_umap) - np.amin(x_umap)
    text_separation = 0.02 * x_scale

    fig = plt.figure(figsize=figsize)

    ax = plt.subplot(111)
    ax.axis('off')

    # states and modes should be ordered in the same way as in the CCC matrix calculation
    node_state = []
    i = 0
    for s in states:
        for m in modes:
            node_state.append(s + '-' + m)
            adata_sel = adata[adata.obs[key] == s].copy()
            adata_sel = adata_sel[adata_sel.obs[pathway + '_modes'] == m]
            x[i], y[i] = np.mean(adata_sel.obsm['X_umap'][:, 0]), np.mean(adata_sel.obsm['X_umap'][:, 1])
            i = i + 1

    # x_new, y_new, count = enforce_spacing(x, y, text_separation, text_separation)
    x_new, y_new = x, y

    # plot the states/modes and their labels
    horizontal_align_dct = {n: 'right' for n in node_state}
    if update_horiz_align:
        for k in update_horiz_align.keys():
            horizontal_align_dct[k] = update_horiz_align[k]
    vertical_align_dct = {n: 'bottom' for n in node_state}
    if update_vert_align:
        for k in update_vert_align.keys():
            vertical_align_dct[k] = update_vert_align[k]
    i = 0
    col_list = []
    for j in range(len(states)):
        for k in range(len(modes)):
            col_list.append(cmap[j])
            if any(mat[i] > thr) or any(mat[:, i] > thr):
                # make size scale with population size
                plt.scatter(x_new[i], y_new[i], c=cmap[j], s=100, edgecolors='k', linewidths=0.5)
                if not strip_text:
                    plt.text(x_new[i] - text_separation, y_new[i], states[j] + '-' + modes[k], fontsize=fontsize,
                             horizontalalignment=horizontal_align_dct[node_state[i]],
                             verticalalignment=vertical_align_dct[node_state[i]], fontweight="bold")
            i = i + 1

    scv.pl.velocity_embedding_stream(adata, basis='umap', size=0, arrow_size=0.5, linewidth=linewidth, ax=ax, show=False,
                                     arrow_style=arrow_style,
                                     save=False,
                                     legend_loc=False, title='')

    connections = []
    for n in node_state:
        for m in node_state:
            connections.append('start: ' + n + '; end: ' + m)
    connstyle_dct = {c: "arc3, rad=0.5" for c in connections}

    if update_connection:
        for k in update_connection.keys():
            connstyle_dct[k] = update_connection[k]

    if single_source:
        ind = node_state.index(single_source)
        mat_update = np.zeros((len(node_state), len(node_state)))
        mat_update[ind] = mat[ind]
        mat = mat_update

    if single_target:
        ind = node_state.index(single_target)
        mat_update = np.zeros((len(node_state), len(node_state)))
        mat_update[:, ind] = mat[:, ind]
        mat = mat_update

    for i in range(x.size):
        for j in range(x.size):
            if mat[i][j] > thr:
                if i == j:
                    # draw_self_loop(ax, center=(x_new[i], y_new[i]), radius=1)
                    style = patches.ArrowStyle('Fancy', head_length=5, head_width=5,
                                               tail_width=1 + 9 * (mat[i][j] / np.amax(mat)))
                    a = patches.FancyArrowPatch((x_new[i] + 0.5, y_new[i]), (x_new[j] - 0.5, y_new[j]),
                                                connectionstyle="arc3, rad=2.0",
                                                alpha=0.5, arrowstyle=style, color=col_list[i])

                else:
                    style = patches.ArrowStyle('Fancy', head_length=5, head_width=5,
                                               tail_width=1 + 9 * (mat[i][j] / np.amax(mat)))
                    a = patches.FancyArrowPatch((x_new[i], y_new[i]), (x_new[j], y_new[j]),
                                                connectionstyle=connstyle_dct[
                                                    'start: ' + node_state[i] + '; end: ' + node_state[j]],
                                                alpha=0.5, arrowstyle=style, color=col_list[i])
                plt.gca().add_patch(a)


    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=500)


###################################
#
# pseudotime-related functions

def moving_avg(pst, y, npoints):
    bins = np.linspace(0, 1, npoints + 1)
    y_avg = np.zeros(bins.size - 1)
    for i in range(bins.size - 1):
        pst_sel, y_sel = pst[pst >= bins[i]], y[pst >= bins[i]]
        pst_sel, y_sel = pst_sel[pst_sel <= bins[i + 1]], y_sel[pst_sel <= bins[i + 1]]
        y_avg[i] = np.mean(y_sel)
    return y_avg


def state_dist_pst(adata, key='clusters', pst_key='velocity_pseudotime', order=None, alp=0.5, colors='b',
                   title='State distribution', xlab='pseudotime', ax=None, figsize=(4, 2), showfig=False, savefig=True,
                   figname='state_dist_pst.pdf', format='pdf', dpi=300):
    pst = np.asarray(adata.obs[pst_key])
    clusters = np.asarray(adata.obs[key])

    if order == None:
        order = sorted(list(set(clusters)))

    if isinstance(colors, str):
        colorlist = [colors for o in order]
    elif isinstance(colors, list):
        colorlist = colors
    else:
        raise TypeError('Input colors as a single string or list of accepted matplotlib colors')

    data = []
    for i in range(len(order)):
        data.append(pst[clusters == order[i]])

    new_ax = False
    if not ax:
        plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        new_ax = True

    vp = ax.violinplot(data, vert=False, showextrema=False)
    i = 0
    for pc in vp['bodies']:
        pc.set_facecolor(colorlist[i])
        pc.set_edgecolor(colorlist[i])
        pc.set_alpha(alp)
        i = i + 1
    plt.yticks(np.arange(1, len(order) + 1, 1), order)
    plt.xlabel(xlab)
    if title:
        plt.title(title)

    if new_ax:
        plt.tight_layout()
        if showfig:
            plt.show()
        if savefig:
            plt.savefig(figname, format=format, dpi=dpi)



def sign_prob_plot(adata, pathway, key='velocity_pseudotime', legend_font=10, title=None, npoints=10, xlab='pseudotime', ax=None,
                   figsize=(4, 2), showfig=False, savefig=True, figname='pst.pdf', format='pdf', dpi=300, return_curve=False):
    pst = np.asarray(adata.obs[key])

    new_ax = False
    if not ax:
        plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        new_ax = True

    mode_name, sign_prob = compute_sign_prob(adata, pathway)

    x = np.linspace(0, 1, npoints)
    result_dict = {'x': x}

    n_modes = sign_prob.shape[1]
    colors = ['b', 'g', 'r']

    for i in range(n_modes):
        y = moving_avg(pst, sign_prob[:, i], npoints)
        plt.plot(x, y, 'o-', color=colors[i], label=mode_name[i])
        result_dict[mode_name[i]] = y

    # plt.plot(x, moving_avg(pst, sign_prob[:, 0], npoints), 'o-', color='b', label=mode_name[0])
    # plt.plot(x, moving_avg(pst, sign_prob[:, 1], npoints), 'o-', color='g', label=mode_name[1])
    # plt.plot(x, moving_avg(pst, sign_prob[:, 2], npoints), 'o-', color='r', label=mode_name[2])
    plt.xticks()
    plt.xlabel(xlab)
    plt.ylabel('Cell fraction')
    plt.legend(loc='best', fontsize=legend_font)
    if title:
        plt.title(title)

    if new_ax:
        plt.tight_layout()
        if showfig:
            plt.show()
        if savefig:
            plt.savefig(figname, format=format, dpi=dpi)

    if return_curve:
        return result_dict



def pattern_prob_plot(adata, pattern, key='velocity_pseudotime', title=None, npoints=10, xlab='pseudotime', ax=None,
                   figsize=(4, 2), showfig=False, savefig=True, figname='pst.pdf', format='pdf', dpi=300, return_curve=False):
    pst = np.asarray(adata.obs[key])

    new_ax = False
    if not ax:
        plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        new_ax = True

    mode_name, sign_prob = compute_pattern_prob(adata)

    x = np.linspace(0, 1, npoints)
    result_dict = {'x': x}

    n_modes = sign_prob.shape[1]
    color = 'r'

    y = moving_avg(pst, sign_prob[:, pattern], npoints)
    plt.plot(x, y, 'o-', color=color)

    plt.xticks()
    plt.xlabel(xlab)
    plt.ylabel('Cell fraction')
    if title:
        plt.title('pattern ' + str(pattern))

    if new_ax:
        plt.tight_layout()
        if showfig:
            plt.show()
        if savefig:
            plt.savefig(figname, format=format, dpi=dpi)

    if return_curve:
        return result_dict

'''
plot the average expression of ligands, receptors and targets overlayed on RNA velocity
'''


def expr_map(adata, pathway, key='distances',
             figsize=(12, 4), showfig=False, savefig=True, figname='expr_map.png', format='png'):
    neighbors = extract_neighbors(adata, key=key)

    coords = np.asarray(adata.obsm['X_umap'])
    x, y = coords[:, 0], coords[:, 1]

    rec, lig = np.asarray(adata.obs[pathway + '_rec']), np.asarray(adata.obs[pathway + '_lig'])
    tar = np.asarray(adata.obs[pathway + '_tar'])

    plt.figure(figsize=figsize)

    ax1 = plt.subplot(131)
    ax1.axis('off')
    df = pd.DataFrame.from_dict({'x': x, 'y': y, 'prob': 10 ** (neighbor_avg(rec, neighbors))})
    kdeplot(ax=ax1, data=df, x='x', y='y', weights='prob', fill=True, cmap='Reds', levels=20, cbar=False)
    scv.pl.velocity_embedding_stream(adata, basis='umap', color=None, size=0, linewidth=0.5, ax=ax1, show=False,
                                     save=False)
    plt.title(pathway + ' Receptors')

    ax2 = plt.subplot(132)
    ax2.axis('off')
    df = pd.DataFrame.from_dict({'x': x, 'y': y, 'prob': 10 ** (neighbor_avg(lig, neighbors))})
    kdeplot(ax=ax2, data=df, x='x', y='y', weights='prob', fill=True, cmap='Blues', levels=20, cbar=False)
    scv.pl.velocity_embedding_stream(adata, basis='umap', color=None, size=0, linewidth=0.5, ax=ax2, show=False,
                                     save=False)
    plt.title(pathway + ' Ligands')

    ax3 = plt.subplot(133)
    ax3.axis('off')
    df = pd.DataFrame.from_dict({'x': x, 'y': y, 'prob': 10 ** (neighbor_avg(tar, neighbors))})
    kdeplot(ax=ax3, data=df, x='x', y='y', weights='prob', fill=True, cmap='Greens', levels=20, cbar=False)
    scv.pl.velocity_embedding_stream(adata, basis='umap', color=None, size=0, linewidth=0.5, ax=ax3, show=False,
                                     save=False)
    plt.title(pathway + ' Targets')

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=500)


'''
plot the expression of top ligands, receptors and targets overlayed on RNA velocity
'''


def plot_row(adata, x, y, neighbors, weight, genelist, index, top, nrow, cmap, fontsize=10):
    ind = np.flip(np.argsort(weight))
    for i in range(min(top, ind.size)):
        ax = plt.subplot2grid((nrow, top), (index, i), rowspan=1, colspan=1)
        ax.axis('off')
        z = adata[:, genelist[ind][i]].X.toarray()
        df = pd.DataFrame.from_dict({'x': x, 'y': y, 'prob': 10 ** (neighbor_avg(z, neighbors))})
        kdeplot(ax=ax, data=df, x='x', y='y', weights='prob', fill=True, cmap=cmap, levels=20, cbar=False)
        scv.pl.velocity_embedding_stream(adata, basis='umap', color=None, size=0, linewidth=0.5, ax=ax, show=False,
                                         save=False)
        ax.set_title(genelist[ind][i], fontsize=fontsize)


def top_players_map(adata, pathway, key='distances', top=3, fontsize=10,
                    panel_height=3, panel_length=3, showfig=False, savefig=True, figname='expr_map.png', format='png'):
    neighbors = extract_neighbors(adata, key=key)

    coords = np.asarray(adata.obsm['X_umap'])
    x, y = coords[:, 0], coords[:, 1]

    rec_list, lig_list = adata.uns['pathways'][pathway]['receptors'], adata.uns['pathways'][pathway]['ligands']

    weight_lig, weight_rec = np.zeros(lig_list.size), np.zeros(rec_list.size)
    for i in range(lig_list.size):
        weight_lig[i] = np.mean(adata[:, lig_list[i]].X.toarray())

    include_target = False
    if 'TF' in adata.uns.keys():
        if pathway in adata.uns['TF'].keys():
            include_target = True
            tar_list = np.asarray(adata.uns['TF'][pathway])
            weight_tar = np.zeros(tar_list.size)
            for i in range(tar_list.size):
                weight_tar[i] = np.mean(adata[:, tar_list[i]].X.toarray())

    fig = plt.figure(figsize=(panel_length * top, panel_height * 3)) if include_target \
        else plt.figure(figsize=(panel_length * top, panel_height * 2))
    nrow = 3 if include_target else 2

    plot_row(adata, x, y, neighbors, weight_lig, lig_list, 0, top, nrow, 'Blues', fontsize=fontsize)
    plot_row(adata, x, y, neighbors, weight_rec, rec_list, 1, top, nrow, 'Reds', fontsize=fontsize)
    if include_target:
        plot_row(adata, x, y, neighbors, weight_tar, tar_list, 2, top, nrow, 'Greens', fontsize=fontsize)

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=500)
