"""Plots combining communication modes/patterns with RNA velocity and pseudotime."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .._optional import LazyModule
from ..tools._utils import compute_pattern_prob, compute_sign_prob, extract_neighbors, neighbor_avg
from ._utils import _new_axes, discrete_colors, plot_function

scv = LazyModule("scvelo", extra="velocity")
_sns = LazyModule("seaborn", extra="velocity")


def kdeplot(*args, **kwargs):
    return _sns.kdeplot(*args, **kwargs)


@plot_function()
def plot_maps(adata, pathway, key='distances', basis='umap', panel_height = 4, panel_length = 4,
              linewidth=0.5, arrow_style='->', mode_to_plot='all', fontsize=7,
              showfig=False, savefig=True, figname='velocity_plot', format='png'):
    '''Density of each communication mode on the embedding, overlaid on the RNA velocity stream.

    Parameters
    ----------
    adata: scRNA-seq data object with RNA velocity (scvelo)
    pathway: pathway of interest
    key: neighbour graph used to compute the local fraction of each mode ('distances' or 'connectivities')
    basis: embedding in adata.obsm['X_' + basis] (default='umap')
    mode_to_plot: 'all' (one panel per mode) or the name of a single mode
    '''
    mode_name, sign_prob = compute_sign_prob(adata, pathway, key=key)

    coords = np.asarray(adata.obsm['X_'+basis])
    x, y = coords[:, 0], coords[:, 1]

    colors = ['Greens', 'Oranges', 'Blues', 'Reds', 'Purples', 'Greys', 'YlOrBr', 'PuRd', 'BuGn', 'RdPu']

    if mode_to_plot == 'all':
        selected = list(range(len(mode_name)))
    else:
        if mode_to_plot not in mode_name:
            raise ValueError(f"mode_to_plot must be 'all' or one of {list(mode_name)}, got {mode_to_plot!r}")
        selected = [list(mode_name).index(mode_to_plot)]

    fig = plt.figure(figsize=(len(selected) * panel_length, panel_height))
    for j, i in enumerate(selected):
        ax = plt.subplot2grid((1, len(selected)), (0, j), rowspan=1, colspan=1)
        ax.axis('off')
        df = pd.DataFrame.from_dict({'x': x, 'y': y, 'prob': sign_prob[:, i]})
        kdeplot(ax=ax, data=df, x='x', y='y', weights='prob', fill=True, cmap=colors[i % len(colors)], levels=20, cbar=False)
        scv.pl.velocity_embedding_stream(adata, basis=basis, size=0, linewidth=linewidth, arrow_style=arrow_style,
                                         arrow_size=0.5, ax=ax, show=False, save=False)
        ax.set_title(str(mode_name[i]), fontsize=fontsize)
    plt.tight_layout()


@plot_function()
def pattern_plot(adata, key='distances', basis='umap', pattern=0, colormap='Reds', ax=None,
                 showfig=False, savefig=True, figname='pattern_plot.png', format='png'):
    mode_name, sign_prob = compute_pattern_prob(adata, key=key)
    n_modes = sign_prob.shape[1]

    coords = np.asarray(adata.obsm['X_' + basis])
    x, y = coords[:, 0], coords[:, 1]

    # figure parameters
    fig, ax = _new_axes(ax, (4, 4))
    ax.axis('off')
    df = pd.DataFrame.from_dict({'x': x, 'y': y, 'prob': sign_prob[:, pattern]})
    kdeplot(ax=ax, data=df, x='x', y='y', weights='prob', fill=True, cmap=colormap, levels=20, cbar=False)
    scv.pl.velocity_embedding_stream(adata, basis=basis, size=0, linewidth=0.5, ax=ax, show=False, save=False)
    plt.title('Pattern ' + str(pattern))

    plt.tight_layout()
    if savefig:
        plt.savefig(figname, format=format, dpi=500)
    if showfig:
        plt.show()


def moving_avg(pst, y, npoints):
    bins = np.linspace(0, 1, npoints + 1)
    y_avg = np.zeros(bins.size - 1)
    for i in range(bins.size - 1):
        pst_sel, y_sel = pst[pst >= bins[i]], y[pst >= bins[i]]
        pst_sel, y_sel = pst_sel[pst_sel <= bins[i + 1]], y_sel[pst_sel <= bins[i + 1]]
        y_avg[i] = np.mean(y_sel)
    return y_avg


@plot_function()
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

    new_ax = ax is None
    fig, ax = _new_axes(ax, figsize)

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
        if savefig:
            plt.savefig(figname, format=format, dpi=dpi)
        if showfig:
            plt.show()


@plot_function()
def sign_prob_plot(adata, pathway, key='velocity_pseudotime', legend_font=10, title=None, npoints=10, xlab='pseudotime', ax=None,
                   figsize=(4, 2), showfig=False, savefig=True, figname='pst.pdf', format='pdf', dpi=300, return_curve=False):
    pst = np.asarray(adata.obs[key])

    new_ax = ax is None
    fig, ax = _new_axes(ax, figsize)

    mode_name, sign_prob = compute_sign_prob(adata, pathway)

    x = np.linspace(0, 1, npoints)
    result_dict = {'x': x}

    n_modes = sign_prob.shape[1]
    colors = discrete_colors(['b', 'g', 'r'], n_modes)

    for i in range(n_modes):
        y = moving_avg(pst, sign_prob[:, i], npoints)
        plt.plot(x, y, 'o-', color=colors[i], label=mode_name[i])
        result_dict[mode_name[i]] = y

    plt.xticks()
    plt.xlabel(xlab)
    plt.ylabel('Cell fraction')
    plt.legend(loc='best', fontsize=legend_font)
    if title:
        plt.title(title)

    if new_ax:
        plt.tight_layout()
        if savefig:
            plt.savefig(figname, format=format, dpi=dpi)
        if showfig:
            plt.show()

    if return_curve:
        return result_dict


@plot_function()
def pattern_prob_plot(adata, pattern, key='velocity_pseudotime', title=None, npoints=10, xlab='pseudotime', ax=None,
                   figsize=(4, 2), showfig=False, savefig=True, figname='pst.pdf', format='pdf', dpi=300, return_curve=False):
    pst = np.asarray(adata.obs[key])

    new_ax = ax is None
    fig, ax = _new_axes(ax, figsize)

    mode_name, sign_prob = compute_pattern_prob(adata)

    x = np.linspace(0, 1, npoints)
    result_dict = {'x': x}

    n_modes = sign_prob.shape[1]
    color = 'r'

    y = moving_avg(pst, sign_prob[:, pattern], npoints)
    plt.plot(x, y, 'o-', color=color)
    result_dict['pattern ' + str(pattern)] = y

    plt.xticks()
    plt.xlabel(xlab)
    plt.ylabel('Cell fraction')
    if title:
        plt.title(title if isinstance(title, str) else 'pattern ' + str(pattern))

    if new_ax:
        plt.tight_layout()
        if savefig:
            plt.savefig(figname, format=format, dpi=dpi)
        if showfig:
            plt.show()

    if return_curve:
        return result_dict


@plot_function()
def expr_map(adata, pathway, key='distances', basis='umap',
             figsize=(12, 4), showfig=False, savefig=True, figname='expr_map.png', format='png'):
    neighbors = extract_neighbors(adata, key=key)

    coords = np.asarray(adata.obsm['X_' + basis])
    x, y = coords[:, 0], coords[:, 1]

    rec, lig = np.asarray(adata.obs[pathway + '_rec']), np.asarray(adata.obs[pathway + '_lig'])
    has_tar = pathway + '_tar' in adata.obs
    ncol = 3 if has_tar else 2

    plt.figure(figsize=figsize if has_tar else (figsize[0] * 2 / 3, figsize[1]))

    ax1 = plt.subplot(1, ncol, 1)
    ax1.axis('off')
    df = pd.DataFrame.from_dict({'x': x, 'y': y, 'prob': 10 ** (neighbor_avg(rec, neighbors))})
    kdeplot(ax=ax1, data=df, x='x', y='y', weights='prob', fill=True, cmap='Reds', levels=20, cbar=False)
    scv.pl.velocity_embedding_stream(adata, basis=basis, color=None, size=0, linewidth=0.5, ax=ax1, show=False,
                                     save=False)
    plt.title(pathway + ' Receptors')

    ax2 = plt.subplot(1, ncol, 2)
    ax2.axis('off')
    df = pd.DataFrame.from_dict({'x': x, 'y': y, 'prob': 10 ** (neighbor_avg(lig, neighbors))})
    kdeplot(ax=ax2, data=df, x='x', y='y', weights='prob', fill=True, cmap='Blues', levels=20, cbar=False)
    scv.pl.velocity_embedding_stream(adata, basis=basis, color=None, size=0, linewidth=0.5, ax=ax2, show=False,
                                     save=False)
    plt.title(pathway + ' Ligands')

    if has_tar:
        tar = np.asarray(adata.obs[pathway + '_tar'])
        ax3 = plt.subplot(1, ncol, 3)
        ax3.axis('off')
        df = pd.DataFrame.from_dict({'x': x, 'y': y, 'prob': 10 ** (neighbor_avg(tar, neighbors))})
        kdeplot(ax=ax3, data=df, x='x', y='y', weights='prob', fill=True, cmap='Greens', levels=20, cbar=False)
        scv.pl.velocity_embedding_stream(adata, basis=basis, color=None, size=0, linewidth=0.5, ax=ax3, show=False,
                                         save=False)
        plt.title(pathway + ' Targets')

    plt.tight_layout()
    if savefig:
        plt.savefig(figname, format=format, dpi=500)
    if showfig:
        plt.show()


def plot_row(adata, x, y, neighbors, weight, genelist, index, top, nrow, cmap, fontsize=10, basis='umap'):
    ind = np.flip(np.argsort(weight))
    for i in range(min(top, ind.size)):
        ax = plt.subplot2grid((nrow, top), (index, i), rowspan=1, colspan=1)
        ax.axis('off')
        z = adata[:, genelist[ind][i]].X
        z = np.asarray(z.toarray() if hasattr(z, 'toarray') else z).ravel()
        df = pd.DataFrame.from_dict({'x': x, 'y': y, 'prob': 10 ** (neighbor_avg(z, neighbors))})
        kdeplot(ax=ax, data=df, x='x', y='y', weights='prob', fill=True, cmap=cmap, levels=20, cbar=False)
        scv.pl.velocity_embedding_stream(adata, basis=basis, color=None, size=0, linewidth=0.5, ax=ax, show=False,
                                         save=False)
        ax.set_title(genelist[ind][i], fontsize=fontsize)


@plot_function()
def top_players_map(adata, pathway, key='distances', top=3, fontsize=10, basis='umap',
                    panel_height=3, panel_length=3, showfig=False, savefig=True, figname='expr_map.png', format='png'):
    neighbors = extract_neighbors(adata, key=key)

    coords = np.asarray(adata.obsm['X_' + basis])
    x, y = coords[:, 0], coords[:, 1]

    rec_list = np.asarray(adata.uns['pathways'][pathway]['receptors'])
    lig_list = np.asarray(adata.uns['pathways'][pathway]['ligands'])

    def _mean_expr(gene):
        return float(np.asarray(adata[:, gene].X.mean()))

    weight_lig = np.array([_mean_expr(g) for g in lig_list])
    weight_rec = np.array([_mean_expr(g) for g in rec_list])

    include_target = False
    if 'TF' in adata.uns.keys():
        if pathway in adata.uns['TF'].keys():
            include_target = True
            tar_list = np.asarray(adata.uns['TF'][pathway])
            weight_tar = np.array([_mean_expr(g) for g in tar_list])

    fig = plt.figure(figsize=(panel_length * top, panel_height * 3)) if include_target \
        else plt.figure(figsize=(panel_length * top, panel_height * 2))
    nrow = 3 if include_target else 2

    plot_row(adata, x, y, neighbors, weight_lig, lig_list, 0, top, nrow, 'Blues', fontsize=fontsize, basis=basis)
    plot_row(adata, x, y, neighbors, weight_rec, rec_list, 1, top, nrow, 'Reds', fontsize=fontsize, basis=basis)
    if include_target:
        plot_row(adata, x, y, neighbors, weight_tar, tar_list, 2, top, nrow, 'Greens', fontsize=fontsize, basis=basis)

    plt.tight_layout()
    if savefig:
        plt.savefig(figname, format=format, dpi=500)
    if showfig:
        plt.show()
