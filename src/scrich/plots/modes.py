"""Plots of the communication modes of a pathway."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ..tools._utils import _mode_order
from ._utils import _new_axes, discrete_colors, plot_function


@plot_function()
def plot_mode_gap(adata,
                  pathway,
                  max=10,
                  tick_fontize=10,
                  axis_fontsize=10,
                  legend_fontsize=10,
                  eigen_style='bo',
                  gap_style='r-',
                  legend=True,
                  legend_loc='best',
                  figsize=(5,4),
                  ax=None,
                  showfig=False,
                  savefig=True,
                  figname='eigen_gap.pdf',
                  format='pdf'):
    '''Plot the sorted eigenvalues and gap between them to identify the optimal number of signaling modes
    By default, the figure is saved in the directory. filepath must be added in the 'figname' argument

    Parameters
    ----------
    adata: anndata pbject of gene counts
    pathway: pathway of interest
    max: max number of eigenvalues to display
    tick_fontize: fontsize of tick labels (default=10)
    axis_fontsize: fontsize of axis labels (default=10)
    legend_fontsize: fontsize of legend (default=10)
    eigen_style: pyplot style for eigenvalues (defalut='bo')
    gap_style: pyplot style for gap (default='r-')
    legend: if True, plot legend (default=True)
    legend_loc: legend position (default='best')
    figsize: size of figure (default=(5,4))
    showfig: if True, show figure (default=False)
    savefig: if True, save figure (default=True)
    figname: name and path of saved figure (default='eigen_gap.pdf')
    format: format of saved figure (default='pdf')

    Returns
    -------
    None

    '''

    # extract spectral information from adata
    w = adata.uns['SymNMF'][pathway]['eigenvalues']
    gap = adata.uns['SymNMF'][pathway]['gap']

    # plotting
    fig, ax = _new_axes(ax, figsize)
    plt.plot(np.arange(1, max+1, 1), np.sort(w)[0:max], eigen_style, label='Eigenvalues')
    plt.plot(np.arange(2, max+1, 1), gap[0:max], gap_style, label='Gap between\nconsecutive\neigenvalues')
    plt.xticks(np.arange(1, max+1, 1), fontsize=tick_fontize)
    if legend:
        plt.legend(loc=legend_loc, fontsize=legend_fontsize)
    plt.xlabel('Eigenvalue rank', fontsize=axis_fontsize)
    plt.ylabel('Eigenvalue', fontsize=axis_fontsize)
    plt.title('Mode gap - ' + pathway + ' pathway')

    plt.tight_layout()
    if savefig:
        plt.savefig(figname, format=format)
    if showfig:
        plt.show()


@plot_function(aliases={'legens_pos': 'legend_pos'})
def scatter2D(adata, key, s=2, order=None, cmap=None, ax=None, legend_pos=(0.5, 1.2), legend_loc='upper center', ncol=4, legend_font=10,
                 plot_figure=True, figsize=(4,4), basis='umap', showfig=False, savefig=True, figname='umap_scatter.pdf', format='pdf'):
    if order is None:
        order = _mode_order(adata.obs[key])

    # Set2 by default for signaling modes; extended automatically if there are more than 8 categories
    colors = discrete_colors(plt.cm.Set2 if cmap is None else cmap, len(order))

    fig, ax1 = _new_axes(ax, figsize)

    patches = []
    for l, c in zip(order, colors):
        coords = adata.obsm['X_' + basis][np.asarray(adata.obs[key]==l)]
        ax1.scatter( coords[:,0], coords[:,1], s=s, c=[c for i in range(coords[:,1].size)], alpha=0.5, label=l )
        patches.append( mpatches.Patch(color=c, label=l) )
    plt.axis('off')
    plt.legend(bbox_to_anchor=legend_pos, handles=patches, loc=legend_loc, ncol=ncol, fontsize=legend_font)

    plt.tight_layout()
    if plot_figure:
        plt.tight_layout()
        if savefig:
            plt.savefig(figname, format=format, dpi=300)
        if showfig:
            plt.show()


@plot_function()
def mode_violin(adata, key, order=None, cmap=None, legend_font=10,
                 plot_figure=True, figsize=(2,4), showfig=False, savefig=True, figname='mode_violin.pdf', format='pdf'):
    if order is None:
        order = _mode_order(adata.obs[key])


    # Set2 by default for signaling modes; extended automatically if there are more than 8 categories
    colors = discrete_colors(plt.cm.Set2 if cmap is None else cmap, len(order))

    fig = plt.figure(figsize=figsize)

    if key.endswith('modes'):
        path = key.replace('_modes', '')

        rec, lig = [], []
        for l in order:
            rec.append( list(adata.obs[path + '_rec'][ adata.obs[path + '_modes']==l ]) )
            lig.append( list(adata.obs[path + '_lig'][ adata.obs[path + '_modes']==l ]) )

        ax2 = plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1)
        ax3 = plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1)

        parts = ax2.violinplot(rec, showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        ax2.set_ylabel('receptors', fontsize=legend_font)
        plt.yticks(fontsize=legend_font)
        ax2.set_xticks([])

        parts = ax3.violinplot(lig, showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        ax3.set_ylabel('Ligands', fontsize=legend_font)
        ax3.set_xticks(np.arange(1, len(order) + 1, 1), order, rotation=45,
                       fontsize=legend_font)

    plt.tight_layout()

    if plot_figure:
        plt.tight_layout()
        if savefig:
            plt.savefig(figname, format=format, dpi=300)
        if showfig:
            plt.show()


@plot_function()
def pathway_umap(adata, pathway, include='all', key='clusters', s=10, cmap='Blues', basis='umap',
                 plot_figure=True, showfig=False, savefig=True, figname='pathway_scatter.pdf', format='pdf'):
    if include!='all':
        adata = adata[adata.obs[key].isin(include)].copy()

    coords = adata.obsm['X_' + basis]
    lig, rec = np.asarray(adata.obs[pathway+'_lig']), np.asarray(adata.obs[pathway+'_rec'])

    if pathway+'_tar' in adata.obs.keys():
        tar = np.asarray(adata.obs[pathway+'_tar'])
        fig = plt.figure(figsize=(9,3))
        ax1 = plt.subplot(131)
        plt.axis('off')
        ax2 = plt.subplot(132)
        plt.axis('off')
        ax3 = plt.subplot(133)
        plt.axis('off')
        ax3.scatter(coords[:,0], coords[:,1], c=tar, s=s, cmap=cmap)
        ax3.set_title(pathway + ' targets')
        plt.axis('off')
    else:
        fig = plt.figure(figsize=(6,3))
        ax1 = plt.subplot(121)
        plt.axis('off')
        ax2 = plt.subplot(122)
        plt.axis('off')

    ax1.scatter(coords[:, 0], coords[:, 1], c=lig, s=s, cmap=cmap)
    ax1.set_title(pathway + ' ligands')
    ax2.scatter(coords[:, 0], coords[:, 1], c=rec, s=s, cmap=cmap)
    ax2.set_title(pathway + ' receptors')

    if plot_figure:
        plt.tight_layout()
        if savefig:
            plt.savefig(figname, format=format, dpi=300)
        if showfig:
            plt.show()


@plot_function()
def feature_plot(adata, pathway=None, key='average', cells='all', groupby='clusters', basis='umap',
                 figsize=(6,3), showfig=False, savefig=True, figname='single_path_role.pdf', format='pdf'):
    '''Total ligand and receptor expression of a pathway on an embedding.

    Parameters
    ----------
    adata: scRNA-seq data object
    pathway: pathway of interest
    key: only 'average' (sum of ligand / receptor expression) is supported
    cells: 'all' or a value of adata.obs[groupby] to plot only those cells
    groupby: adata.obs key used to select cells when cells != 'all'
    basis: embedding in adata.obsm['X_' + basis] (default='umap')
    '''
    if key != 'average':
        raise ValueError("feature_plot currently supports only key='average'.")
    adata_sel = adata if cells == 'all' else adata[adata.obs[groupby] == cells].copy()

    lig, rec = adata.uns['pathways'][pathway]['ligands'], adata.uns['pathways'][pathway]['receptors']
    X_lig, X_rec = adata_sel[:, lig].X, adata_sel[:, rec].X
    lig_arr = np.asarray(X_lig.sum(axis=1)).ravel()
    rec_arr = np.asarray(X_rec.sum(axis=1)).ravel()

    figure = plt.figure(figsize=figsize)

    ax = plt.subplot(121)
    plt.axis('off')
    plt.scatter(adata_sel.obsm['X_' + basis][:,0], adata_sel.obsm['X_' + basis][:,1], c=lig_arr, cmap='Reds', s=10)
    plt.title(pathway+' ligands')

    ax = plt.subplot(122)
    plt.axis('off')
    plt.scatter(adata_sel.obsm['X_' + basis][:, 0], adata_sel.obsm['X_' + basis][:, 1], c=rec_arr, cmap='Reds', s=10)
    plt.title(pathway+' receptors')

    plt.tight_layout()
    if savefig:
        plt.savefig(figname, format=format)
    if showfig:
        plt.show()
