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
    r"""Eigenvalues used to choose the number of communication modes of a pathway.

    Shows the smallest eigenvalues of the normalized Laplacian of the similarity matrix and the gaps between
    consecutive eigenvalues; the optimal number of modes is at the largest gap.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.all_path_sim`.
    pathway : str
        Pathway of interest.
    max : int, default 10
        Number of eigenvalues to show.
    tick_fontize : int, default 10
        Font size of the tick labels.
    axis_fontsize : int, default 10
        Font size of the axis labels.
    legend_fontsize : int, default 10
        Font size of the legend.
    eigen_style : str, default 'bo'
        Matplotlib format string for the eigenvalues.
    gap_style : str, default 'r-'
        Matplotlib format string for the gaps.
    legend : bool, default True
        Show the legend.
    legend_loc : str, default 'best'
        Legend position.
    figsize : tuple, default (5, 4)
        Figure size.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created by default.

    Returns
    -------
    matplotlib.axes.Axes or None
        The axes if ``show=False``, otherwise ``None``.
    """
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
    r"""Cells on an embedding (e.g. UMAP), coloured by a categorical annotation such as communication modes.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with ``adata.obsm['X_' + basis]``.
    key : str
        Key in ``adata.obs`` to colour by, e.g. ``'IGF_modes'``.
    s : float, default 2
        Marker size.
    order : list, optional
        Categories to plot and their order; by default all, in the stored mode order.
    cmap : matplotlib colormap, optional
        Colours of the categories (default ``Set2``, extended automatically for many categories).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created by default.
    legend_pos : tuple, default (0.5, 1.2)
        Legend anchor (``bbox_to_anchor``).
    legend_loc : str, default 'upper center'
        Legend location.
    ncol : int, default 4
        Number of legend columns.
    legend_font : int, default 10
        Legend font size.
    plot_figure : bool, default True
        Apply the tight layout.
    figsize : tuple, default (4, 4)
        Figure size.
    basis : str, default 'umap'
        Embedding in ``adata.obsm['X_' + basis]``.

    Returns
    -------
    matplotlib.axes.Axes or None
        The axes if ``show=False``, otherwise ``None``.
    """
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
    r"""Receptor and ligand expression in each communication mode of a pathway (violin plots).

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.all_path_sim`.
    key : str
        Modes of the pathway, ``pathway + '_modes'`` (e.g. ``'IGF_modes'``).
    order : list, optional
        Modes to show and their order; by default all, in the stored mode order.
    cmap : matplotlib colormap, optional
        Colours of the modes (default ``Set2``).
    legend_font : int, default 10
        Font size.
    plot_figure : bool, default True
        Apply the tight layout.
    figsize : tuple, default (2, 4)
        Figure size.

    Returns
    -------
    list of matplotlib.axes.Axes or None
        The receptor and ligand axes if ``show=False``, otherwise ``None``.
    """
    if order is None:
        order = _mode_order(adata.obs[key])


    # Set2 by default for signaling modes; extended automatically if there are more than 8 categories
    colors = discrete_colors(plt.cm.Set2 if cmap is None else cmap, len(order))

    plt.figure(figsize=figsize)

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
    r"""Average ligand, receptor (and downstream-gene) expression of a pathway on an embedding.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.all_path_sim`.
    pathway : str
        Pathway of interest.
    include : list or 'all', default 'all'
        Cell types (values of ``adata.obs[key]``) to show.
    key : str, default 'clusters'
        Key in ``adata.obs`` with cell-type annotations, used with ``include``.
    s : float, default 10
        Marker size.
    cmap : str, default 'Blues'
        Colormap.
    basis : str, default 'umap'
        Embedding in ``adata.obsm['X_' + basis]``.
    plot_figure : bool, default True
        Apply the tight layout.

    Returns
    -------
    list of matplotlib.axes.Axes or None
        The axes if ``show=False``, otherwise ``None``.
    """
    if include!='all':
        adata = adata[adata.obs[key].isin(include)].copy()

    coords = adata.obsm['X_' + basis]
    lig, rec = np.asarray(adata.obs[pathway+'_lig']), np.asarray(adata.obs[pathway+'_rec'])

    if pathway+'_tar' in adata.obs.keys():
        tar = np.asarray(adata.obs[pathway+'_tar'])
        plt.figure(figsize=(9,3))
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
        plt.figure(figsize=(6,3))
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
    r"""Total ligand and receptor expression of a pathway on an embedding.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.pathways_overview` or :func:`scrich.tl.find_genes`.
    pathway : str
        Pathway of interest.
    key : {'average'}, default 'average'
        Summary of the genes; only the sum of ligand / receptor expression is currently supported.
    cells : str, default 'all'
        ``'all'`` or a value of ``adata.obs[groupby]`` to show only those cells.
    groupby : str, default 'clusters'
        Key in ``adata.obs`` used to select cells when ``cells != 'all'``.
    basis : str, default 'umap'
        Embedding in ``adata.obsm['X_' + basis]``.
    figsize : tuple, default (6, 3)
        Figure size.

    Returns
    -------
    list of matplotlib.axes.Axes or None
        The ligand and receptor axes if ``show=False``, otherwise ``None``.
    """
    if key != 'average':
        raise ValueError("feature_plot currently supports only key='average'.")
    adata_sel = adata if cells == 'all' else adata[adata.obs[groupby] == cells].copy()

    lig, rec = adata.uns['pathways'][pathway]['ligands'], adata.uns['pathways'][pathway]['receptors']
    X_lig, X_rec = adata_sel[:, lig].X, adata_sel[:, rec].X
    lig_arr = np.asarray(X_lig.sum(axis=1)).ravel()
    rec_arr = np.asarray(X_rec.sum(axis=1)).ravel()

    plt.figure(figsize=figsize)

    plt.subplot(121)
    plt.axis('off')
    plt.scatter(adata_sel.obsm['X_' + basis][:,0], adata_sel.obsm['X_' + basis][:,1], c=lig_arr, cmap='Reds', s=10)
    plt.title(pathway+' ligands')

    plt.subplot(122)
    plt.axis('off')
    plt.scatter(adata_sel.obsm['X_' + basis][:, 0], adata_sel.obsm['X_' + basis][:, 1], c=rec_arr, cmap='Reds', s=10)
    plt.title(pathway+' receptors')

    plt.tight_layout()
    if savefig:
        plt.savefig(figname, format=format)
    if showfig:
        plt.show()
