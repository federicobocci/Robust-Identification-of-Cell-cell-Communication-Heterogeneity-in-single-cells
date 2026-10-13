"""Plots of signaling patterns across pathways and of the similarity (mutual information) between pathways."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering

from .. import db
from ..tools._utils import _resolve_species
from ._utils import _new_axes, discrete_colors, plot_function


def find_pathway(gene, lig, rec, lig_set, rec_set):
    r"""Pathway of a ligand or receptor gene.

    Parameters
    ----------
    gene : str
        Gene name.
    lig, rec : pandas.DataFrame
        Bundled pathway -> ligand / receptor tables.
    lig_set, rec_set : list of str
        All ligands / receptors.

    Returns
    -------
    str or None
        The (first) pathway of the gene, or ``None`` if it is neither a ligand nor a receptor.
    """
    if gene in lig_set:
        i = list(lig['pathway'][ lig['ligand']==gene ])
        return i[0]
    elif gene in rec_set:
        i = list(rec['pathway'][ rec['receptor']==gene ])
        return i[0]


def unique(paths):
    r"""Unique elements of a list, in order of first appearance.

    Parameters
    ----------
    paths : list
        Elements.

    Returns
    -------
    list
        Unique elements.
    """
    unique = []
    for p in paths:
        if p not in unique:
            unique.append(p)
    return unique


def pattern_marker(ax, adata, pattern, ntop=25, species='human', cmap=plt.cm.tab20,
                   fontsize=10):

    # path_list = list(adata.uns['pathways'].keys())
    r"""Top marker genes of a pattern that are ligands or receptors of the selected pathways (bar plot).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.find_sign_patterns`.
    pattern : int or str
        Pattern.
    ntop : int, default 25
        Number of genes.
    species : {'human', 'mouse'}, default 'human'
        Species of the ligand-receptor database.
    cmap : matplotlib colormap, default tab20
        Colours of the pathways.
    fontsize : int, default 10
        Font size.

    Returns
    -------
    tuple
        Pathways of the genes (in order of rank) and a pathway -> colour dictionary.
    """
    path_list = adata.uns['selected_pathways']
    lig = db.load_pathway_genes('ligand', species=species)
    rec = db.load_pathway_genes('receptor', species=species)
    lig_set = list(lig['ligand'])
    rec_set = list(rec['receptor'])

    gene, score, path = [], [], []

    i = 0
    while len(gene)<ntop:
        all_genes = adata.uns['rank_genes_groups']['names'][i].tolist()
        all_scores = np.array(adata.uns['rank_genes_groups']['scores'][i].tolist())
        p = find_pathway(all_genes[pattern], lig, rec, lig_set, rec_set)
        i = i + 1
        if p in path_list:
            gene.append(all_genes[pattern])
            score.append(all_scores[pattern])
            path.append(p)

    x = np.arange(1, len(gene) + 1, 1)
    y = np.flip(np.asarray(score))
    lab = np.flip(np.asarray(gene))
    path = np.flip(np.asarray(path))

    color_dict = {}

    unique_paths = unique(list(path))  # order of appearance (deterministic, unlike set())
    colors = discrete_colors(cmap, len(unique_paths))
    for u, c in zip(unique_paths, colors):
        ax.barh(x[path==u], y[path==u], height=0.8, color=c, alpha=0.5, label=u)
        color_dict[u] = c

    plt.xticks(fontsize=fontsize)
    plt.yticks(x, lab, fontsize=fontsize)
    plt.xlabel('Score', fontsize=fontsize)
    # plt.legend(loc='lower right', fontsize=fontsize)
    plt.title('Pattern ' + str(pattern) + ' vs. Rest', fontsize=fontsize)

    unique_paths = unique(np.flip(path))
    return unique_paths, color_dict


def state_frac(states, clusters):
    r"""Fraction of cells of every state.

    Parameters
    ----------
    states : list
        States.
    clusters : numpy.ndarray
        State of every cell.

    Returns
    -------
    numpy.ndarray
        Fraction of cells in each state.
    """
    frac0 = np.zeros(len(states))
    for i in range(len(states)):
        frac0[i] = clusters[clusters == states[i]].size
    frac0 = frac0 / np.sum(frac0)
    return frac0


def state_repr(ax, adata, pattern_adata, key, order, fontsize=10, bar_color='b', bar_alpha=0.5):
    r"""Over- or under-representation of cell types in a pattern (relative change of their fraction).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    adata : anndata.AnnData
        Full data set.
    pattern_adata : anndata.AnnData
        Cells of the pattern.
    key : str
        Key in ``adata.obs`` with cell-type annotations.
    order : list or None
        Order of the cell types.
    fontsize : int, default 10
        Font size.
    bar_color : str, default 'b'
        Not used.
    bar_alpha : float, default 0.5
        Transparency of the bars.

    Returns
    -------
    None
    """
    clusters0 = np.asarray(adata.obs[key])
    clusters_patt = np.asarray(pattern_adata.obs[key])

    if order:
        states = order
    else:
        states = sorted(set(clusters0))

    frac0 = state_frac(states, clusters0)
    frac = state_frac(states, clusters_patt)

    rel_repr = (frac-frac0)/frac0
    x = np.arange(1, len(states) + 1, 1)

    x_pos, y_pos = x[rel_repr>0], rel_repr[rel_repr>0]
    x_neg, y_neg = x[rel_repr<0], rel_repr[rel_repr<0]

    lim = 1.1*np.amax(np.abs(rel_repr))

    ax.barh(x_pos, y_pos, height=0.8, color='r', alpha=bar_alpha)
    ax.barh(x_neg, y_neg, height=0.8, color='b', alpha=bar_alpha)
    plt.xlim( [ -lim, lim ] )
    plt.yticks(np.arange(1, len(states) + 1, 1), states, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    # ax.set_xlabel('State Over/Under\nRepresentation (fold-change)', fontsize=fontsize)
    ax.set_xlabel('Pattern composition', fontsize=fontsize)


@plot_function()
def pattern(adata, p, key='cluster', order=None, species=None, ntop=25, marker_cmap= plt.cm.tab20,                       # general function parameters
            fontsize=10,                                                               # plotting parameter for all plots
            lig_color='b', rec_color='r', showmean=False, meanwidth=1, meancolor='k',   # plotting parameter for violin plots
            bar_color='b', bar_alpha=0.5, figsize=(15, 6),                                                  # plotting parameter for composition plot
            savefig=True, figname='pattern_detail.pdf', showfig=False, format='pdf', dpi=300,      # figure parameters
            human=None                                                                 # deprecated, use species
            ):
    r"""Details of one signaling pattern.

    Four panels: over- or under-representation of each cell type in the pattern compared with the whole
    dataset; the top marker genes of the pattern that are ligands or receptors of the selected pathways
    (coloured by pathway); and the receptor and ligand expression of the pattern's top pathways in the
    pattern's cells (violins; horizontal lines show the dataset average with ``showmean=True``).

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.find_sign_patterns`.
    p : int or str
        Pattern (value of ``adata.obs['sign_pattern']``).
    key : str, default 'cluster'
        Key in ``adata.obs`` with cell-type annotations.
    order : list, optional
        Order of the cell types.
    species : {'human', 'mouse'}, default 'human'
        Species of the ligand-receptor database.
    ntop : int, default 25
        Number of marker genes to show.
    marker_cmap : matplotlib colormap, default tab20
        Colours of the pathways.
    fontsize : int, default 10
        Font size.
    lig_color, rec_color : str
        Not used; kept for backward compatibility.
    showmean : bool, default False
        Show the dataset averages on the violin plots.
    meanwidth : float, default 1
        Width of the average lines.
    meancolor : str, default 'k'
        Colour of the average lines.
    bar_color : str, default 'b'
        Not used; kept for backward compatibility.
    bar_alpha : float, default 0.5
        Transparency of the representation bars.
    figsize : tuple, default (15, 6)
        Figure size.
    human : bool, optional
        Deprecated, use ``species``.

    Returns
    -------
    list of matplotlib.axes.Axes or None
        The axes if ``show=False``, otherwise ``None``.
    """
    if 'sign_pattern' not in adata.obs:
        raise KeyError("Run scrich.tl.find_sign_patterns() before scrich.pl.pattern().")
    species = _resolve_species(species, human)

    pattern_adata = adata[adata.obs['sign_pattern'] == str(p)]

    plt.figure(figsize=figsize)

    ax1 = plt.subplot2grid((7, 15), (0, 0), rowspan=6, colspan=2)
    state_repr(ax1, adata, pattern_adata, key, order, fontsize=fontsize)

    ax2 = plt.subplot2grid((7, 15), (0, 4), rowspan=6, colspan=4)
    unique_paths, color_dict = pattern_marker(ax2, adata, p, ntop=ntop, species=species, cmap=marker_cmap, fontsize=fontsize)

    ax3 = plt.subplot2grid((7, 15), (0, 10), rowspan=3, colspan=5)

    rec_list, lig_list = [], []
    base_rec, base_lig = [], []
    colorlist = []
    for p in unique_paths[0:5]:
        rec_list.append( np.asarray(pattern_adata.obs[p+'_rec']) )
        lig_list.append( np.asarray(pattern_adata.obs[p+'_lig']) )
        base_rec.append( np.mean(np.asarray(adata.obs[p+'_rec'])) )
        base_lig.append( np.mean(np.asarray(adata.obs[p+'_lig'])) )
        colorlist.append( color_dict[p] )


    parts = ax3.violinplot(rec_list, showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colorlist[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)

    ax3.set_ylabel('Receptors', fontsize=fontsize)
    ax3.set_xticks([])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # ax3.set_yticks(fontsize=fontsize)
    # ax3.tick_params(axis='both', which='minor', labelsize=fontsize)

    ax4 = plt.subplot2grid((7, 15), (3, 10), rowspan=3, colspan=5)
    parts = ax4.violinplot(lig_list, showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colorlist[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)

    ax4.set_ylabel('Ligands', fontsize=fontsize)
    plt.xticks( np.arange(1, len(unique_paths[0:5])+1, 1), unique_paths[0:5], rotation=45, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    if showmean:
        for i in range(len(base_rec)):
            ax3.plot([i + 0.6, i + 1.4], [base_rec[i], base_rec[i]], lw=meanwidth, color=meancolor)
            ax4.plot([i + 0.6, i + 1.4], [base_lig[i], base_lig[i]], lw=meanwidth, color=meancolor)

    plt.tight_layout()
    if savefig:
        plt.savefig(figname, format=format)
    if showfig:
        plt.show()


@plot_function()
def pattern_summary(adata, key, order=None, panel_per_row=4, panel_length=3, panel_height=3, cmap=plt.cm.tab20,
                    fontsize=10,
                    savefig=True, figname='pattern_summary.pdf', showfig=False, format='pdf', dpi=300):
    r"""Distribution of the cells of every cell type across signaling patterns.

    One panel per cell type shows the fraction of its cells in each pattern.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.find_sign_patterns`.
    key : str
        Key in ``adata.obs`` with cell-type annotations.
    order : list, optional
        Cell types to show and their order; by default all, sorted.
    panel_per_row : int, default 4
        Number of panels per row.
    panel_length, panel_height : float, default 3
        Size of each panel.
    cmap : matplotlib colormap, default tab20
        Colours of the patterns (extended automatically for many patterns).
    fontsize : int, default 10
        Font size.

    Returns
    -------
    list of matplotlib.axes.Axes or None
        The axes if ``show=False``, otherwise ``None``.
    """
    if order:
        states = order
    else:
        states = sorted(set(adata.obs[key]))

    patterns = np.sort(np.array(list( set(list(adata.obs['sign_pattern'])) ), dtype='int'))
    bins = np.append(patterns-0.5, patterns[-1]+0.5)

    n = len(states)
    nrow = int(n / panel_per_row) + 1 if n % panel_per_row > 0 else int(n / panel_per_row)
    ncol = max(n % panel_per_row, panel_per_row)

    plt.figure(figsize=(panel_length*ncol, panel_height*nrow))
    colors = discrete_colors(cmap, patterns.size)

    for i in range(len(states)):
        sel_adata = adata[ adata.obs[key]==states[i] ]
        frac = np.asarray(sel_adata.obs['sign_pattern'], dtype='int')
        dist, bins = np.histogram(frac, bins=bins)
        dist = dist/np.sum(dist)

        # panel coordinates for plotting
        j = int(i / panel_per_row)
        k = i % panel_per_row

        plt.subplot2grid((nrow, ncol), (j, k), rowspan=1, colspan=1)

        for j in range(patterns.size):
            plt.bar( np.array([patterns[j]]), [dist[j]], color=colors[j] )

        # plt.bar(patterns, dist)
        plt.xticks(patterns, fontsize=fontsize)
        plt.title(states[i], fontsize=fontsize)
        plt.xlabel('Signaling Pattern', fontsize=fontsize)
        plt.ylabel('Cell Fraction', fontsize=fontsize)
    plt.tight_layout()
    if savefig:
        plt.savefig(figname, format=format)
    if showfig:
        plt.show()


@plot_function()
def redundancy(adata, include='all', key='clusters', figsize=(8, 6), fontsize=10, ax=None, showfig=False, savefig=False, figname='path_redundancy.pdf', format='pdf'):
    r"""Mutual information between the communication modes of every pair of selected pathways.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.pairwise_MI`.
    include : list or 'all', default 'all'
        Not used (the matrix is computed on all cells by :func:`scrich.tl.pairwise_MI`); kept for backward
        compatibility.
    key : str, default 'clusters'
        Not used; kept for backward compatibility.
    figsize : tuple, default (8, 6)
        Figure size.
    fontsize : int, default 10
        Font size.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created by default.

    Returns
    -------
    matplotlib.axes.Axes or None
        The axes if ``show=False``, otherwise ``None``.
    """
    if include!='all':
        adata = adata[adata.obs[key].isin(include)].copy()

    sim_mat = adata.uns['CCC_similarity_map']

    mask = np.tri(sim_mat.shape[0], k=0)
    plot_mat = np.ma.array(sim_mat, mask=mask)

    fig, ax = _new_axes(ax, figsize)

    cmap = matplotlib.colormaps["Reds"].copy()
    cmap.set_bad('w')

    pt = plt.pcolor(plot_mat.T, cmap=cmap, vmin=0, vmax=np.amax(plot_mat))

    for i in range(plot_mat.shape[0] + 1):
        if i == plot_mat.shape[0]:
            plt.plot([0, i - 1], [i, i], 'k-', lw=0.5)
        else:
            plt.plot([0, i], [i, i], 'k-', lw=0.5)
    for i in range(plot_mat.shape[0] + 1):
        if i == 0:
            plt.plot([i, i], [plot_mat.shape[0], i + 1], 'k-', lw=0.5)
        else:
            plt.plot([i, i], [plot_mat.shape[0], i], 'k-', lw=0.5)

    cbar = plt.colorbar(pt)
    cbar.set_label('Mutual Information', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    # labs = list(adata.uns['pathways'].keys())
    labs = adata.uns['selected_pathways']
    plt.xticks(np.arange(0, len(labs), 1) + 0.5, labs, rotation=90, fontsize=fontsize)
    plt.yticks(np.arange(0, len(labs), 1) + 0.5, labs, fontsize=fontsize)

    ax.spines.right.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.xaxis.tick_top()

    plt.tight_layout()
    if savefig:
        plt.savefig(figname, format=format)
    if showfig:
        plt.show()


def plot_dendrogram(adata, ax, model, cmap=plt.cm.tab20, color_threshold=None, fontsize=10, above_threshold_color='tomato', **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    r"""Draw the dendrogram of a fitted :class:`~sklearn.cluster.AgglomerativeClustering` model.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix (pathway labels from ``adata.uns['selected_pathways']``).
    ax : matplotlib.axes.Axes
        Axes to draw on.
    model : sklearn.cluster.AgglomerativeClustering
        Fitted model with ``distance_threshold`` set (so that distances are computed).
    cmap : matplotlib colormap, default tab20
        Colours of the clusters.
    color_threshold : float, optional
        Distance below which branches are coloured by cluster.
    fontsize : int, default 10
        Font size.
    above_threshold_color : str, default 'tomato'
        Colour of branches above ``color_threshold``.
    **kwargs
        Passed to :func:`scipy.cluster.hierarchy.dendrogram`.

    Returns
    -------
    None
    """
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # labels = list(adata.uns['pathways'].keys())
    labels = adata.uns['selected_pathways']


    # Plot the corresponding dendrogram
    colors = list(cmap.colors)
    hex_colors = [ to_hex(c) for c in colors ]
    hierarchy.set_link_color_palette(hex_colors)
    hierarchy.dendrogram(linkage_matrix, ax=ax, orientation='right', labels=labels,
                         color_threshold=color_threshold, above_threshold_color=above_threshold_color, **kwargs)


@plot_function()
def pathway_hierarchy(adata, affinity='precomputed', linkage='complete', distance_threshold=0, n_clusters=None,
                      title='Pathway Hierarchical Clustering', cmap=plt.cm.tab20, color_threshold=None, above_threshold_color='tomato',
                      xlim=False, show_ticks=False, fontsize=10,
                      figsize=(4,6), orientation='vertical', ax=None, savefig=False, showfig=False, figname='dendrogram.pdf', format='pdf', **kwargs):
    r"""Hierarchical clustering of the selected pathways by the similarity of their communication modes.

    Pathways are clustered with distance ``1 - mutual information`` (:func:`scrich.tl.pairwise_MI`).

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.pairwise_MI`.
    affinity : str, default 'precomputed'
        Metric of :class:`~sklearn.cluster.AgglomerativeClustering` (``metric`` in scikit-learn).
    linkage : str, default 'complete'
        Linkage criterion.
    distance_threshold : float, default 0
        Distance threshold of the clustering (0 builds the full tree).
    n_clusters : int, optional
        Number of clusters (use with ``distance_threshold=None``).
    title : str, default 'Pathway Hierarchical Clustering'
        Plot title.
    cmap : matplotlib colormap, default tab20
        Colours of the clusters.
    color_threshold : float, optional
        Distance below which branches are coloured by cluster.
    above_threshold_color : str, default 'tomato'
        Colour of branches above ``color_threshold``.
    xlim : tuple, optional
        Limits of the distance axis.
    show_ticks : bool, default False
        Show the distance ticks.
    fontsize : int, default 10
        Font size.
    figsize : tuple, default (4, 6)
        Figure size.
    orientation : str, default 'vertical'
        Not used (the dendrogram is drawn horizontally); kept for backward compatibility.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created by default.
    **kwargs
        Passed to :func:`scipy.cluster.hierarchy.dendrogram`.

    Returns
    -------
    matplotlib.axes.Axes or None
        The axes if ``show=False``, otherwise ``None``.
    """
    sim_mat = adata.uns['CCC_similarity_map']
    dist = 1 - sim_mat
    model = AgglomerativeClustering(metric=affinity, linkage=linkage, distance_threshold=distance_threshold,
                                    n_clusters=n_clusters).fit(dist)

    fig, ax = _new_axes(ax, figsize)
    plot_dendrogram(adata, ax, model, cmap=cmap, color_threshold=color_threshold, above_threshold_color=above_threshold_color, fontsize=fontsize, **kwargs)
    if not show_ticks:
        ax.set_xticks([])
    if xlim:
        plt.xlim(xlim)
    if title:
        plt.title(title)

    plt.tight_layout()
    if savefig:
        plt.savefig(figname, format=format)
    if showfig:
        plt.show()
