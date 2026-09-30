"""Alluvial (Sankey) plots of cell types, communication modes and patterns (plotly)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .._optional import LazyModule
from ..tools._utils import _mode_order
from ._utils import discrete_colors, plot_function

go = LazyModule("plotly.graph_objects", extra="alluvial")


def compute_weight(list1, list2, v1, v2):
    r"""Number of cells with value ``v1`` in ``list1`` and ``v2`` in ``list2``.

    Parameters
    ----------
    list1, list2 : list
        Labels of every cell.
    v1, v2 : object
        Labels to count.

    Returns
    -------
    int
        Number of cells.
    """
    w = 0
    for l1, l2 in zip(list1, list2):
        if l1==v1 and l2==v2:
            w = w + 1
    return w


def set_color(map, features):
    # convert colors from matplotlib colormap to RGB
    r"""Plotly ``rgb(...)`` colours for a list of categories.

    Parameters
    ----------
    map : matplotlib colormap
        Colormap.
    features : list
        Categories.

    Returns
    -------
    list of str
        One colour per category.
    """
    colors = discrete_colors(map, len(features))
    colorlist = []
    for i in range(len(features)):
        rgb_scaled = tuple([255 * c for c in colors[i]])
        colorlist.append('rgb' + str(rgb_scaled))
    return colorlist


def twostate_sankey(v1, v2, lab1, lab2, col1, col2, name1, name2, alpha=0.5,
                     pad=5, thickness=40, linecolor='black', linewidth=1, width=400, height=600, font_size=15,
                    savefig=False, figname='compare_paths.pdf', format='pdf', showfig=False, strip_text=False):
    r"""Alluvial (Sankey) diagram between two labellings of the same cells.

    Parameters
    ----------
    v1, v2 : list
        Left and right label of every cell.
    lab1, lab2 : list
        Left and right categories (node order).
    col1, col2 : list of str
        Node colours.
    name1, name2 : str
        Titles of the two sides.
    alpha, pad, thickness, linecolor, linewidth, width, height, font_size, strip_text
        See :func:`scrich.pl.alluvial_onepath`.
    savefig, figname, format, showfig
        Not used (saving and showing are handled by the public functions).

    Returns
    -------
    plotly.graph_objects.Figure
        The figure.
    """
    label=lab1+lab2

    source, target, value, edge_color = [], [], [], []
    for n1 in lab1:
        for n2 in lab2:
            source.append(label.index(n1))
            target.append(label.index(n2))
            value.append(compute_weight(np.asarray(v1), np.asarray(v2), n1, n2))
            edge_color.append('rgba' + col2[lab2.index(n2)][3:-1] + ', ' + str(alpha) + ')')

    if strip_text:
        label = None

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=pad,  # separation between nodes
            thickness=thickness,
            line=dict(color=linecolor, width=linewidth),
            label=label,
            color=col1+col2
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=edge_color
        ))])
    if not strip_text:
        fig.add_annotation(text=name1, xref="paper", yref="paper", x=0., y=1.1, showarrow=False, align='center')
        fig.add_annotation(text=name2, xref="paper", yref="paper", x=1.05, y=1.1, showarrow=False, align='center')

    fig.update_layout(autosize=False, width=width, height=height, font_size=font_size)

    # saving and showing are handled by the plot_function decorator of the public functions
    return fig


@plot_function()
def alluvial_onepath(adata, pathway, key, map1=plt.cm.Set3, map2=plt.cm.Set2, alpha=0.5,
                     pad=5, thickness=40, linecolor='black', linewidth=1, width=400, height=600, font_size=15,
                     savefig=False, figname='alluvial.pdf', format='pdf', showfig=False, strip_text=False):
    r"""Alluvial plot between cell types and the communication modes of a pathway.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.all_path_sim`.
    pathway : str
        Pathway of interest.
    key : str
        Key in ``adata.obs`` with cell-type annotations.

    map1 : matplotlib colormap, default Set3
        Colours of the left nodes.
    map2 : matplotlib colormap
        Colours of the right nodes (default ``Set2``; ``tab20`` for patterns).
    alpha : float, default 0.5
        Transparency of the links.
    pad : float, default 5
        Padding between nodes (pixels).
    thickness : float, default 40
        Node thickness (pixels).
    linecolor : str, default 'black'
        Node outline colour.
    linewidth : float, default 1
        Node outline width.
    width : int, default 400
        Figure width (pixels).
    height : int, default 600
        Figure height (pixels).
    font_size : int, default 15
        Font size.
    strip_text : bool, default False
        Hide all text.

    Returns
    -------
    plotly.graph_objects.Figure or None
        The figure if ``show=False``, otherwise ``None``. Save it with ``fig.write_html('plot.html')``, or
        ``fig.write_image('plot.pdf')`` (requires ``kaleido``).

    Notes
    -----
    Requires ``plotly``: ``pip install 'scrich[alluvial]'``.
    """
    memb = list(adata.obs[key])
    p = list(adata.obs[pathway+'_modes'])

    states, modes = sorted(set(memb)), _mode_order(adata.obs[pathway + '_modes'])

    # set colors for source and targets
    source_color = set_color(map1, states)
    target_color = set_color(map2, modes)

    return twostate_sankey(memb, p, states, modes, source_color, target_color, 'Cell Type', pathway+' Mode', alpha=alpha,
                    pad=pad, thickness=thickness, linecolor=linecolor, linewidth=linewidth, width=width,
                    height=height, font_size=font_size,
                    savefig=savefig, figname=figname, format=format, showfig=showfig, strip_text=strip_text)


@plot_function()
def alluvial_twopath(adata, pathway1, pathway2, include='all', key='clusters', map1=plt.cm.Set3, map2=plt.cm.Set2, alpha=0.5,
                     pad=5, thickness=40, linecolor='black', linewidth=1, width=400, height=600, font_size=15,
                     savefig=False, figname='compare_paths.pdf', format='pdf', showfig=False, strip_text=False):
    r"""Alluvial plot between the communication modes of two pathways.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.all_path_sim`.
    pathway1, pathway2 : str
        Pathways to compare.
    include : list or 'all', default 'all'
        Cell types (values of ``adata.obs[key]``) to include.
    key : str, default 'clusters'
        Key in ``adata.obs`` with cell-type annotations, used with ``include``.

    map1 : matplotlib colormap, default Set3
        Colours of the left nodes.
    map2 : matplotlib colormap
        Colours of the right nodes (default ``Set2``; ``tab20`` for patterns).
    alpha : float, default 0.5
        Transparency of the links.
    pad : float, default 5
        Padding between nodes (pixels).
    thickness : float, default 40
        Node thickness (pixels).
    linecolor : str, default 'black'
        Node outline colour.
    linewidth : float, default 1
        Node outline width.
    width : int, default 400
        Figure width (pixels).
    height : int, default 600
        Figure height (pixels).
    font_size : int, default 15
        Font size.
    strip_text : bool, default False
        Hide all text.

    Returns
    -------
    plotly.graph_objects.Figure or None
        The figure if ``show=False``, otherwise ``None``. Save it with ``fig.write_html('plot.html')``, or
        ``fig.write_image('plot.pdf')`` (requires ``kaleido``).

    Notes
    -----
    Requires ``plotly``: ``pip install 'scrich[alluvial]'``.
    """
    if include!='all':
        adata = adata[adata.obs[key].isin(include)].copy()
    p1 = list(adata.obs[pathway1 + '_modes'])
    p2 = list(adata.obs[pathway2 + '_modes'])
    modes1, modes2 = _mode_order(adata.obs[pathway1 + '_modes']), _mode_order(adata.obs[pathway2 + '_modes'])

    # if signaling modes were not renamed, add the pathway name to distinguish
    if(modes1[0]==modes2[0]):
        p1 = [str(p) + '-' + pathway1 for p in p1]
        p2 = [str(p) + '-' + pathway2 for p in p2]
        modes1 = [str(m) + '-' + pathway1 for m in modes1]
        modes2 = [str(m) + '-' + pathway2 for m in modes2]

    # set colors for source and targets
    source_color = set_color(map1, modes1)
    target_color = set_color(map2, modes2)

    return twostate_sankey(p1, p2, modes1, modes2, source_color, target_color, pathway1, pathway2, alpha=alpha,
                    pad=pad, thickness=thickness, linecolor=linecolor, linewidth=linewidth, width=width,
                    height=height, font_size=font_size,
                    savefig=savefig, figname=figname, format=format, showfig=showfig, strip_text=strip_text)


@plot_function()
def alluvial_pattern(adata, key, map1=plt.cm.Set3, map2=plt.cm.tab20, alpha=0.5,
                     pad=5, thickness=40, linecolor='black', linewidth=1, width=400, height=600, font_size=15,
                     savefig=False, figname='patterns.pdf', format='pdf', showfig=False, strip_text=False):
    r"""Alluvial plot between cell types and signaling patterns.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.find_sign_patterns`.
    key : str
        Key in ``adata.obs`` with cell-type annotations.

    map1 : matplotlib colormap, default Set3
        Colours of the left nodes.
    map2 : matplotlib colormap
        Colours of the right nodes (default ``Set2``; ``tab20`` for patterns).
    alpha : float, default 0.5
        Transparency of the links.
    pad : float, default 5
        Padding between nodes (pixels).
    thickness : float, default 40
        Node thickness (pixels).
    linecolor : str, default 'black'
        Node outline colour.
    linewidth : float, default 1
        Node outline width.
    width : int, default 400
        Figure width (pixels).
    height : int, default 600
        Figure height (pixels).
    font_size : int, default 15
        Font size.
    strip_text : bool, default False
        Hide all text.

    Returns
    -------
    plotly.graph_objects.Figure or None
        The figure if ``show=False``, otherwise ``None``. Save it with ``fig.write_html('plot.html')``, or
        ``fig.write_image('plot.pdf')`` (requires ``kaleido``).

    Notes
    -----
    Requires ``plotly``: ``pip install 'scrich[alluvial]'``.
    """
    clusters, patterns = list(adata.obs[key]), np.asarray(adata.obs['sign_pattern'], dtype='int')

    states, modes = sorted(list(set(clusters))), sorted(list(set(patterns)))

    source_color = set_color(map1, states)
    target_color = set_color(map2, modes)

    return twostate_sankey(clusters, patterns, states, modes, source_color, target_color, 'Cell Type', 'Pattern', alpha=alpha,
                    pad=pad, thickness=thickness, linecolor=linecolor, linewidth=linewidth, width=width,
                    height=height, font_size=font_size,
                    savefig=savefig, figname=figname, format=format, showfig=showfig, strip_text=strip_text)
