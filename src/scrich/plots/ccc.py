"""Plots of the cell-cell communication (CCC) matrix."""

from __future__ import annotations

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection

from .._optional import LazyModule
from .._settings import settings
from ._utils import _new_axes, discrete_colors, plot_function

_d3blocks = LazyModule("d3blocks", extra="chord")
scv = LazyModule("scvelo", extra="velocity")


def chord_diagram(adata, p, eps=0.01, show=None):
    r"""Interactive chord diagram of the CCC matrix of a pathway (requires ``d3blocks``).

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.compute_ccc_matrix`.
    p : str
        Pathway of interest.
    eps : float, default 0.01
        Interactions below this value are not shown.
    show : bool or None, default None
        Open the diagram; ``None`` follows :attr:`scrich.settings.autoshow`.

    Returns
    -------
    d3blocks.D3Blocks
        The d3blocks object (see the d3blocks documentation to save it as HTML).

    Notes
    -----
    Install the optional dependency with ``pip install 'scrich[chord]'``.
    """
    states, mat = adata.uns['ccc_mat'][p]['states'], adata.uns['ccc_mat'][p]['mat']

    # transform CCC mat to pandas dataframe
    ccc_dict = {'source': [], 'target': [], 'weight': []}
    for i in range(len(states)):
        for j in range(len(states)):
            if mat[i][j] > eps:
                ccc_dict['source'].append(states[i])
                ccc_dict['target'].append(states[j])
                ccc_dict['weight'].append(mat[i][j])
    ccc_df = pd.DataFrame.from_dict(ccc_dict)

    d3 = _d3blocks.D3Blocks()
    d3.chord(ccc_df)
    if settings.autoshow if show is None else show:
        d3.show()
    return d3


def enforce_spacing(x, y, dx, thr):
    r"""Placeholder for spacing overlapping labels (currently not used).

    Parameters
    ----------
    x, y : numpy.ndarray
        Label coordinates.
    dx : float
        Horizontal spacing.
    thr : float
        Minimum distance.

    Returns
    -------
    tuple
        New coordinates and number of changes.
    """
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
    r"""Draw a circular arrow for self-interactions (currently not used).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    center : tuple
        Centre of the loop.
    radius : float
        Loop radius.
    facecolor, edgecolor : str
        Colours of the arrow.
    theta1, theta2 : float
        Start and end angles (degrees).

    Returns
    -------
    None
    """
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


@plot_function()
def coarse_grained_map(adata, pathway, key='clusters', thr=0.01, fontsize=10, update_connection=None,
                       update_horiz_align=None, update_vert_align=None, linewidth=0.25, arrow_style='->',
                       single_source=None, single_target=None, figsize=(4, 4), strip_text=False,
                       basis='umap', ax=None, showfig=False, savefig=True, figname='coarse_grained.png', format='png'):
    r"""CCC matrix of a pathway drawn on the embedding, over the RNA velocity stream (requires ``scvelo``).

    Each cell state (cell type and communication mode, as in :func:`scrich.tl.compute_ccc_matrix`) is placed
    at the mean position of its cells; arrows from sender to receiver states have a width proportional to the
    interaction.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix after :func:`scrich.tl.compute_ccc_matrix`, with RNA velocity
        (:func:`scvelo.tl.velocity_graph`).
    pathway : str
        Pathway of interest.
    key : str, default 'clusters'
        Key in ``adata.obs`` with cell-type annotations (the one used for the CCC matrix).
    thr : float, default 0.01
        Interactions below this value are not drawn.
    fontsize : int, default 10
        Font size of the state labels.
    update_connection : dict, optional
        ``'start: <state>; end: <state>'`` -> matplotlib ``connectionstyle`` for individual arrows.
    update_horiz_align, update_vert_align : dict, optional
        State -> horizontal / vertical alignment of its label.
    linewidth : float, default 0.25
        Line width of the velocity stream.
    arrow_style : str, default '->'
        Arrow style of the velocity stream.
    single_source, single_target : str, optional
        Draw only the interactions from / to this state.
    figsize : tuple, default (4, 4)
        Figure size.
    strip_text : bool, default False
        Hide the state labels.
    basis : str, default 'umap'
        Embedding in ``adata.obsm['X_' + basis]``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created by default.

    Returns
    -------
    matplotlib.axes.Axes or None
        The axes if ``show=False``, otherwise ``None``.
    """
    mat = adata.uns['ccc_mat'][pathway]['mat']
    states = sorted(list(set(adata.obs[key])))
    modes = sorted(set(list(adata.obs[pathway + '_modes'])))

    x, y = np.zeros(len(modes) * len(states)), np.zeros(len(modes) * len(states))

    cmap = discrete_colors('tab20', len(states))

    x_umap, y_umap = adata.obsm['X_' + basis][:, 0], adata.obsm['X_' + basis][:, 1]
    x_scale = np.amax(x_umap) - np.amin(x_umap)
    text_separation = 0.02 * x_scale

    fig, ax = _new_axes(ax, figsize)

    ax.axis('off')

    # states and modes should be ordered in the same way as in the CCC matrix calculation
    node_state = []
    i = 0
    for s in states:
        for m in modes:
            node_state.append(s + '-' + str(m))
            adata_sel = adata[adata.obs[key] == s].copy()
            adata_sel = adata_sel[adata_sel.obs[pathway + '_modes'] == m]
            x[i], y[i] = np.mean(adata_sel.obsm['X_' + basis][:, 0]), np.mean(adata_sel.obsm['X_' + basis][:, 1])
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
                    plt.text(x_new[i] - text_separation, y_new[i], states[j] + '-' + str(modes[k]), fontsize=fontsize,
                             horizontalalignment=horizontal_align_dct[node_state[i]],
                             verticalalignment=vertical_align_dct[node_state[i]], fontweight="bold")
            i = i + 1

    scv.pl.velocity_embedding_stream(adata, basis=basis, size=0, arrow_size=0.5, linewidth=linewidth, ax=ax, show=False,
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
    if savefig:
        plt.savefig(figname, format=format, dpi=500)
    if showfig:
        plt.show()
