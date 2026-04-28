'''
functions for plotting of similarity and dendrograms
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy

from .analysis_func import pairwise_MI

def redundancy(adata, include='all', key='clusters', figsize=(8, 6), fontsize=10, showfig=False, savefig='True', figname='path_redundancy.pdf', format='pdf'):
    if include!='all':
        adata = adata[adata.obs[key].isin(include)].copy()

    sim_mat = adata.uns['CCC_similarity_map']

    mask = np.tri(sim_mat.shape[0], k=0)
    plot_mat = np.ma.array(sim_mat, mask=mask)

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)

    cmap = cm.get_cmap("Reds").copy()
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
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format)


def plot_dendrogram(adata, ax, model, cmap=plt.cm.tab20, color_threshold=None, fontsize=10, above_threshold_color='tomato', **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
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



def pathway_hierarchy(adata, affinity='precomputed', linkage='complete', distance_threshold=0, n_clusters=None,
                      title='Pathway Hierarchical Clustering', cmap=plt.cm.tab20, color_threshold=None, above_threshold_color='tomato',
                      xlim=False, show_ticks=False, fontsize=10,
                      figsize=(4,6), orientation='vertical', savefig='True', showfig=False, figname='dendrogram.pdf', format='pdf', **kwargs):
    # sim_mat = pairwise_MI(adata)
    sim_mat = adata.uns['CCC_similarity_map']
    dist = 1 - sim_mat
    model = AgglomerativeClustering(affinity=affinity, linkage=linkage, distance_threshold=distance_threshold,
                                    n_clusters=n_clusters).fit(dist)

    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    plot_dendrogram(adata, ax, model, cmap=cmap, color_threshold=color_threshold, above_threshold_color=above_threshold_color, fontsize=fontsize, **kwargs)
    if not show_ticks:
        ax.set_xticks([])
    if xlim:
        plt.xlim(xlim)
    if title:
        plt.title(title)

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format)



