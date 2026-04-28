import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib.text as mtext
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go
from matplotlib.colors import to_hex
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy

from .analysis_func import pairwise_MI


def pathways_overview(adata,
                  figsize=(8,8),
                  ticksize=7,
                  fontsize=7,
                  top=None,
                  showfig=False,
                  savefig=True,
                  figname='pathways_overview.pdf',
                  format='pdf',
                  dpi=500):
    assert 'paths_overview' in adata.uns.keys(), "Run mc.pathways_overview before pf.pathways_overview"

    # number of detected genes
    number_dct = adata.uns['paths_overview']['gene_number']
    path, num = [], []
    for p in number_dct.keys():
        path.append(p)
        num.append(number_dct[p])
    rec, lig = np.asarray(num)[:,0], np.asarray(num)[:,1]

    plt.figure(figsize=figsize)

    ax = plt.subplot(211)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ind = np.flip(np.argsort(rec+lig))
    if top:
        ind = ind[0:top]
    plt.bar(np.arange(1, len(ind) + 1, 1), rec[ind], label='Receptors', color='tomato')
    plt.bar(np.arange(1, len(ind) + 1, 1), lig[ind], bottom=rec[ind], label='Ligands', color='mediumseagreen')
    plt.xticks(np.arange(1, len(ind) + 1, 1), [path[i] for i in ind], rotation=90, fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlim([0, len(ind) + 1])
    plt.ylabel('Detected genes', fontsize=fontsize)
    plt.legend(loc='upper right', fontsize=fontsize)



    # gene expression
    expr_dct = adata.uns['paths_overview']['gene_expression']
    path, expr = [], []
    for p in expr_dct.keys():
        path.append(p)
        expr.append( expr_dct[p] )
    expr = np.mean(np.asarray(expr), axis=1)

    ax = plt.subplot(212)
    ind = np.flip(np.argsort(expr))
    if top:
        ind = ind[0:top]
    plt.bar( np.arange(1, len(ind)+1, 1), expr[ind], color='skyblue' )
    plt.xticks( np.arange(1, len(ind)+1, 1), [path[i] for i in ind], rotation=90, fontsize=ticksize )
    plt.yticks(fontsize=ticksize)
    plt.xlim([0, len(ind) + 1])
    plt.ylabel('Average expression\n(Receptors + Ligands)', fontsize=fontsize)

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=dpi)

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
    plt.figure(figsize=figsize)
    plt.plot(np.arange(1, max+1, 1), np.sort(w)[0:max], eigen_style, label='Eigenvalues')
    plt.plot(np.arange(2, max+1, 1), gap[0:max], gap_style, label='Gap between\nconsecutive\neigenvalues')
    plt.xticks(np.arange(1, max+1, 1), fontsize=tick_fontize)
    if legend:
        plt.legend(loc=legend_loc, fontsize=legend_fontsize)
    plt.xlabel('Eigenvalue rank', fontsize=axis_fontsize)
    plt.ylabel('Eigenvalue', fontsize=axis_fontsize)
    plt.title('Mode gap - ' + pathway + ' pathway')

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format)


'''
plot the umap of signaling modes as well as the violinplot
'''
def scatter2D(adata, key, s=2, order=None, cmap=None, ax=None, legens_pos=(0.5, 1.2), legend_loc='upper center', ncol=4, legend_font=10,
                 plot_figure=True, figsize=(4,4), showfig=False, savefig=True, figname='umap_scatter.pdf', format='pdf'):
    if order==None:
        labs = list(adata.obs[key])
        order = sorted(list(set(labs)))

    # use  Set2 for signaling models (8 values)
    if cmap==None:
        colors = list(plt.cm.Set2.colors)
    else:
        colors = list(cmap.colors)

    fig = plt.figure(figsize=figsize)

    ### umap plot ###
    if not ax:
        ax1 = plt.subplot()
    else:
        ax1 = ax

    patches = []
    for l, c in zip(order, colors):
        coords = adata.obsm['X_umap'][adata.obs[key]==l]
        ax1.scatter( coords[:,0], coords[:,1], s=s, c=[c for i in range(coords[:,1].size)], alpha=0.5, label=l )
        patches.append( mpatches.Patch(color=c, label=l) )
    plt.axis('off')
    plt.legend(bbox_to_anchor=legens_pos, handles=patches, loc=legend_loc, ncol=ncol, fontsize=legend_font)

    plt.tight_layout()
    if plot_figure:
        plt.tight_layout()
        if showfig:
            plt.show()
        if savefig:
            plt.savefig(figname, format=format, dpi=300)


'''
plot the umap of signaling modes as well as the violinplot
'''
def mode_violin(adata, key, order=None, cmap=None, legend_font=10,
                 plot_figure=True, figsize=(2,4), showfig=False, savefig=True, figname='mode_violin.pdf', format='pdf'):
    if order==None:
        labs = list(adata.obs[key])
        order = sorted(list(set(labs)))


    # use Set3 (12 values) for clusters/time points and Set2 for signaling models (8 values)
    if cmap==None:
        colors = list(plt.cm.Set2.colors)
    else:
        colors = list(cmap.colors)

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
        if showfig:
            plt.show()
        if savefig:
            plt.savefig(figname, format=format, dpi=300)



# def modes_violin(adata, key, plot_figure=True, fontsize=10, showfig=False, savefig=True, figname='modes_violin.pdf',
#                  format='pdf', figsize=(1.5,2), strip_text=False):
#     path = key.replace('_modes', '')
#     labs = sorted(list(set(list(adata.obs[path + '_modes']))))
#     colors = list(plt.cm.Set2.colors)[0:len(labs)]
#
#     rec, lig = [], []
#     for l in labs:
#         rec.append(list(adata.obs[path + '_rec'][adata.obs[path + '_modes'] == l]))
#         lig.append(list(adata.obs[path + '_lig'][adata.obs[path + '_modes'] == l]))
#
#     fig = plt.figure(figsize=(1.5,2))
#
#     ax2 = plt.subplot(211)
#     parts = ax2.violinplot(rec, showmeans=False, showmedians=False, showextrema=False)
#     for i, pc in enumerate(parts['bodies']):
#         pc.set_facecolor(colors[i])
#         pc.set_edgecolor('black')
#         pc.set_alpha(1)
#     ax2.set_ylabel('Receptors', fontsize=fontsize)
#     ax2.set_xticks([])
#     plt.yticks(fontsize=fontsize)
#
#     ax3 = plt.subplot(212)
#     parts = ax3.violinplot(lig, showmeans=False, showmedians=False, showextrema=False)
#     for i, pc in enumerate(parts['bodies']):
#         pc.set_facecolor(colors[i])
#         pc.set_edgecolor('black')
#         pc.set_alpha(1)
#     ax3.set_ylabel('Ligands', fontsize=fontsize)
#     plt.yticks(fontsize=fontsize)
#     plt.xticks(np.arange(1, len(labs) + 1, 1), ['Mode ' + str(l) for l in labs], rotation=45, fontsize=fontsize)
#
#     if strip_text:
#         for text_obj in fig.findobj(mtext.Text):
#             text_obj.set_visible(False)
#
#     if plot_figure:
#         plt.tight_layout()
#         if showfig:
#             plt.show()
#         if savefig:
#             plt.savefig(figname, format=format, dpi=300)

def pathway_umap(adata, pathway, include='all', key='clusters', s=10, cmap='Blues',
                 plot_figure=True, showfig=False, savefig=True, figname='pathway_scatter.pdf', format='pdf'):
    if include!='all':
        adata = adata[adata.obs[key].isin(include)].copy()

    coords = adata.obsm['X_umap']
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
        if showfig:
            plt.show()
        if savefig:
            plt.savefig(figname, format=format, dpi=300)


def heatmap_one_pathway(adata, pathway, annotation, order=None, legend_top=False, legend_loc_top='best', legend_loc_right='best',
                        ncol_top=1, ncol_right=1, ylim_top=False, xlim_right=False, legend_font=10,
                        plot_figure=True, showfig=False, savefig=True, figname='types_table.pdf', format='pdf', width=6, height=6):

    assert pathway + '_modes' in adata.obs.keys(), 'no key found'

    # use anchor to put legend on right
    # order of clusters as input
    # use another colormap for the modes

    labs = np.asarray(list(adata.obs[pathway + '_modes']))
    lab_set = list(set(labs))
    clusters = np.asarray(list(adata.obs[annotation]))
    if order==None:
        clst_set = sorted(list(set(clusters)))
    else:
        clst_set = list(np.asarray(order))


    ntypes, nclst = len(lab_set), len(clst_set)
    frac = adata.uns['fraction_mat'][pathway]

    fig = plt.figure(figsize=(width, height))

    x = np.arange(0.5, ntypes, 1)
    y = np.arange(0.5, nclst, 1)

    # break down types based on annotations
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set3.colors)
    ax1 = plt.subplot2grid((6, 6), (0, 0), rowspan=2, colspan=4)

    mode_norm = np.sum(frac, axis=1)
    for i in range(nclst):
        bot = 0.
        if i > 0:
            for j in range(i):
                bot = bot + frac[:, j]
        plt.bar(x, frac[:, i]/mode_norm, bottom=bot/mode_norm, align='center', alpha=0.75, label=clst_set[i])
    plt.xticks([])
    if ylim_top:
        plt.ylim([0, ylim_top])
    plt.ylabel('Cell Type distribution')
    if legend_top:
        plt.legend(loc=legend_loc_top, ncol=ncol_top, fontsize=legend_font)

    ax2 = plt.subplot2grid((6, 6), (2, 0), rowspan=4, colspan=4)
    pt = plt.pcolor(np.transpose(frac) / np.sum(frac), cmap='Reds', vmin=0)
    plt.ylabel('Cell Type')
    plt.xlabel('Signaling Mode - ' + pathway)
    plt.xticks(x, [str(i) for i in range(ntypes)])
    plt.yticks(y, clst_set)

    # break down clusters based on types
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set2.colors)
    ax1 = plt.subplot2grid((6, 6), (2, 4), rowspan=4, colspan=2)

    clst_norm = np.sum(frac, axis=0)

    for i in range(ntypes):
        bot = 0.
        if i > 0:
            for j in range(i):
                bot = bot + frac[j]
        plt.barh(y, frac[i]/clst_norm, left=bot/clst_norm, align='center', alpha=0.75, label='Mode ' + str(i))
    plt.yticks([])
    plt.ylim([0, y[-1]+0.5])
    if xlim_right:
        plt.ylim([lim_right, 0])
    plt.xlabel('Mode distribution')
    plt.legend(loc=legend_loc_right, ncol=ncol_right, fontsize=legend_font, fancybox=True, framealpha=0.5)
    # plt.legend(bbox_to_anchor=(0, 1.05), loc='lower left', ncol=ncol_right, fontsize=legend_font)

    # plt.tight_layout()
    # if savefig:
    #     plt.savefig(figname, format=format, dpi=300)

    if plot_figure:
        plt.tight_layout()
        if showfig:
            plt.show()
        if savefig:
            plt.savefig(figname, format=format, dpi=300)

# heterogeneity score of a given pathway for all clusters
def single_pathway_heterogeneity(adata, pathway, key, order=None, by_score=True, show_perm=True, perm_threshold=0.05, fontsize=10, xlab_rot=90, horizontal=True,
                                 color='mediumseagreen', showfig=False, savefig=True, figname='single_pathway.pdf', format='pdf', figsize=(6,3), strip_text=False):

    # extract data and labels from anndata object
    labels = sorted(list(set(list(adata.obs[key]))))
    het = adata.uns['pathways'][pathway]['cci_heterogeneity']
    p_vals = np.asarray(adata.uns['permutation_test'][1][pathway])

    # set up cell type ordering
    if order:
        ind = [labels.index(o) for o in order]
    elif by_score:
        ind = np.flip(np.argsort(het))
    else:
        ind = np.arange(0, het.size, 1)

    fig = plt.figure(figsize=figsize)

    if horizontal:
        bars = plt.bar(np.arange(0, het.size, 1), het[ind], color=color)
        plt.xticks(np.arange(0, het.size, 1), [labels[i] for i in ind], rotation=xlab_rot, fontsize=fontsize)
        plt.ylabel('CCI heterogeneity', fontsize=fontsize)
        plt.ylim([0, 1.2 * np.amax(het[ind])])
    else:
        ind = np.flip(ind)
        bars = plt.barh(np.arange(0, het.size, 1), het[ind], color=color)
        plt.yticks(np.arange(0, het.size, 1), [labels[i] for i in ind], fontsize=fontsize)
        plt.xlabel('CCI heterogeneity', fontsize=fontsize)
        plt.xlim([0, 1.2 * np.amax(het[ind])])
    plt.title(pathway)

    for pv, bar in zip(p_vals[ind], bars):
        width, height = bar.get_width(), bar.get_height()
        if horizontal:
            x, y = bar.get_x() + bar.get_width() / 2, height
        else:
            x, y = width, bar.get_y() + bar.get_height() / 2
        if pv < perm_threshold:
            plt.text(x, y, '*', ha='left', va='center' )

    if strip_text:
        for text_obj in fig.findobj(mtext.Text):
            text_obj.set_visible(False)

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)

# heterogeneity of each pathway (all clusters considered or an individual cluster)
def pathway_heterogeneity_summary(adata, key='clusters', state='all', fontsize=10, color='lightskyblue', edgecolor='k', lw=1, errorbar=False, orientation='vertical',
                                  showfig=False, savefig=True, figname='pathways_summary.pdf', format='pdf', figsize=(3,5)):
    assert orientation == 'vertical' or orientation == 'horizontal', "Invalid Orientation value. Please choose between 'orientation=vertical' and 'orientation=horizontal'"

    if state!='all':
        labels = sorted(list(set(list(adata.obs[key]))))
        pos = labels.index(state)
        adata = adata[adata.obs[key].isin([state])].copy()

    # paths = list(adata.uns['pathways'].keys())
    paths = adata.uns['permutation_test'][2]
    labels = sorted(list(set(list(adata.obs[key]))))

    het_mat = np.zeros((len(paths), len(labels)))

    for i in range(len(paths)):
        v_het = adata.uns['pathways'][paths[i]]['cci_heterogeneity']
        if state=='all':
            for j in range(len(labels)):
                het_mat[i][j] = v_het[j]
        else:
            het_mat[i][0] = v_het[pos]
    avg, std = np.mean(het_mat, axis=1), np.std(het_mat, axis=1)

    ind = np.argsort(avg)

    fig = plt.figure(figsize=figsize)

    if orientation=='vertical':
        if errorbar:
            # plt.barh(np.arange(0, avg.size, 1), 1-avg[ind], xerr=std[ind], color=color, edgecolor=edgecolor, lw=lw)
            plt.barh(np.arange(0, avg.size, 1), avg[ind], xerr=std[ind], color=color, edgecolor=edgecolor, lw=lw)
        else:
            # plt.barh(np.arange(0, avg.size, 1), 1-avg[ind], color=color, edgecolor=edgecolor, lw=lw)
            plt.barh(np.arange(0, avg.size, 1), avg[ind], color=color, edgecolor=edgecolor, lw=lw)
        plt.yticks(np.arange(0, avg.size, 1), [paths[i] for i in ind], fontsize=fontsize)
        plt.xlabel('CCI heterogeneity', fontsize=fontsize)
    else:
        if errorbar:
            # plt.bar(np.arange(0, avg.size, 1), 1-avg[ind], xerr=std[ind], color=color, edgecolor=edgecolor, lw=lw)
            plt.bar(np.arange(0, avg.size, 1), avg[ind], xerr=std[ind], color=color, edgecolor=edgecolor, lw=lw)
        else:
            # plt.bar(np.arange(0, avg.size, 1), 1-avg[ind], color=color, edgecolor=edgecolor, lw=lw)
            plt.bar(np.arange(0, avg.size, 1), avg[ind], color=color, edgecolor=edgecolor, lw=lw)
        plt.xticks(np.arange(0, avg.size, 1), [paths[i] for i in ind], fontsize=fontsize, rotation=90)
        plt.ylabel('CCI heterogeneity', fontsize=fontsize)

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)

def state_heterogeneity_summary(adata, key, order=None, fontsize=10, color='lightskyblue', edgecolor='k', lw=1, errorbar=False, orientation='vertical',
                                  showfig=False, savefig=True, figname='states_summary.pdf', format='pdf', figsize=(3,5)):

    assert orientation=='vertical' or orientation=='horizontal', "Invalid Orientation value. Please choose between 'orientation=vertical' and 'orientation=horizontal'"

    # paths = list(adata.uns['pathways'].keys())
    paths = adata.uns['selected_pathways']
    labels = sorted(list(set(list(adata.obs[key]))))

    het_mat = np.zeros((len(paths), len(labels)))
    for i in range(len(paths)):
        v_het = adata.uns['pathways'][paths[i]]['cci_heterogeneity']
        for j in range(len(labels)):
            het_mat[i][j] = v_het[j]
    avg, std = np.mean(het_mat, axis=0), np.std(het_mat, axis=0)

    if order==None:
        ind = np.argsort(avg)
    else:
        ind = [labels.index(o) for o in order]

    fig = plt.figure(figsize=figsize)

    if orientation=='vertical':
        if errorbar:
            plt.barh(np.arange(0, avg.size, 1), avg[ind], xerr=std[ind], color=color, edgecolor=edgecolor, lw=lw)
        else:
            plt.barh(np.arange(0, avg.size, 1), avg[ind], color=color, edgecolor=edgecolor, lw=lw)
        plt.yticks(np.arange(0, avg.size, 1), [labels[i] for i in ind], fontsize=fontsize)
        plt.xlabel('CCI heterogeneity', fontsize=fontsize)
    else:
        if errorbar:
            plt.bar(np.arange(0, avg.size, 1), avg[ind], xerr=std[ind], color=color, edgecolor=edgecolor, lw=lw)
        else:
            plt.bar(np.arange(0, avg.size, 1), avg[ind], color=color, edgecolor=edgecolor, lw=lw)
        plt.xticks(np.arange(0, avg.size, 1), [labels[i] for i in ind], fontsize=fontsize)
        plt.ylabel('CCI heterogeneity', fontsize=fontsize)

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)

def heterogeneity_heatmap(adata, key, fontsize=10, cmap='Reds', showfig=False, savefig=True, figname='het_table.pdf', format='pdf', figsize=(6,5)):
    paths = list(adata.uns['pathways'].keys())
    labels = sorted(list(set(list(adata.obs[key]))))

    het_mat = np.zeros((len(paths), len(labels)))
    for i in range(len(paths)):
        v_het = adata.uns['pathways'][paths[i]]['cci_heterogeneity']
        for j in range(len(labels)):
            het_mat[i][j] = v_het[j]

    fig = plt.figure(figsize=figsize)

    x, y = np.arange(0, len(paths)+1, 1), np.arange(0, len(labels)+1, 1)
    pt = plt.pcolor(x, y, het_mat.transpose(), cmap=cmap, vmin=0, vmax=np.amax(het_mat))
    cbar = plt.colorbar(pt)
    cbar.set_label(label='Pathway heterogeneity',size=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    plt.xticks( x[0:-1]+0.5, paths, rotation=90, fontsize=fontsize)
    plt.yticks( y[0:-1]+0.5, labels, fontsize=fontsize)

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)


#########################
#
# featureplot
#
#########################
def feature_plot(adata, pathway=None, key='average', cells='all',
                 figsize=(6,3), showfig=False, savefig=True, figname='single_path_role.pdf', format='pdf'):
    if cells!='all':
        adata_sel = adata[adata.obs['final.clusters']==cells].copy()

    if key=='average':
        lig, rec = adata.uns['pathways'][pathway]['ligands'], adata.uns['pathways'][pathway]['receptors']
        lig_arr = np.sum(adata_sel[:, lig].X.toarray(), axis=1)
        rec_arr = np.sum(adata_sel[:, rec].X.toarray(), axis=1)

    figure = plt.figure(figsize=figsize)

    ax = plt.subplot(121)
    plt.axis('off')
    plt.scatter(adata_sel.obsm['X_umap'][:,0], adata_sel.obsm['X_umap'][:,1], c=lig_arr, cmap='Reds', s=10)
    plt.title(pathway+' ligands')

    ax = plt.subplot(122)
    plt.axis('off')
    plt.scatter(adata_sel.obsm['X_umap'][:, 0], adata_sel.obsm['X_umap'][:, 1], c=rec_arr, cmap='Reds', s=10)
    plt.title(pathway+' receptors')

    plt.tight_layout()
    if showfig:
        plt.showfig()
    if savefig:
        plt.savefig(figname, format=format)


def mode_composition(adata, key, pathway, mode, order=None, rename_states=None, fontsize=7, ticksize=7,
                     figsize=(3,3), showfig=False, savefig=True, figname='mode_composition.pdf', format='pdf'):
    # plot the composition of a signaling mode - normalize based on the total number of cells in each cluster
    frac = adata.uns['fraction_mat'][pathway]
    tot_cells = np.sum(frac, axis=0)
    fracs = frac[mode]

    clusters = np.asarray(list(adata.obs[key]))
    if order == None:
        clst_set = sorted(list(set(clusters)))
    else:
        clst_set = list(np.asarray(order))
    colors = list(plt.cm.Accent.colors)

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    ax.spines[['right', 'top']].set_visible(False)

    if rename_states:
        xlab = rename_states
    else:
        xlab = clst_set

    for i in range(fracs.size):
        plt.bar([i], [fracs[i]/tot_cells[i]], color=colors[i], width=0.9, label=clst_set[i])
    # plt.bar(np.arange(0, fracs.size, 1), fracs/tot_cells)
    plt.yticks(fontsize=ticksize)
    plt.xticks(np.arange(0, fracs.size, 1), xlab, rotation=90, fontsize=ticksize)
    plt.ylabel('Cell fraction', fontsize=fontsize)

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format)


########################
#
#  pseudotime functions
#
########################

def pst_overview(adata, key, pathway, order, figname='pseudotime_overview.pdf', format='pdf', alp=0.5, c='b'):

    pst = adata.obs['pst']
    clusters = np.asarray(adata.obs[key])
    labels = np.asarray(adata.obs[pathway+'_modes'])
    type_lab = list(set(labels))


    data = []
    for i in range(len(order)):
        data.append(pst[clusters == order[i]])

    plt.figure(figsize=(10, 4))
    ax = plt.subplot(121)
    vp = ax.violinplot(data, vert=False, showextrema=False)
    for pc in vp['bodies']:
        pc.set_facecolor(c)
        pc.set_edgecolor(c)
        pc.set_alpha(alp)
    plt.yticks(np.arange(1, len(order) + 1, 1), order)
    plt.xlabel('Pseudotime')
    plt.title('Clusters')

    data = []
    for i in range(len(type_lab)):
        data.append(pst[labels == i])

    ax = plt.subplot(122)
    vp = ax.violinplot(data, vert=False, showextrema=False)
    for pc in vp['bodies']:
        pc.set_facecolor(c)
        pc.set_edgecolor(c)
        pc.set_alpha(alp)
    plt.yticks(np.arange(1, len(type_lab) + 1, 1), type_lab)
    plt.xlabel('Pseudotime')
    plt.title('Signaling types')

    plt.tight_layout()
    plt.savefig(figname, format=format)