import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
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
    plt.plot(np.arange(2, max+2, 1), gap[0:max], gap_style, label='Gap between\nconsecutive\neigenvalues')
    plt.xticks(np.arange(1, max+1, 1), fontsize=tick_fontize)
    if legend:
        plt.legend(loc=legend_loc, fontsize=legend_fontsize)
    plt.xlabel('Eigenvalue rank', fontsize=axis_fontsize)
    plt.ylabel('Eigenvalue', fontsize=axis_fontsize)
    plt.title('Mode gap - ' + pathway + ' pathway')

    plt.tight_layout()
    if showfig:
        plt.showfig()
    if savefig:
        plt.savefig(figname, format=format)


'''
plot the umap of signaling modes as well as the violinplot
'''
def scatter2D(adata, key, s=2, order=None, cmap=None, legens_pos=(0.5, 1.2), legend_loc='upper center', ncol=4, legend_font=10,
                 plot_figure=True, figsize=None, showfig=False, savefig=True, figname='umap_scatter.pdf', format='pdf'):
    if order==None:
        labs = list(adata.obs[key])
        order = sorted(list(set(labs)))

    # use Set3 (12 values) for clusters/time points and Set2 for signaling models (8 values)
    if cmap==None:
        if key.endswith('modes'):
            colors = list(plt.cm.Set2.colors)
            set_figsize = (6, 4)
        else:
            colors = list(plt.cm.Set3.colors)
            set_figsize = (5, 5)
    else:
        colors = list(cmap.colors)
        set_figsize = (5, 5)


    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=set_figsize)

    ### umap plot ###
    if key.endswith('modes'):
        ax1 = plt.subplot2grid((6, 6), (0, 0), rowspan=6, colspan=4 )
    else:
        ax1 = plt.subplot()

    patches = []
    for l, c in zip(order, colors):
        coords = adata.obsm['X_umap'][adata.obs[key]==l]
        ax1.scatter( coords[:,0], coords[:,1], s=s, c=[c for i in range(coords[:,1].size)], alpha=0.5, label=l )
        patches.append( mpatches.Patch(color=c, label=l) )
    plt.axis('off')
    plt.legend(bbox_to_anchor=legens_pos, handles=patches, loc=legend_loc, ncol=ncol, fontsize=legend_font)

    if key.endswith('modes'):
        path = key.replace('_modes', '')

        labs = sorted(list(set(list(adata.obs[path + '_modes']))))
        colors = list(plt.cm.Set2.colors)[0:len(labs)]

        rec, lig = [], []
        for l in labs:
            rec.append( list(adata.obs[path + '_rec'][ adata.obs[path + '_modes']==l ]) )
            lig.append( list(adata.obs[path + '_lig'][ adata.obs[path + '_modes']==l ]) )

        # include an additional subplot if targets are included
        if path + '_tar' in adata.obs.keys():
            tar = []
            for l in labs:
                tar.append(list(adata.obs[path + '_tar'][adata.obs[path + '_modes'] == l]))
            ax2 = plt.subplot2grid((6, 6), (0, 4), rowspan=2, colspan=2)
            ax3 = plt.subplot2grid((6, 6), (2, 4), rowspan=2, colspan=2)
            ax4 = plt.subplot2grid((6, 6), (4, 4), rowspan=2, colspan=2)
        else:
            ax2 = plt.subplot2grid((6, 6), (0, 4), rowspan=3, colspan=2)
            ax3 = plt.subplot2grid((6, 6), (3, 4), rowspan=3, colspan=2)

        parts = ax2.violinplot(rec, showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        ax2.set_ylabel('receptors', fontsize=legend_font)
        plt.yticks(fontsize=legend_font)

        parts = ax3.violinplot(lig, showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        ax3.set_ylabel('Ligands', fontsize=legend_font)
        plt.xticks(np.arange(1, len(labs) + 1, 1), ['Mode '+str(l) for l in labs], rotation=45, fontsize=legend_font)
        #plt.xlabel('Signaling Modes - ' + path)

        # plot the target expression
        if path + '_tar' in adata.obs.keys():
            parts = ax4.violinplot(tar, showmeans=False, showmedians=False, showextrema=False)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_edgecolor('black')
                pc.set_alpha(1)
            ax4.set_ylabel('Targets', fontsize=legend_font)
            ax4.set_xticklabels(fontsize=legend_font)
            plt.xticks(np.arange(1, len(labs) + 1, 1), ['Mode ' + str(l) for l in labs], rotation=45, fontsize=legend_font)
            # plt.xlabel('Signaling Modes - ' + path)

    plt.tight_layout()

    if plot_figure:
        plt.tight_layout()
        if showfig:
            plt.show()
        if savefig:
            plt.savefig(figname, format=format, dpi=300)


def modes_violin(adata, key, plot_figure=True, fontsize=10, showfig=False, savefig=True, figname='modes_violin.pdf', format='pdf'):
    path = key.replace('_modes', '')
    labs = sorted(list(set(list(adata.obs[path + '_modes']))))
    colors = list(plt.cm.Set2.colors)[0:len(labs)]

    rec, lig = [], []
    for l in labs:
        rec.append(list(adata.obs[path + '_rec'][adata.obs[path + '_modes'] == l]))
        lig.append(list(adata.obs[path + '_lig'][adata.obs[path + '_modes'] == l]))

    fig = plt.figure(figsize=(1.5,2))

    ax2 = plt.subplot(211)
    parts = ax2.violinplot(rec, showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    ax2.set_ylabel('Receptors', fontsize=fontsize)
    ax2.set_xticks([])
    plt.yticks(fontsize=fontsize)

    ax3 = plt.subplot(212)
    parts = ax3.violinplot(lig, showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    ax3.set_ylabel('Ligands', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(np.arange(1, len(labs) + 1, 1), ['Mode ' + str(l) for l in labs], rotation=45, fontsize=fontsize)

    if plot_figure:
        plt.tight_layout()
        if showfig:
            plt.show()
        if savefig:
            plt.savefig(figname, format=format, dpi=300)

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
                        savefig=True, figname='types_table.pdf', format='pdf', width=6, height=6):

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

    plt.tight_layout()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)

# heterogeneity score of a given pathway for all clusters
def single_pathway_heterogeneity(adata, pathway, key, order=None, by_score=True, show_perm=True, perm_threshold=0.05, fontsize=10, xlab_rot=90,
                                 color='mediumseagreen', showfig=False, savefig=True, figname='single_pathway.pdf', format='pdf', figsize=(6,3)):

    labels = sorted(list(set(list(adata.obs[key]))))
    het = adata.uns['pathways'][pathway]['cci_heterogeneity']
    p_vals = np.asarray(adata.uns['permutation_test'][1][pathway])

    if order:
        ind = [labels.index(o) for o in order]
    elif by_score:
        ind = np.flip(np.argsort(het))
    else:
        ind = np.arange(0, het.size, 1)

    fig = plt.figure(figsize=figsize)

    # bars = plt.bar( np.arange(0, het.size, 1), 1-het[ind], color=color )  # THE (1-) NEEDS TO BE REMOVED IN THE END
    bars = plt.bar(np.arange(0, het.size, 1), het[ind], color=color)

    if show_perm:
        for pv, bar in zip(p_vals[ind], bars):
            height = bar.get_height()
            if pv<perm_threshold:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    '*',
                    ha='center',
                    va='bottom'
                )

    plt.xticks(np.arange(0, het.size, 1), [labels[i] for i in ind], rotation=xlab_rot, fontsize=fontsize)
    plt.ylabel('CCI heterogeneity', fontsize=fontsize)
    plt.ylim([0, 1.2*np.amax(het[ind])])
    plt.title(pathway)

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

    paths = list(adata.uns['pathways'].keys())
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
        plt.showfig()
    if savefig:
        plt.savefig(figname, format=format)

#########################
#
# similarity and dendrogram
#
#########################


# def redundancy(adata, include='all', key='clusters', figsize=(8, 6), fontsize=10, savefig='True', figname='path_redundancy.pdf', format='pdf'):
#     if include!='all':
#         adata = adata[adata.obs[key].isin(include)].copy()
#
#     sim_mat = pairwise_MI(adata)
#
#     mask = np.tri(sim_mat.shape[0], k=0)
#     plot_mat = np.ma.array(sim_mat, mask=mask)
#
#     print(plot_mat)
#
#     fig = plt.figure(figsize=figsize)
#     ax = plt.subplot(111)
#
#     cmap = cm.get_cmap("Reds").copy()
#     cmap.set_bad('w')
#
#     pt = plt.pcolor(plot_mat.T, cmap=cmap, vmin=0, vmax=np.amax(plot_mat))
#
#     for i in range(plot_mat.shape[0] + 1):
#         if i == plot_mat.shape[0]:
#             plt.plot([0, i - 1], [i, i], 'k-', lw=0.5)
#         else:
#             plt.plot([0, i], [i, i], 'k-', lw=0.5)
#     for i in range(plot_mat.shape[0] + 1):
#         if i == 0:
#             plt.plot([i, i], [plot_mat.shape[0], i + 1], 'k-', lw=0.5)
#         else:
#             plt.plot([i, i], [plot_mat.shape[0], i], 'k-', lw=0.5)
#
#     cbar = plt.colorbar(pt)
#     cbar.set_label('Pairwise Redundancy', fontsize=fontsize)
#     cbar.ax.tick_params(labelsize=fontsize)
#     labs = list(adata.uns['pathways'].keys())
#     plt.xticks(np.arange(0, len(labs), 1) + 0.5, labs, rotation=90, fontsize=fontsize)
#     plt.yticks(np.arange(0, len(labs), 1) + 0.5, labs, fontsize=fontsize)
#
#     ax.spines.right.set_visible(False)
#     ax.spines.bottom.set_visible(False)
#     ax.xaxis.tick_top()
#
#     plt.tight_layout()
#     if savefig:
#         plt.savefig(figname, format=format)
#
#
#
#
# def plot_dendrogram(adata, ax, model, cmap=plt.cm.tab20, color_threshold=None, fontsize=10, above_threshold_color='tomato', **kwargs):
#     # Create linkage matrix and then plot the dendrogram
#
#     # create the counts of samples under each node
#     counts = np.zeros(model.children_.shape[0])
#     n_samples = len(model.labels_)
#     for i, merge in enumerate(model.children_):
#         current_count = 0
#         for child_idx in merge:
#             if child_idx < n_samples:
#                 current_count += 1  # leaf node
#             else:
#                 current_count += counts[child_idx - n_samples]
#         counts[i] = current_count
#
#     linkage_matrix = np.column_stack(
#         [model.children_, model.distances_, counts]
#     ).astype(float)
#
#     labels = list(adata.uns['pathways'].keys())
#
#
#     # Plot the corresponding dendrogram
#     colors = list(cmap.colors)
#     hex_colors = [ to_hex(c) for c in colors ]
#     hierarchy.set_link_color_palette(hex_colors)
#     hierarchy.dendrogram(linkage_matrix, ax=ax, orientation='right', labels=labels,
#                          color_threshold=color_threshold, above_threshold_color=above_threshold_color, **kwargs)
#
#
#
# def pathway_hierarchy(adata, affinity='precomputed', linkage='complete', distance_threshold=0, n_clusters=None,
#                       title='Pathway Hierarchical Clustering', cmap=plt.cm.tab20, color_threshold=None, above_threshold_color='tomato',
#                       xlim=False, show_ticks=False, fontsize=10,
#                       figsize=(4,6), orientation='vertical', savefig='True', figname='dendrogram.pdf', format='pdf', **kwargs):
#     sim_mat = pairwise_MI(adata)
#     dist = 1 - sim_mat
#     model = AgglomerativeClustering(affinity=affinity, linkage=linkage, distance_threshold=distance_threshold,
#                                     n_clusters=n_clusters).fit(dist)
#
#     plt.figure(figsize=figsize)
#     ax = plt.subplot(111)
#     plot_dendrogram(adata, ax, model, cmap=cmap, color_threshold=color_threshold, above_threshold_color=above_threshold_color, fontsize=fontsize, **kwargs)
#     if not show_ticks:
#         ax.set_xticks([])
#     if xlim:
#         plt.xlim(xlim)
#     if title:
#         plt.title(title)
#     plt.tight_layout()
#     if savefig:
#         plt.savefig(figname, format=format)
#


##########################
#
# alluvial plot functions
#
##########################

# def compute_weight(list1, list2, v1, v2):
#     w = 0
#     for l1, l2 in zip(list1, list2):
#         if l1==v1 and l2==v2:
#             w = w + 1
#     return w
#
# def set_color(map, features):
#     # convert colors from matplotlib colormap to RGB
#     colors = list(map.colors)[0:len(features)]
#     colorlist = []
#     for i in range(len(features)):
#         rgb_scaled = tuple([255 * c for c in colors[i]])
#         colorlist.append('rgb' + str(rgb_scaled))
#     return colorlist
#
#
# def twostate_sankey(v1, v2, lab1, lab2, col1, col2, name1, name2, alpha=0.5,
#                      pad=5, thickness=40, linecolor='black', linewidth=1, width=400, height=600, font_size=15,
#                     savefig='True', figname='compare_paths.pdf', format='pdf', showfig=True):
#
#     label=lab1+lab2
#
#     source, target, value, edge_color = [], [], [], []
#     for n1 in lab1:
#         for n2 in lab2:
#             source.append(label.index(n1))
#             target.append(label.index(n2))
#             value.append(compute_weight(np.asarray(v1), np.asarray(v2), n1, n2))
#             edge_color.append('rgba' + col2[lab2.index(n2)][3:-1] + ', ' + str(alpha) + ')')
#
#
#     fig = go.Figure(data=[go.Sankey(
#         node=dict(
#             pad=pad,  # separation between nodes
#             thickness=thickness,
#             line=dict(color=linecolor, width=linewidth),
#             label=label,
#             color=col1+col2
#         ),
#         link=dict(
#             source=source,
#             target=target,
#             value=value,
#             color=edge_color
#         ))])
#     fig.add_annotation(text=name1, xref="paper", yref="paper", x=0., y=1.1, showarrow=False, align='center')
#     fig.add_annotation(text=name2, xref="paper", yref="paper", x=1.05, y=1.1, showarrow=False,
#                        align='center')
#     fig.update_layout(autosize=False, width=width, height=height, font_size=font_size)
#     if savefig:
#         fig.write_image(figname, format=format)
#     # if showfig:
#     #     fig.show()
#
#
#
# def alluvial_onepath(adata, pathway, key, map1=plt.cm.Set3, map2=plt.cm.Set2, alpha=0.5,
#                      pad=5, thickness=40, linecolor='black', linewidth=1, width=400, height=600, font_size=15,
#                      savefig='True', figname='alluvial.pdf', format='pdf', showfig=True):
#
#     memb = list(adata.obs[key])
#     p = list(adata.obs[pathway+'_modes'])
#
#     states, modes = sorted(list(set(memb))),  sorted(list(set(p)))
#
#     # set colors for source and targets
#     source_color = set_color(map1, states)
#     target_color = set_color(map2, modes)
#
#     twostate_sankey(memb, p, states, modes, source_color, target_color, 'Cell Type', pathway+' Mode', alpha=alpha,
#                     pad=pad, thickness=thickness, linecolor=linecolor, linewidth=linewidth, width=width,
#                     height=height, font_size=font_size,
#                     savefig=savefig, figname=figname, format=format, showfig=showfig)
#
#
# def alluvial_twopath(adata, pathway1, pathway2, include='all', key='clusters', map1=plt.cm.Set3, map2=plt.cm.Set2, alpha=0.5,
#                      pad=5, thickness=40, linecolor='black', linewidth=1, width=500, height=500, font_size=15,
#                      savefig='True', figname='compare_paths.pdf', format='pdf'):
#     if include!='all':
#         adata = adata[adata.obs[key].isin(include)].copy()
#     p1 = list(adata.obs[pathway1 + '_modes'])
#     p2 = list(adata.obs[pathway2 + '_modes'])
#     modes1, modes2 = sorted(list(set(p1))), sorted(list(set(p2)))
#
#     # if signaling modes were not renamed, add the pathway name to distinguish
#     if(modes1[0]==modes2[0]):
#         p1 = [str(p) + '-' + pathway1 for p in p1]
#         p2 = [str(p) + '-' + pathway2 for p in p2]
#         # p1 = [str(p) for p in p1]
#         # p2 = [str(p) for p in p2]
#         modes1, modes2 = sorted(list(set(p1))), sorted(list(set(p2)))
#
#     # set colors for source and targets
#     source_color = set_color(map1, modes1)
#     target_color = set_color(map2, modes2)
#
#     twostate_sankey(p1, p2, modes1, modes2, source_color, target_color, pathway1, pathway2, alpha=alpha,
#                     pad=pad, thickness=thickness, linecolor=linecolor, linewidth=linewidth, width=width,
#                     height=height, font_size=font_size,
#                     savefig=savefig, figname=figname, format=format)
#
#
# def alluvial_pattern(adata, key, map1=plt.cm.Set3, map2=plt.cm.tab20, alpha=0.5,
#                      pad=5, thickness=40, linecolor='black', linewidth=1, width=400, height=600, font_size=15,
#                      savefig='True', figname='patterns.pdf', format='pdf'):
#     clusters, patterns = list(adata.obs[key]), np.asarray(adata.obs['sign_pattern'], dtype='int')
#
#     states, modes = sorted(list(set(clusters))), sorted(list(set(patterns)))
#
#     source_color = set_color(map1, states)
#     target_color = set_color(map2, modes)
#
#     twostate_sankey(clusters, patterns, states, modes, source_color, target_color, 'Cell Type', 'Pattern', alpha=alpha,
#                     pad=pad, thickness=thickness, linecolor=linecolor, linewidth=linewidth, width=width,
#                     height=height, font_size=font_size,
#                     savefig=savefig, figname=figname, format=format)
#



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