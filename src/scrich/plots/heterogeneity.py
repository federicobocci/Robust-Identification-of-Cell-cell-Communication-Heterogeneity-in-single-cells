"""Plots of cell-type heterogeneity of communication modes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np

from ._utils import _fraction_matrix, _new_axes, discrete_colors, plot_function


@plot_function(aliases={'annotation': 'key'})
def heatmap_one_pathway(adata, pathway, key, order=None, legend_top=False, legend_loc_top='best', legend_loc_right='best',
                        ncol_top=1, ncol_right=1, ylim_top=False, xlim_right=False, legend_font=10,
                        plot_figure=True, showfig=False, savefig=True, figname='types_table.pdf', format='pdf', width=6, height=6):

    if pathway + '_modes' not in adata.obs:
        raise KeyError(f"adata.obs[{pathway + '_modes'!r}] not found. Run scrich.tl.all_path_sim() first.")

    # use anchor to put legend on right
    # order of clusters as input
    # use another colormap for the modes

    frac, lab_set, clst_set = _fraction_matrix(adata, pathway, key, order)
    ntypes, nclst = len(lab_set), len(clst_set)

    fig = plt.figure(figsize=(width, height))

    x = np.arange(0.5, ntypes, 1)
    y = np.arange(0.5, nclst, 1)

    # break down types based on annotations
    ax1 = plt.subplot2grid((6, 6), (0, 0), rowspan=2, colspan=4)
    ax1.set_prop_cycle(color=discrete_colors(plt.cm.Set3, nclst))

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
    plt.xticks(x, [str(m) for m in lab_set])
    plt.yticks(y, clst_set)

    # break down clusters based on types
    ax1 = plt.subplot2grid((6, 6), (2, 4), rowspan=4, colspan=2)
    ax1.set_prop_cycle(color=discrete_colors(plt.cm.Set2, ntypes))

    clst_norm = np.sum(frac, axis=0)

    for i in range(ntypes):
        bot = 0.
        if i > 0:
            for j in range(i):
                bot = bot + frac[j]
        plt.barh(y, frac[i]/clst_norm, left=bot/clst_norm, align='center', alpha=0.75, label='Mode ' + str(lab_set[i]))
    plt.yticks([])
    plt.ylim([0, y[-1]+0.5])
    if xlim_right:
        plt.xlim([0, xlim_right])
    plt.xlabel('Mode distribution')
    plt.legend(loc=legend_loc_right, ncol=ncol_right, fontsize=legend_font, fancybox=True, framealpha=0.5)
    # plt.legend(bbox_to_anchor=(0, 1.05), loc='lower left', ncol=ncol_right, fontsize=legend_font)

    if plot_figure:
        plt.tight_layout()
        if savefig:
            plt.savefig(figname, format=format, dpi=300)
        if showfig:
            plt.show()


@plot_function()
def single_pathway_heterogeneity(adata, pathway, key, order=None, by_score=True, show_perm=True, perm_threshold=0.05, fontsize=10, xlab_rot=90, horizontal=True,
                                 color='mediumseagreen', ax=None, showfig=False, savefig=True, figname='single_pathway.pdf', format='pdf', figsize=(6,3), strip_text=False):

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

    fig, ax = _new_axes(ax, figsize)

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
    if savefig:
        plt.savefig(figname, format=format, dpi=300)
    if showfig:
        plt.show()


@plot_function()
def pathway_heterogeneity_summary(adata, key='clusters', state='all', fontsize=10, color='lightskyblue', edgecolor='k', lw=1, errorbar=False, orientation='vertical',
                                  ax=None, showfig=False, savefig=True, figname='pathways_summary.pdf', format='pdf', figsize=(3,5)):
    if orientation not in ('vertical', 'horizontal'):
        raise ValueError(f"orientation must be 'vertical' or 'horizontal', got {orientation!r}")

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

    fig, ax = _new_axes(ax, figsize)

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
    if savefig:
        plt.savefig(figname, format=format, dpi=300)
    if showfig:
        plt.show()


@plot_function()
def state_heterogeneity_summary(adata, key, order=None, fontsize=10, color='lightskyblue', edgecolor='k', lw=1, errorbar=False, orientation='vertical',
                                  ax=None, showfig=False, savefig=True, figname='states_summary.pdf', format='pdf', figsize=(3,5)):

    if orientation not in ('vertical', 'horizontal'):
        raise ValueError(f"orientation must be 'vertical' or 'horizontal', got {orientation!r}")

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

    fig, ax = _new_axes(ax, figsize)

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
    if savefig:
        plt.savefig(figname, format=format, dpi=300)
    if showfig:
        plt.show()


@plot_function()
def heterogeneity_heatmap(adata, key, fontsize=10, cmap='Reds', ax=None, showfig=False, savefig=True, figname='het_table.pdf', format='pdf', figsize=(6,5)):
    # only pathways analysed with all_path_sim have a heterogeneity score
    paths = [p for p in adata.uns['pathways'] if 'cci_heterogeneity' in adata.uns['pathways'][p]]
    labels = sorted(list(set(list(adata.obs[key]))))

    het_mat = np.zeros((len(paths), len(labels)))
    for i in range(len(paths)):
        v_het = adata.uns['pathways'][paths[i]]['cci_heterogeneity']
        for j in range(len(labels)):
            het_mat[i][j] = v_het[j]

    fig, ax = _new_axes(ax, figsize)

    x, y = np.arange(0, len(paths)+1, 1), np.arange(0, len(labels)+1, 1)
    pt = plt.pcolor(x, y, het_mat.transpose(), cmap=cmap, vmin=0, vmax=np.amax(het_mat))
    cbar = plt.colorbar(pt)
    cbar.set_label(label='Pathway heterogeneity',size=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    plt.xticks( x[0:-1]+0.5, paths, rotation=90, fontsize=fontsize)
    plt.yticks( y[0:-1]+0.5, labels, fontsize=fontsize)

    plt.tight_layout()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)
    if showfig:
        plt.show()


@plot_function()
def mode_composition(adata, key, pathway, mode, order=None, rename_states=None, fontsize=7, ticksize=7, ax=None,
                     figsize=(3,3), showfig=False, savefig=True, figname='mode_composition.pdf', format='pdf'):
    # plot the composition of a signaling mode - normalize based on the total number of cells in each cluster
    frac, lab_set, clst_set = _fraction_matrix(adata, pathway, key, order)
    tot_cells = np.sum(frac, axis=0)
    # 'mode' can be the mode label (e.g. 'Receiver' after rename_modes) or its row index
    row = lab_set.index(mode) if mode in lab_set else mode
    fracs = frac[row]
    colors = discrete_colors(plt.cm.Accent, fracs.size)

    fig, ax = _new_axes(ax, figsize)
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
    if savefig:
        plt.savefig(figname, format=format)
    if showfig:
        plt.show()
