import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches


def select_points(x, initial_list, labels):
    x_final, y_final, label_final = [], [], []
    for i in range(len(x)):
        if len(initial_list[i]) > 0:
            x_final.append(x[i])
            y_final.append(initial_list[i])
            label_final.append(labels[i])
    return x_final, y_final, label_final


def violin_part(ax, labs, expr_list, states, colors, path, ylabel, ticks=True, legend=True, ncol=1, legend_loc='best', fontsize=10):
    gap = len(labs) + 2
    x = np.asarray([gap * i for i in range(len(expr_list[0]))])

    patch_color, patch_text = [], []

    for i in range(len(labs)):
        x_final, y_final, label_final = select_points(x + i, expr_list[i], sorted(set(states)))
        # further customize violin plot here:
        # https://matplotlib.org/stable/gallery/statistics/customized_violin.html#sphx-glr-gallery-statistics-customized-violin-py
        parts = ax.violinplot(y_final, positions=x_final, showmeans=False, showmedians=False,
                              showextrema=False)

        patch_color.append(colors[i])
        patch_text.append(str(labs[i]))

        for pc in parts['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(1)

    # set the labels on x-axis:
    n = len(labs)
    xmin, xmax = ax.get_xlim()
    start = n / 2 - 1 if n % 2 == 1 else n / 2 - 0.5
    if ticks:
        ax.set_xticks(np.arange(start, xmax, gap), sorted(set(states)), rotation=45, fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)
    else:
        ax.set_xticks([])
    ax.set_ylabel(path + ' ' + ylabel, fontsize=fontsize)

    ymin, ymax = ax.get_ylim()
    for x in np.arange(gap - 1.5, x[-1], gap):
        ax.plot([x, x], [ymin, ymax], 'k-', linewidth=0.5)
    ax.set_ylim([ymin, ymax])

    if legend:
        patches = [mpatches.Patch(color=colors[i], label="{:s}".format(patch_text[i])) for i in range(len(patch_text))]
        ax.legend(handles=patches, loc=legend_loc, fontsize=fontsize, ncol=ncol)


def violin(adata, path, key='clusters', target=True, moments=True, legend_loc='best', ncol=1, fontsize=10,
           figsize=None, plot_figure=True, showfig=False, savefig=True, figname='violin.pdf', format='pdf'):

    include_tar = False
    if target and 'TF' in adata.uns.keys():
        if path in adata.uns['TF'].keys():
            tar_list = adata.uns['TF'][path]
            if moments:
                tar = adata[:, tar_list].layers['Mu'].mean(axis=1) + adata[:, tar_list].layers['Ms'].mean(axis=1)
            else:
                tar = adata[:, tar_list].X.toarray().mean(axis=1)
            include_tar = True


    labs = sorted(list(set(list(adata.obs[path + '_modes']))))
    states = np.asarray(adata.obs[key])
    modes = np.asarray(adata.obs[path + '_modes'])
    lig, rec = np.asarray(adata.obs[path + '_lig']), np.asarray(adata.obs[path + '_rec'])
    # if include_tar:
    #     tar = np.asarray(adata.obs[path + '_tar'])
    colors = list(plt.cm.Set2.colors)[0:len(labs)]

    lig_list, rec_list, tar_list = [[] for l in labs], [[] for l in labs], [[] for l in labs]
    for s in sorted(set(states)):
        lig_sel, rec_sel, modes_sel = lig[states == s], rec[states == s], modes[states == s]
        if include_tar:
            tar_sel = tar[states == s]

        for i in range(len(labs)):
            lig_list[i].append(lig_sel[modes_sel == labs[i]])
            rec_list[i].append(rec_sel[modes_sel == labs[i]])
            if include_tar:
                tar_list[i].append(tar_sel[modes_sel == labs[i]])

    if figsize and include_tar:
        fig = plt.figure(figsize=figsize)
        ax1, ax2, ax3 = plt.subplot(311), plt.subplot(312), plt.subplot(313)
    elif figsize and not(include_tar):
        fig = plt.figure(figsize=figsize)
        ax1, ax2 = plt.subplot(211), plt.subplot(212)
    elif not(figsize) and include_tar:
        fig = plt.figure(figsize=(8,9))
        ax1, ax2, ax3 = plt.subplot(311), plt.subplot(312), plt.subplot(313)
    elif not(figsize) and not(include_tar):
        fig = plt.figure(figsize=(8,6))
        ax1, ax2 = plt.subplot(211), plt.subplot(212)

    violin_part(ax1, labs, lig_list, states, colors, path, 'ligands', ticks=False, legend=True, ncol=ncol, legend_loc=legend_loc, fontsize=fontsize)
    # plot the receptor violinplot without ticks on the x-axis of targets are plotted as well, otherwise plot the xticks
    if include_tar:
        violin_part(ax2, labs, rec_list, states, colors, path, 'receptors', ticks=False, legend=False,
                    legend_loc=legend_loc, fontsize=fontsize)
        violin_part(ax3, labs, tar_list, states, colors, path, 'targets', ticks=True, legend=False, legend_loc=legend_loc, fontsize=fontsize)
    else:
        violin_part(ax2, labs, rec_list, states, colors, path, 'receptors', ticks=True, legend=False,
                    legend_loc=legend_loc, fontsize=fontsize)

    if plot_figure:
        plt.tight_layout()
        if showfig:
            plt.show()
        if savefig:
            plt.savefig(figname, format=format, dpi=300)
