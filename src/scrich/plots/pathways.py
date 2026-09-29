"""Plots of the CCC pathways detected in the data."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from ._utils import plot_function


@plot_function()
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
    if 'paths_overview' not in adata.uns:
        raise KeyError("Run scrich.tl.pathways_overview() before scrich.pl.pathways_overview().")

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
    if savefig:
        plt.savefig(figname, format=format, dpi=dpi)
    if showfig:
        plt.show()
