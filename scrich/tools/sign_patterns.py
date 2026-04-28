import numpy as np
import matplotlib.pyplot as plt
import anndata
import scanpy as sc

from .database_func import loadDB

def find_sign_patterns(adata, paths='all', res=1):
    '''
    use scanpy clustering to find signaling patterns
    '''
    if paths=='all':
        # paths = list(adata.uns['pathways'].keys())
        paths = adata.uns['selected_pathways']
    ncell, npath = adata.shape[0], len(paths)

    sel_data = np.zeros((npath, ncell))
    for i in range(len(paths)):
        sel_data[i] = adata.obs[paths[i]+'_modes']

    path_adata = anndata.AnnData(X=sel_data.transpose())
    path_adata.obs_names = adata.obs_names
    path_adata.var_names = paths

    sc.tl.pca(path_adata, svd_solver='arpack')
    sc.pp.neighbors(path_adata, n_neighbors=10, n_pcs=len(paths))
    sc.tl.umap(path_adata)
    sc.tl.leiden(path_adata, resolution=res)

    adata.obs['sign_pattern'] = path_adata.obs['leiden']

    sc.tl.rank_genes_groups(adata, 'sign_pattern')


# def find_pathway(gene, lig, rec, lig_set, rec_set):
#     if gene in lig_set:
#         i = list(lig['pathway'][ lig['ligand']==gene ])
#         return i[0]
#     elif gene in rec_set:
#         i = list(rec['pathway'][ rec['receptor']==gene ])
#         return i[0]
#
# def unique(paths):
#     unique = []
#     for p in paths:
#         if not(p in unique):
#             unique.append(p)
#     return unique
#
# def pattern_marker(ax, adata, pattern, ntop=25, human=True, cmap=plt.cm.tab20,
#                    fontsize=10):
#
#     path_list = list(adata.uns['pathways'].keys())
#     # if human:
#     #     lig = loadDB('/Users/federicobocci/Desktop/commflow_project/ccc_project/pathway_database/ligand_human.csv')
#     #     rec = loadDB('/Users/federicobocci/Desktop/commflow_project/ccc_project/pathway_database/receptor_human.csv')
#     # else:
#     #     lig = loadDB('/Users/federicobocci/Desktop/commflow_project/ccc_project/pathway_database/ligand_mouse.csv')
#     #     rec = loadDB('/Users/federicobocci/Desktop/commflow_project/ccc_project/pathway_database/receptor_mouse.csv')
#     if human:
#         lig = loadDB('pathway_database/receptor_human.csv')
#         rec = loadDB('pathway_database/receptor_human.csv')
#     else:
#         lig = loadDB('pathway_database/ligand_mouse.csv')
#         rec = loadDB('pathway_database/receptor_mouse.csv')
#     lig_set = list(lig['ligand'])
#     rec_set = list(rec['receptor'])
#
#     gene, score, path = [], [], []
#
#     i = 0
#     while len(gene)<ntop:
#         all_genes = adata.uns['rank_genes_groups']['names'][i].tolist()
#         all_scores = np.array(adata.uns['rank_genes_groups']['scores'][i].tolist())
#         p = find_pathway(all_genes[pattern], lig, rec, lig_set, rec_set)
#         i = i + 1
#         if p in path_list:
#             gene.append(all_genes[pattern])
#             score.append(all_scores[pattern])
#             path.append(p)
#
#     x = np.arange(1, len(gene) + 1, 1)
#     y = np.flip(np.asarray(score))
#     lab = np.flip(np.asarray(gene))
#     path = np.flip(np.asarray(path))
#
#     colors = list(cmap.colors)
#     color_dict = {}
#
#     unique_paths = list(set(path))
#     for u, c in zip(unique_paths, colors):
#         ax.barh(x[path==u], y[path==u], height=0.8, color=c, alpha=0.5, label=u)
#         color_dict[u] = c
#
#     plt.xticks(fontsize=fontsize)
#     plt.yticks(x, lab, fontsize=fontsize)
#     plt.xlabel('Score', fontsize=fontsize)
#     # plt.legend(loc='lower right', fontsize=fontsize)
#     plt.title('Pattern ' + str(pattern) + ' vs. Rest', fontsize=fontsize)
#
#     unique_paths = unique(np.flip(path))
#     return unique_paths, color_dict
#
#
#
# def state_frac(states, clusters):
#     '''
#     compute fractions of cells of each state
#     '''
#     frac0 = np.zeros(len(states))
#     for i in range(len(states)):
#         frac0[i] = clusters[clusters == states[i]].size
#     frac0 = frac0 / np.sum(frac0)
#     return frac0
#
#
# def state_repr(ax, adata, pattern_adata, key, order, fontsize=10, bar_color='b', bar_alpha=0.5):
#     '''
#     plot the over/under representation of cell states in the pattern
#     '''
#
#     clusters0 = np.asarray(adata.obs[key])
#     clusters_patt = np.asarray(pattern_adata.obs[key])
#
#     if order:
#         states = order
#     else:
#         states = list(set(clusters0))
#
#     frac0 = state_frac(states, clusters0)
#     frac = state_frac(states, clusters_patt)
#
#     rel_repr = (frac-frac0)/frac0
#     x = np.arange(1, len(states) + 1, 1)
#
#     x_pos, y_pos = x[rel_repr>0], rel_repr[rel_repr>0]
#     x_neg, y_neg = x[rel_repr<0], rel_repr[rel_repr<0]
#
#     lim = 1.1*np.amax(np.abs(rel_repr))
#
#     ax.barh(x_pos, y_pos, height=0.8, color='r', alpha=bar_alpha)
#     ax.barh(x_neg, y_neg, height=0.8, color='b', alpha=bar_alpha)
#     plt.xlim( [ -lim, lim ] )
#     plt.yticks(np.arange(1, len(states) + 1, 1), states, fontsize=fontsize)
#     plt.xticks(fontsize=fontsize)
#     # ax.set_xlabel('State Over/Under\nRepresentation (fold-change)', fontsize=fontsize)
#     ax.set_xlabel('Pattern composition', fontsize=fontsize)
#
#
#
# def pattern(adata, p, key='cluster', order=None, human=True, ntop=25, marker_cmap= plt.cm.tab20,                       # general function parameters
#             fontsize=10,                                                               # plotting parameter for all plots
#             lig_color='b', rec_color='r', showmean=False, meanwidth=1, meancolor='k',   # plotting parameter for violin plots
#             bar_color='b', bar_alpha=0.5, figsize=(15, 6),                                                  # plotting parameter for composition plot
#             savefig=True, figname='pattern_detail.pdf', showfig=False, format='pdf', dpi=300       # figure parameters
#             ):
#
#     assert 'sign_pattern' in adata.obs.keys(), "Run signaling pattern analysis before calling pattern()"
#
#     pattern_adata = adata[adata.obs['sign_pattern'] == str(p)]
#
#     fig = plt.figure(figsize=figsize)
#
#     ax1 = plt.subplot2grid((7, 15), (0, 0), rowspan=6, colspan=2)
#     state_repr(ax1, adata, pattern_adata, key, order, fontsize=fontsize)
#
#     ax2 = plt.subplot2grid((7, 15), (0, 4), rowspan=6, colspan=4)
#     unique_paths, color_dict = pattern_marker(ax2, adata, p, ntop=ntop, human=human, cmap=marker_cmap, fontsize=fontsize)
#
#     ax3 = plt.subplot2grid((7, 15), (0, 10), rowspan=3, colspan=5)
#
#     rec_list, lig_list = [], []
#     base_rec, base_lig = [], []
#     colorlist = []
#     for p in unique_paths[0:5]:
#         rec_list.append( np.asarray(pattern_adata.obs[p+'_rec']) )
#         lig_list.append( np.asarray(pattern_adata.obs[p+'_lig']) )
#         base_rec.append( np.mean(np.asarray(adata.obs[p+'_rec'])) )
#         base_lig.append( np.mean(np.asarray(adata.obs[p+'_lig'])) )
#         colorlist.append( color_dict[p] )
#
#
#     parts = ax3.violinplot(rec_list, showmeans=False, showmedians=False, showextrema=False)
#     for i, pc in enumerate(parts['bodies']):
#         pc.set_facecolor(colorlist[i])
#         pc.set_edgecolor('black')
#         pc.set_alpha(0.5)
#
#     ax3.set_ylabel('Receptors', fontsize=fontsize)
#     ax3.set_xticks([])
#     plt.xticks(fontsize=fontsize)
#     plt.yticks(fontsize=fontsize)
#     # ax3.set_yticks(fontsize=fontsize)
#     # ax3.tick_params(axis='both', which='minor', labelsize=fontsize)
#
#     ax4 = plt.subplot2grid((7, 15), (3, 10), rowspan=3, colspan=5)
#     parts = ax4.violinplot(lig_list, showmeans=False, showmedians=False, showextrema=False)
#     for i, pc in enumerate(parts['bodies']):
#         pc.set_facecolor(colorlist[i])
#         pc.set_edgecolor('black')
#         pc.set_alpha(0.5)
#
#     ax4.set_ylabel('Ligands', fontsize=fontsize)
#     plt.xticks( np.arange(1, len(unique_paths[0:5])+1, 1), unique_paths[0:5], rotation=45, fontsize=fontsize)
#     plt.yticks(fontsize=fontsize)
#
#     if showmean:
#         for i in range(len(base_rec)):
#             ax3.plot([i + 0.6, i + 1.4], [base_rec[i], base_rec[i]], lw=meanwidth, color=meancolor)
#             ax4.plot([i + 0.6, i + 1.4], [base_lig[i], base_lig[i]], lw=meanwidth, color=meancolor)
#
#     plt.tight_layout()
#     if showfig:
#         plt.show()
#     if savefig:
#         plt.savefig(figname, format=format)
#
#
# def pattern_summary(adata, key, order=None, panel_per_row=4, panel_length=3, panel_height=3, cmap=plt.cm.tab20,
#                     fontsize=10,
#                     savefig=True, figname='pattern_summary.pdf', showfig=False, format='pdf', dpi=300):
#     if order:
#         states = order
#     else:
#         states = list(set(list(adata.obs[key])))
#
#     patterns = np.sort(np.array(list( set(list(adata.obs['sign_pattern'])) ), dtype='int'))
#     bins = np.append(patterns-0.5, patterns[-1]+0.5)
#
#     n = len(states)
#     nrow = int(n / panel_per_row) + 1 if n % panel_per_row > 0 else int(n / panel_per_row)
#     ncol = max(n % panel_per_row, panel_per_row)
#
#     fig = plt.figure(figsize=(panel_length*ncol, panel_height*nrow))
#     colors = list(cmap.colors)
#
#     for i in range(len(states)):
#         sel_adata = adata[ adata.obs[key]==states[i] ]
#         frac = np.asarray(sel_adata.obs['sign_pattern'], dtype='int')
#         dist, bins = np.histogram(frac, bins=bins)
#         dist = dist/np.sum(dist)
#
#         # panel coordinates for plotting
#         j = int(i / panel_per_row)
#         k = i % panel_per_row
#
#         ax = plt.subplot2grid((nrow, ncol), (j, k), rowspan=1, colspan=1)
#
#         for j in range(patterns.size):
#             plt.bar( np.array([patterns[j]]), [dist[j]], color=colors[j] )
#
#         # plt.bar(patterns, dist)
#         plt.xticks(patterns, fontsize=fontsize)
#         plt.title(states[i], fontsize=fontsize)
#         plt.xlabel('Signaling Pattern', fontsize=fontsize)
#         plt.ylabel('Cell Fraction', fontsize=fontsize)
#     plt.tight_layout()
#
#     plt.tight_layout()
#     if showfig:
#         plt.show()
#     if savefig:
#         plt.savefig(figname, format=format)
