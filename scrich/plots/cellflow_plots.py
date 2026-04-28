import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import scanpy as sc

def NormalizeData(data):
    scaled = (data - np.min(data)) / (np.max(data) - np.min(data))
    return scaled

def grn_plot(grn_name, path1, path2, time, weight_quantile=0.5, filepath=''):
    A = np.loadtxt(filepath + grn_name + '_' + time + '_grn.txt')

    genes_df = pd.read_csv(filepath + grn_name + '_' + time + '_gene_annotation.csv')
    genes, path = list(genes_df['gene']), list(genes_df['annotation'])
    col = []
    for p in path:
        if p==path1:
            col.append('plum')
        elif p==path2:
            col.append('mediumseagreen')
        else:
            col.append('skyblue')

    q_pos = np.quantile(A[A > 0], weight_quantile)
    q_neg = np.quantile(A[A < 0], 1 - weight_quantile)
    A[(A > q_neg) & (A < q_pos)] = 0

    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    nx.relabel_nodes(G, dict(zip(range(len(genes)), genes)), copy=False)

    G_undirected = G.to_undirected()
    subgraphs = [G.subgraph(c) for c in sorted(nx.connected_components(G_undirected), key=len, reverse=True) if
                 len(c) > 1]

    if len(subgraphs) > 1:
        print("There exist multiple connected components. Choose the parameter cc_id to show other components")

    pos = nx.spring_layout(G, seed=0)

    ### edges and weights
    epos = []
    eneg = []
    wpos = []
    wneg = []

    for (u, v, d) in G.edges(data=True):
        if d["weight"] > 0:
            epos.append((u, v))
            wpos.append(d["weight"])

        elif d["weight"] < 0:
            eneg.append((u, v))
            wneg.append(d["weight"])

    edge_width_pos = NormalizeData(np.array(wpos))
    edge_width_neg = NormalizeData(-np.array(wneg))


    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111)
    nx.draw_networkx_nodes(G, pos, node_color=col, alpha=0.5, node_size=700)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

    nx.draw_networkx_edges(G, pos, edgelist=epos, width=edge_width_pos + 0.5, edge_color='b',
                           arrowsize=10, alpha=0.75, connectionstyle='arc3')
    nx.draw_networkx_edges(G, pos, edgelist=eneg, width=edge_width_neg + 0.5, edge_color='r',
                           arrowstyle="-[", arrowsize=10, alpha=0.75, connectionstyle='arc3')

    patch1 = mpatches.Patch(color='plum', label=path1)
    patch2 = mpatches.Patch(color='mediumseagreen', label=path2)
    patch3 = mpatches.Patch(color='skyblue', label='shared')
    plt.legend(handles=[patch1, patch2, patch3])

    fig.tight_layout()
    plt.axis("off")
    plt.savefig(filepath + grn_name + '_' + time + '_GRN.pdf', format='pdf', dpi=300)


def regulation_plot(adata, path_list, path_weight, grn_name, path1, path2, state, weight_quantile=0.5, ntop=5, scale=1,
                    title=None, filepath='', verbose=True):
    # select the top n paths to color
    if len(path_list)<ntop:
        sel_path = path_list
    else:
        inds = np.flip(np.argsort(np.asarray(path_weight)))[0:ntop]
        sel_path = [path_list[i] for i in inds]

    if verbose:
        print('key regulatory paths connecting the ' + path1 + ' and ' + path2 + ' pathways:')
        print(sel_path)

    edge_list = []
    for p in sel_path:
        for i in range(len(p)-1):
            edge_list.append([p[i], p[i+1]])

    # print(path_list)
    A = np.loadtxt(filepath + grn_name + '_' + state + '_grn.txt')

    genes_df = pd.read_csv(filepath + grn_name + '_' + state + '_gene_annotation.csv')
    genes, path = list(genes_df['gene']), list(genes_df['annotation'])
    col = []
    for p in path:
        if p == path1:
            col.append('plum')
        elif p == path2:
            col.append('mediumseagreen')
        else:
            col.append('skyblue')

    q_pos = np.quantile(A[A > 0], weight_quantile)
    q_neg = np.quantile(A[A < 0], 1 - weight_quantile)
    A[(A > q_neg) & (A < q_pos)] = 0

    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    nx.relabel_nodes(G, dict(zip(range(len(genes)), genes)), copy=False)

    G_undirected = G.to_undirected()
    subgraphs = [G.subgraph(c) for c in sorted(nx.connected_components(G_undirected), key=len, reverse=True) if
                 len(c) > 1]

    if len(subgraphs) > 1:
        print("There exist multiple connected components. Choose the parameter cc_id to show other components")

    pos = nx.spring_layout(G, seed=0, scale=scale)

    rec1 = adata.uns['pathways'][path1]['receptors'].tolist()
    rec2 = adata.uns['pathways'][path2]['receptors'].tolist()
    lig1 = adata.uns['pathways'][path1]['ligands'].tolist()
    lig2 = adata.uns['pathways'][path2]['ligands'].tolist()

    layer1 = [g for g in rec1 + lig1 if g in genes]
    layer3 = [g for g in rec2 + lig2 if g in genes]
    layer1_points = len(layer1)
    layer3_points = len(layer3)

    y1 = np.linspace(-0.5, 0.5, num=layer1_points)
    y3 = np.linspace(-0.5, 0.5, num=layer3_points)

    for i in range(len(layer1)):
        pos[layer1[i]] = np.array([-2, y1[i]])
    for i in range(len(layer3)):
        pos[layer3[i]] = np.array([2, y3[i]])

    epos = []
    eneg = []
    wpos = []
    wneg = []
    ecol_pos, ecol_neg = [], []

    for (u, v, d) in G.edges(data=True):
        if d["weight"] > 0:
            epos.append((u, v))
            wpos.append(d["weight"])
            if [u,v] in edge_list:
                ecol_pos.append('r')
                wpos[-1] = 1
            else:
                ecol_pos.append('grey')

        elif d["weight"] < 0:
            eneg.append((u, v))
            wneg.append(d["weight"])
            if [u, v] in edge_list:
                ecol_neg.append('r')
                wneg[-1] = -1
            else:
                ecol_neg.append('grey')


    edge_width_pos = NormalizeData(np.array(wpos))
    edge_width_neg = NormalizeData(-np.array(wneg))

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111)
    nx.draw_networkx_nodes(G, pos, node_color=col, alpha=0.5, node_size=700)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

    nx.draw_networkx_edges(G, pos, edgelist=epos, width=edge_width_pos + 0.5, edge_color=ecol_pos,
                           arrowsize=10, alpha=0.75, connectionstyle='arc3')
    nx.draw_networkx_edges(G, pos, edgelist=eneg, width=edge_width_neg + 0.5, edge_color=ecol_neg,
                           arrowstyle="-[", arrowsize=10, alpha=0.75, connectionstyle='arc3')

    patch1 = mpatches.Patch(color='plum', label=path1)
    patch2 = mpatches.Patch(color='mediumseagreen', label=path2)
    patch3 = mpatches.Patch(color='skyblue', label='shared')
    plt.legend(handles=[patch1, patch2, patch3])

    if title:
        plt.title(title)

    fig.tight_layout()
    plt.axis("off")
    plt.savefig(filepath + 'regulation_' + path1 + '_' + path2 + '.pdf', format='pdf', dpi=300)




def umap_plot(adata, key):
    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111)

    sc.pl.umap(adata, color=key, ax=ax, show=False, legend_fontsize=15)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig('umap_time.pdf', format='pdf', dpi=300)


def centrality_plot(grn_name, path1, path2, time_point, time_point2, filepath=''):

    df0 = pd.read_csv(filepath + grn_name + '_' + time_point + '_gene_statistics.csv', index_col=0)
    genes_df = pd.read_csv(filepath + grn_name + '_' + time_point + '_gene_annotation.csv', index_col='gene', header=0)
    genes_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    genes_df = genes_df.transpose()
    genes, btc = list(df0.index), np.asarray(df0.mean(axis=1))

    col = []
    for g in genes:
        if list(genes_df[g])[0] == path1:
            col.append('plum')
        elif list(genes_df[g])[0] == path2:
            col.append('mediumseagreen')
        else:
            col.append('skyblue')

    fig = plt.figure(figsize=(4,4))
    ax = plt.subplot(111)

    plt.barh( np.arange(0, btc.size, 1), btc, color=col )
    plt.yticks(np.arange(0, btc.size, 1), genes)
    plt.xlabel('Gene centrality')

    plt.tight_layout()
    plt.savefig(filepath + 'centrality_' + time_point + '.pdf', format='pdf', dpi=300)

    df3 = pd.read_csv(filepath + grn_name + '_' + time_point2 + '_gene_statistics.csv', index_col=0)

    genes0d, genes3d = list(df0.index), list(df3.index)
    inters = [g for g in genes0d if g in genes3d]

    df0 = df0.transpose()
    df3 = df3.transpose()

    df0_sel, df3_sel = df0[inters], df3[inters]

    btc0 = np.asarray(df0_sel.mean(axis=0))
    btc3 = np.asarray(df3_sel.mean(axis=0))

    btc_fc = btc3-btc0

    fig = plt.figure(figsize=(4, 4))
    ax = plt.subplot(111)
    plt.barh(np.arange(0, btc_fc.size, 1), btc_fc)
    plt.yticks(np.arange(0, btc_fc.size, 1), inters)
    plt.xlabel('Change in centrality')
    plt.tight_layout()
    plt.savefig(filepath + 'centrality_change.pdf', format='pdf', dpi=300)
