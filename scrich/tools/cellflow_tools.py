import numpy as np
import pandas as pd
import splicejac as sp
import networkx as nx

# extract and return the lists of lig, rec, TF, targets
def geneset(adata, path):
    return adata.uns['pathways'][path]['ligands'] + adata.uns['pathways'][path]['receptors'] + \
           adata.uns['pathways'][path]['downstream']

# quantify overlapping genes between two paths and save them to adata.uns['overlap']
def pathway_overlap(adata, path1, path2, verbose=False):
    assert path1 in adata.uns['pathways'].keys(), "pathway information not available"
    assert path2 in adata.uns['pathways'].keys(), "pathway information not available"

    if 'overlap' not in adata.uns.keys():
        adata.uns['overlap'] = {}

    path1_geneset = geneset(adata, path1)
    path2_geneset = geneset(adata, path2)
    overlap = [p for p in path1_geneset if p in path2_geneset]
    adata.uns['overlap'][path1 + '_' + path2] = overlap

    if verbose:
        print(str(len(overlap)) + ' common genes were identified between ' + path1 + ' and ' + path2 + ':')
        print(overlap)

# exclude genes that are not present in the dataset
def pathway_filter(adata, path, verbose=False):
    genes = list(adata.var_names)
    for k in ['ligands', 'receptors', 'downstream']:
        sel_genes = [g for g in adata.uns['pathways'][path][k] if g in genes]
        if verbose:
            excluded = [g for g in adata.uns['pathways'][path][k] if g not in sel_genes]
            if len(excluded)>0:
                print('The following genes from the ' + path + ' ' + k + ' gene set were not detected in the dataset:')
                print(excluded)
        adata.uns['pathways'][path][k] = sel_genes

# select the top genes for each pathway, including at least one receptor and one ligand
def select_geneset(adata, path, n=20):
    path_geneset = geneset(adata, path)
    expr = np.mean(adata[:, path_geneset].X.toarray(), axis=0)

    # 1) Select the top n genes in the pathway
    inds = np.flip(np.argsort(expr))
    selected_geneset = [path_geneset[i] for i in inds[0:n]]

    # 2) Check that at least one ligand and one receptor are included
    rec, lig = adata.uns['pathways'][path]['receptors'], adata.uns['pathways'][path]['ligands']

    sel_rec = [s for s in selected_geneset if s in rec]
    if len(sel_rec)==0:
        if len(rec)==1:
            selected_geneset.append(rec[0])                 # if only one receptor is present, just add it
        else:                                               # otherwise, add the top receptor with higher expression
            expr_rec = np.mean(adata[:, rec].X.toarray(), axis=0)
            inds_rec = np.flip(np.argsort(expr_rec))
            selected_geneset.append(rec[inds_rec[0]])

    # same process for ligand
    sel_lig = [s for s in selected_geneset if s in lig]
    if len(sel_lig)==0:
        if len(lig)==1:
            selected_geneset.append(lig[0])
        else:
            expr_lig = np.mean(adata[:, lig].X.toarray(), axis=0)
            inds_lig = np.flip(np.argsort(expr_lig))
            selected_geneset.append(lig[inds_lig[0]])

    return selected_geneset

# run the grn inference for the two pathways
def hierarchical_grn(adata, path1, path2, n=20, key=None, cells=None, select_genes='alldata', verbose=False, filepath='', export_data=None):
    # key = column of adata.obs to use to select cells for inference
    # select_genes=='alldata': top genes selected based on expression in all the dataset; 'specific': top genes only in the cells selected for inference
    assert select_genes=='alldata' or select_genes=='specific', "Please choose between select_genes=='alldata' or select_genes=='specific'"
    # 1) check for genes that are not detected in the dataset
    pathway_filter(adata, path1, verbose=verbose)
    pathway_filter(adata, path2, verbose=verbose)
    # --- add after this step: check that at least one L, one R, one T exists for the pathway ---
    # 2) Check for overlapping genes
    pathway_overlap(adata, path1, path2, verbose=verbose)
    # 3) Slice the data
    if key:
        adata_cff = adata[adata.obs[key]==cells].copy()
    # 4) Select the geneset for inference
    if select_genes=='alldata':
        path1_geneset = select_geneset(adata, path1, n=n)
        path2_geneset = select_geneset(adata, path2, n=n)
    else:
        path1_geneset = select_geneset(adata_cff, path1)
        path2_geneset = select_geneset(adata_cff, path2)

    selected_genes = list(set(path1_geneset+path2_geneset))

    adata_sel = adata_cff[:, selected_genes]

    sp.tl.estimate_jacobian(adata_sel, n_top_genes=len(list(adata_sel.var_names)), filter_and_norm=False)
    sp.tl.grn_statistics(adata_sel)

    # --- export data ---

    # save GRN
    if export_data:
        n = len(list(adata_sel.var_names))
        J = adata_sel.uns['average_jac'][cells][0][0:n, n:].copy()
        np.savetxt(filepath + path1 + '_' + path2 + '_' + cells + '_grn.txt', J)

        # save betweenness centrality
        betwenness_cent, incoming, outgoing, total_sign = adata_sel.uns['GRN_statistics']
        bc = betwenness_cent[cells]
        bc.to_csv(filepath + path1 + '_' + path2 + '_' + cells + '_gene_statistics.csv')

        # gene annotations (path1, shared, path2)
        genes = list(bc.index)
        path1_geneset = geneset(adata, path1)
        path2_geneset = geneset(adata, path2)
        path_annotation = []
        for g in genes:
            if g in path1_geneset and not (g in path2_geneset):
                path_annotation.append(path1)
            elif not (g in path1_geneset) and g in path2_geneset:
                path_annotation.append(path2)
            else:
                path_annotation.append('shared')
        gene_labels = pd.DataFrame.from_dict({'gene': genes, 'annotation': path_annotation})
        gene_labels.to_csv(filepath + path1 + '_' + path2 + '_' + cells + '_gene_annotation.csv')


def compute_max_flow(A, genes_df, adata, path1, path2, source, sink, weight_quantile=0.5):
    # step 1: load GRN and gene names from adata
    # A = np.loadtxt('/Users/federicobocci/Desktop/my_methods/cellular_flow/TGFb_BMP_analysis/TGFB_BMP_8h_grn.txt')
    # genes_df = pd.read_csv('/Users/federicobocci/Desktop/my_methods/cellular_flow/TGFb_BMP_analysis/TGFB_BMP_8h_gene_annotation.csv')
    genes, path = list(genes_df['gene']), list(genes_df['annotation'])

    q_pos = np.quantile(A[A > 0], weight_quantile)
    q_neg = np.quantile(A[A < 0], 1 - weight_quantile)
    A[(A > q_neg) & (A < q_pos)] = 0

    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    nx.relabel_nodes(G, dict(zip(range(len(genes)), genes)), copy=False)

    # step 2: load the receptor/ligand gene sets from adata and exclude if not in GRN
    if source=='receptor':
        source_set = adata.uns['pathways'][path1]['receptors']
    elif source=='ligand':
        source_set = adata.uns['pathways'][path1]['ligands']
    else:
        source_set = source
    source_set = [s for s in source_set if s in genes]
    assert len(source_set)>0, "No source gene in the GRN"

    if sink=='receptor':
        sink_set = adata.uns['pathways'][path2]['receptors']
    elif sink=='ligand':
        sink_set = adata.uns['pathways'][path2]['ligands']
    else:
        sink_set = sink
    sink_set = [s for s in sink_set if s in genes]
    assert len(sink_set)>0, "No source gene in the GRN"

    path_list, path_weight = [], []
    for s1 in source_set:
        for s2 in sink_set:
            paths = nx.all_simple_paths(G, s1, s2)
            # print(s1, s2)
            # compute weight of each path
            for p in list(paths):
                weight = 1.
                for i in range(len(p) - 1):
                    start, end = p[i], p[i + 1]
                    j, k = genes.index(start), genes.index(end)
                    link = A[j][k]
                    weight = weight * link
                if weight>0.:
                    path_list.append(p)
                    path_weight.append(weight)


    if len(path_list)==0:
        print('No path found, try decreasing the weight_quantile parameter')
        return None, None
    else:
        return path_list, path_weight
