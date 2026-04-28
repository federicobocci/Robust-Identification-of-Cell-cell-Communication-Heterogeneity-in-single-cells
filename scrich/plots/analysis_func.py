import numpy as np
from sklearn.metrics import mutual_info_score

def twopath_mat(adata, path1, path2):
    lab1 = np.asarray(adata.obs[path1 + '_modes'], dtype='int')
    lab2 = np.asarray(adata.obs[path2 + '_modes'], dtype='int')

    n1, n2 = np.amax(lab1)+1, np.max(lab2)+1
    M, M0 = np.zeros((n1, n2)), np.zeros((n1, n2))

    for i in range(n1):
        lab2_sel = lab2[lab1==i]
        for j in range(n2):
            M[i][j] = lab2_sel[lab2_sel==j].size
            p1 = float(lab1[lab1==i].size)/lab1.size
            p2 = float(lab2[lab2==j].size)/lab1.size
            M0[i][j] = p1*p2
    M = M/lab1.size

    sim = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            if M[i][j]>M0[i][j]:
                sim[i][j] = np.abs(M[i][j]-M0[i][j])/( 1-M0[i][j] )
            else:
                sim[i][j] = np.abs(M[i][j] - M0[i][j]) /M0[i][j]

    # sim = np.sum(np.abs(M-M0))
    print(M)
    print(np.sum(M))
    print()
    print(M0)
    print(np.sum(M0))
    print()
    print(sim)
    print(np.sum(sim))
    # return sim

def onepath_entropy(p):
    s = 0
    for i in range(len(p)):
        s = s - p[i]*np.log2(p[i])
    return s

def mutual_info(m, p1, p2, eps=10**(-5)):
    mi = 0.
    for i in range(len(p1)):
        for j in range(len(p2)):
            if m[i][j]>eps:
                mi = mi + m[i][j] * np.log2(m[i][j] / (p1[i] * p2[j]))
    return mi

def twopath_mutual_info(adata, path1, path2):
    lab1 = np.asarray(adata.obs[path1 + '_modes'], dtype='int')
    lab2 = np.asarray(adata.obs[path2 + '_modes'], dtype='int')

    n1, n2 = len(set(lab1)), len(set(lab2))

    # marginal probabilities:
    p1 = [ lab1[lab1==i].size/lab1.size for i in range(n1) ]
    p2 = [lab2[lab2 == i].size/lab2.size for i in range(n2)]


    M = np.zeros((n1, n2))

    for i in range(n1):
        lab2_sel = lab2[lab1 == i]
        for j in range(n2):
            M[i][j] = lab2_sel[lab2_sel == j].size
    M = M / lab1.size


    mi = mutual_info(M, p1, p2)
    mi_skl = mutual_info_score(lab1, lab2)
    # print(mutual_info_score(lab1, lab2), mutual_info_score(lab2, lab1))
    # h1, h2 = onepath_entropy(p1), onepath_entropy(p2)
    return mi_skl #mi/(h1+h2)


def pairwise_MI(adata):
    pathways = list(adata.uns['pathways'].keys())

    sim_mat = np.zeros((len(pathways), len(pathways)))

    for i in range(len(pathways)):
        for j in range(len(pathways)):
            sim_mat[i][j] = twopath_mutual_info(adata, pathways[i], pathways[j])
    return sim_mat


