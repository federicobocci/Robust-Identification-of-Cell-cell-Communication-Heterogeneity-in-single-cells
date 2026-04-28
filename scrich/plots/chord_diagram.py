'''
Functions for the rpy2 plotting interface
'''
import numpy as np
import matplotlib as mpl
import os

import rpy2
import rpy2.robjects.packages as rpackages
from rpy2 import robjects

def initialize_R_plotting(package_list=['circlize'], verbose=False):
    '''
    Initialize the rpy2 plotting interface by checking that all necessary R libraries are installed

    Parameters
    ----------
    package_list: list of R packages
    verbose: if True, print out information about rpy2 version

    Returns
    -------
    None

    '''
    if verbose:
        print('importing R version ' + str(rpy2.__version__))
    utils = rpackages.importr('utils')

    for p in package_list:
        if not rpackages.isinstalled(p):
            utils.install_packages(p)

'''
Things that still need attention:
- extract CCC matrix, state and mode names form anndata
'''
def chord_diagram(adata, pathway, cmap='tab20', key='clusters', thr = None, include=None, mode_gap=1, clst_gap=10,
                  label_sep='-', transparency=0.5, directional=0, width=4, height=4, res=500, figname='Chord_Diagram'):
    mat = adata.uns['ccc_mat'][pathway]['mat']
    states = sorted(list(set(adata.obs[key])))
    modes = sorted(set(list(adata.obs[pathway + '_modes'])))

    # states and modes should be ordered in the same way as in the CCC matrix calculation
    nodes = adata.uns['ccc_mat'][pathway]['states']
    # nodes = []
    # for s in states:
    #     for m in modes:
    #         nodes.append(str(s) + label_sep + str(m))


    # extract colors based on cell states
    # options for colors: plot links of modes from same state with same color? Or plot a second line of tracks to identify modes from same cell state?
    cmap = mpl.color_sequences[cmap]
    colorlist = []
    for i in range(len(states)):
        for j in range(len(modes)):
            colorlist.append( mpl.colors.rgb2hex(cmap[i]))
    # colorlist = [ [mpl.colors.rgb2hex(cmap[i]) for j in range(len(modes))] for i in range(len(states)) ]

    # 1) process the CCC matrix
    # i) set small elements to zero
    # currently, the threshold is calibrated assuming that the CC matrix elements are normalized in [0,1]
    if thr:
        mat[mat<thr] = 0
    # ii) use only a subset of states passed by the user
    # if include==None, all states are considered, otherwise the state-mode indices are used to filter rows out
    if include:
        index = [] # indices to keep in the nodes list and CC matrix
        for i in range(len(states)):
            if states[i] in include:
                for j in range(len(modes)):
                    index.append(i*len(modes)+j)
        # index = [item for sublist in index for item in sublist]
        nodes = [nodes[i] for i in index]
        mat = mat[index]
        mat = mat[:, index]
        colorlist = [colorlist[i] for i in index]

    # define the gaps for chord plotting
    nmodes, nclst = len(modes), len(include) if include else len(states)
    gap = [[mode_gap for i in range(nmodes-1)] + [clst_gap] for j in range(nclst)]
    flat_gap = [item for sublist in gap for item in sublist]

    # iii) automatically remove empty rows/columns
    # if a node is isolated, the gap.after parameter in chordDiagram will lead to an error
    # it is convenient to check this after having defined the flat_gap list
    keep = []
    for i in range(len(nodes)):
        if np.any(mat[i]) or np.any(mat[:,i]):
            keep.append(i)
    if len(keep)<len(nodes):
        nodes = [nodes[i] for i in keep]
        mat = mat[keep]
        mat = mat[:, keep]
        colorlist = [colorlist[i] for i in keep]
        flat_gap = [flat_gap[i] for i in keep]

    flat_gap = [1 for c in colorlist]

    # for i in range(len(nodes)):
    #     for j in range(len(nodes)):
    #         print(nodes[i], nodes[j], mat[i][j])

    # iv) convert to list of lists before passing to r function
    mat = [mat[i].tolist() for i in range(len(nodes))]

    # call r script for plotting
    r = robjects.r
    r.source(os.path.dirname(__file__) + '/chord_diagram.R')

    print(nodes)

    # nodes = ['BAS-I', 'BAS-II', 'BAS-III', 'BAS-IV-Rec', 'GRN-1-Sen/Rec', 'SPN-1-Rec', 'SPN-3-Rec', 'SPN-3-Sen/Rec']
    # r.test(nodes)

    r.generate_ChordDiagram(nodes, mat, flat_gap, colorlist, transparency, directional, width, height, res, figname)


