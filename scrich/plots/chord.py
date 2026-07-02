from d3blocks import D3Blocks
import pandas as pd

def chord_diagram(adata, p, eps=0.01):
    '''Generate a chord diagram of the CCC pathway p using the d3blocks library.

    Parameters
    ----------
    adata: scRNA-seq data object
    p: pathway of interest
    eps: threshold to select interactions in the CCC matrix

    Returns
    -------
    None

    '''
    states, mat = adata.uns['ccc_mat'][p]['states'], adata.uns['ccc_mat'][p]['mat']

    # transform CCC mat to pandas dataframe
    ccc_dict = {'source': [], 'target': [], 'weight': []}
    for i in range(len(states)):
        for j in range(len(states)):
            if mat[i][j] > eps:
                ccc_dict['source'].append(states[i])
                ccc_dict['target'].append(states[j])
                ccc_dict['weight'].append(mat[i][j])
    ccc_df = pd.DataFrame.from_dict(ccc_dict)

    d3 = D3Blocks()
    d3.chord(ccc_df)
    d3.show()
