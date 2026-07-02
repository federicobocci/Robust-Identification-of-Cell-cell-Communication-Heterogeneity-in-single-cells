import pandas as pd

def loadDB(path):
    '''Load the CellChat ligand/receptor database

    Parameters
    ----------
    path: path to database file

    Returns
    -------
    DB: dataframe with CellChat ligand/receptor database

    '''
    DB = pd.read_csv(path)
    return DB


#### sketchy definition ####
def get_list(df, pathway):
    geneset = list(df[ df['pathway']==pathway ].iloc[:,1])
    # print(geneset)
    new_set = []
    for g in geneset:
        if '_' in g:
            i = g.index('_')
            a, b = g[0:i], g[i + 1:]
            new_set = new_set + [a, b]
            # print( g, g[0:i], g[i+1:] )
        else:
            new_set.append(g)
    return new_set