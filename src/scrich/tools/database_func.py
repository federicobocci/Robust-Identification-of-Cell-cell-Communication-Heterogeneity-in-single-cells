"""Helpers to parse the bundled pathway -> ligand/receptor tables."""


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