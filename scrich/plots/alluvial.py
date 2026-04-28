'''
functions for alluvial plots
'''
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def compute_weight(list1, list2, v1, v2):
    w = 0
    for l1, l2 in zip(list1, list2):
        if l1==v1 and l2==v2:
            w = w + 1
    return w

def set_color(map, features):
    # convert colors from matplotlib colormap to RGB
    colors = list(map.colors)[0:len(features)]
    colorlist = []
    for i in range(len(features)):
        rgb_scaled = tuple([255 * c for c in colors[i]])
        colorlist.append('rgb' + str(rgb_scaled))
    return colorlist


def twostate_sankey(v1, v2, lab1, lab2, col1, col2, name1, name2, alpha=0.5,
                     pad=5, thickness=40, linecolor='black', linewidth=1, width=400, height=600, font_size=15,
                    savefig='True', figname='compare_paths.pdf', format='pdf', showfig=True, strip_text=False):

    label=lab1+lab2

    source, target, value, edge_color = [], [], [], []
    for n1 in lab1:
        for n2 in lab2:
            source.append(label.index(n1))
            target.append(label.index(n2))
            value.append(compute_weight(np.asarray(v1), np.asarray(v2), n1, n2))
            edge_color.append('rgba' + col2[lab2.index(n2)][3:-1] + ', ' + str(alpha) + ')')

    if strip_text:
        label = None

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=pad,  # separation between nodes
            thickness=thickness,
            line=dict(color=linecolor, width=linewidth),
            label=label,
            color=col1+col2
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=edge_color
        ))])
    if not strip_text:
        fig.add_annotation(text=name1, xref="paper", yref="paper", x=0., y=1.1, showarrow=False, align='center')
        fig.add_annotation(text=name2, xref="paper", yref="paper", x=1.05, y=1.1, showarrow=False, align='center')

    fig.update_layout(autosize=False, width=width, height=height, font_size=font_size)

    if savefig:
        fig.write_image(figname, format=format)
    if showfig:
        fig.show()




def alluvial_onepath(adata, pathway, key, map1=plt.cm.Set3, map2=plt.cm.Set2, alpha=0.5,
                     pad=5, thickness=40, linecolor='black', linewidth=1, width=400, height=600, font_size=15,
                     savefig='True', figname='alluvial.pdf', format='pdf', showfig=True, strip_text=False):

    memb = list(adata.obs[key])
    p = list(adata.obs[pathway+'_modes'])

    states, modes = sorted(list(set(memb))),  sorted(list(set(p)))

    # set colors for source and targets
    source_color = set_color(map1, states)
    target_color = set_color(map2, modes)

    twostate_sankey(memb, p, states, modes, source_color, target_color, 'Cell Type', pathway+' Mode', alpha=alpha,
                    pad=pad, thickness=thickness, linecolor=linecolor, linewidth=linewidth, width=width,
                    height=height, font_size=font_size,
                    savefig=savefig, figname=figname, format=format, showfig=showfig, strip_text=strip_text)


def alluvial_twopath(adata, pathway1, pathway2, include='all', key='clusters', map1=plt.cm.Set3, map2=plt.cm.Set2, alpha=0.5,
                     pad=5, thickness=40, linecolor='black', linewidth=1, width=400, height=600, font_size=15,
                     savefig='True', figname='compare_paths.pdf', format='pdf', showfig=True, strip_text=False):
    if include!='all':
        adata = adata[adata.obs[key].isin(include)].copy()
    p1 = list(adata.obs[pathway1 + '_modes'])
    p2 = list(adata.obs[pathway2 + '_modes'])
    modes1, modes2 = sorted(list(set(p1))), sorted(list(set(p2)))

    # if signaling modes were not renamed, add the pathway name to distinguish
    if(modes1[0]==modes2[0]):
        p1 = [str(p) + '-' + pathway1 for p in p1]
        p2 = [str(p) + '-' + pathway2 for p in p2]
        # p1 = [str(p) for p in p1]
        # p2 = [str(p) for p in p2]
        modes1, modes2 = sorted(list(set(p1))), sorted(list(set(p2)))

    # set colors for source and targets
    source_color = set_color(map1, modes1)
    target_color = set_color(map2, modes2)

    twostate_sankey(p1, p2, modes1, modes2, source_color, target_color, pathway1, pathway2, alpha=alpha,
                    pad=pad, thickness=thickness, linecolor=linecolor, linewidth=linewidth, width=width,
                    height=height, font_size=font_size,
                    savefig=savefig, figname=figname, format=format, showfig=showfig, strip_text=strip_text)


def alluvial_pattern(adata, key, map1=plt.cm.Set3, map2=plt.cm.tab20, alpha=0.5,
                     pad=5, thickness=40, linecolor='black', linewidth=1, width=400, height=600, font_size=15,
                     savefig='True', figname='patterns.pdf', format='pdf', showfig=True, strip_text=False):
    clusters, patterns = list(adata.obs[key]), np.asarray(adata.obs['sign_pattern'], dtype='int')

    states, modes = sorted(list(set(clusters))), sorted(list(set(patterns)))

    source_color = set_color(map1, states)
    target_color = set_color(map2, modes)

    twostate_sankey(clusters, patterns, states, modes, source_color, target_color, 'Cell Type', 'Pattern', alpha=alpha,
                    pad=pad, thickness=thickness, linecolor=linecolor, linewidth=linewidth, width=width,
                    height=height, font_size=font_size,
                    savefig=savefig, figname=figname, format=format, showfig=showfig, strip_text=strip_text)

