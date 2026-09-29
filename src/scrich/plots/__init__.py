"""Plotting functions of scRICH, exposed to users as ``scrich.pl``.

Modules
-------
pathways       CCC pathways detected in the data
modes          communication modes of a pathway (spectral gap, embeddings, violin plots)
heterogeneity  cell-type heterogeneity of communication modes
patterns       signaling patterns and similarity between pathways
ccc            cell-cell communication matrix (chord diagram, coarse-grained map)
flows          alluvial (Sankey) plots, requires plotly
velocity       communication modes/patterns with RNA velocity and pseudotime
"""

from .ccc import chord_diagram, coarse_grained_map
from .flows import alluvial_onepath, alluvial_pattern, alluvial_twopath
from .heterogeneity import (
    heatmap_one_pathway,
    heterogeneity_heatmap,
    mode_composition,
    pathway_heterogeneity_summary,
    single_pathway_heterogeneity,
    state_heterogeneity_summary,
)
from .modes import feature_plot, mode_violin, pathway_umap, plot_mode_gap, scatter2D
from .pathways import pathways_overview
from .patterns import pathway_hierarchy, pattern, pattern_summary, redundancy
from .velocity import (
    expr_map,
    pattern_plot,
    pattern_prob_plot,
    plot_maps,
    sign_prob_plot,
    state_dist_pst,
    top_players_map,
)

__all__ = [
    "chord_diagram",
    "coarse_grained_map",
    "alluvial_onepath",
    "alluvial_pattern",
    "alluvial_twopath",
    "heatmap_one_pathway",
    "heterogeneity_heatmap",
    "mode_composition",
    "pathway_heterogeneity_summary",
    "single_pathway_heterogeneity",
    "state_heterogeneity_summary",
    "feature_plot",
    "mode_violin",
    "pathway_umap",
    "plot_mode_gap",
    "scatter2D",
    "pathways_overview",
    "pathway_hierarchy",
    "pattern",
    "pattern_summary",
    "redundancy",
    "expr_map",
    "pattern_plot",
    "pattern_prob_plot",
    "plot_maps",
    "sign_prob_plot",
    "state_dist_pst",
    "top_players_map",
]
