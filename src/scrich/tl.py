"""Analysis tools (``import scrich as ri; ri.tl.<function>``)."""

from .tools import (
    all_path_sim,
    compute_ccc_matrix,
    consensus_clustering,
    find_genes,
    find_sign_patterns,
    import_database,
    pairwise_MI,
    pathways_overview,
    permutation_testing,
    rename_cells,
    rename_modes,
    select_pathways,
    single_path_sim,
)

__all__ = [
    "pathways_overview",
    "select_pathways",
    "find_genes",
    "all_path_sim",
    "single_path_sim",
    "consensus_clustering",
    "permutation_testing",
    "import_database",
    "compute_ccc_matrix",
    "pairwise_MI",
    "find_sign_patterns",
    "rename_modes",
    "rename_cells",
]
