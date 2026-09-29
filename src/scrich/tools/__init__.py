"""Analysis tools of scRICH, exposed to users as ``scrich.tl``.

Modules
-------
pathways       select CCC pathways and find their ligands/receptors in the data
modes          similarity matrix, optimal number of modes and consensus clustering per pathway
heterogeneity  cell-type heterogeneity of communication modes and permutation testing
patterns       signaling patterns across pathways and mutual information between pathways
ccc            cell-cell communication matrix between cell states and modes
targets        downstream genes (transcription factors) of pathways from the exFinder database
annotations    renaming of modes and cell annotations
"""

from .annotations import rename_cells, rename_modes
from .ccc import compute_ccc_matrix
from .heterogeneity import permutation_testing
from .modes import all_path_sim, consensus_clustering, single_path_sim
from .pathways import find_genes, pathways_overview, select_pathways
from .patterns import find_sign_patterns, pairwise_MI
from .targets import import_database

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
