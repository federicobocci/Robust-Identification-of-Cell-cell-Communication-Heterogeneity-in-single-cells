"""Run a tutorial-1-style scRICH pipeline and pickle the key outputs, for old-vs-new comparison.

Usage
-----
    python run_pipeline.py OUT.pkl DATA.h5ad [KEY] [SPECIES] [N_PATHWAYS]

``DATA.h5ad`` must contain Ms/Mu layers and a neighbour graph (e.g. after ``scv.pp.moments``).
For the *old* code run it from the old repository root (it reads ``pathway_database/`` relative
to the working directory) with that folder on ``PYTHONPATH``. Then compare two runs with
``compare.py``.
"""
import sys, pickle, warnings, numpy as np, matplotlib
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import anndata as ad, scrich as ri
import matplotlib.pyplot as plt
out = sys.argv[1]
adata = ad.read_h5ad(sys.argv[2])
KEY = sys.argv[3] if len(sys.argv) > 3 else "celltype"
SPECIES = sys.argv[4] if len(sys.argv) > 4 else "mouse"
NPATH = int(sys.argv[5]) if len(sys.argv) > 5 else 6
HUMAN = SPECIES == "human"
ri.tl.pathways_overview(adata, human=HUMAN)
ri.tl.all_path_sim(adata, KEY, method="expression", pathways_to_use=NPATH, target=False, human=HUMAN,
                   neighbor_average=True, print_info=False)
np.random.seed(0); ri.tl.permutation_testing(adata, key=KEY)
p0 = adata.uns["selected_pathways"][0]
ri.tl.import_database(adata, [p0], species=SPECIES, unspliced=True)
ri.tl.compute_ccc_matrix(adata, p0, key=KEY, include_target=True, conversion=True, model="mass_action", min_cells=10)
ri.tl.pairwise_MI(adata)
ri.tl.find_sign_patterns(adata, res=0.5)
res = {"selected": list(adata.uns["selected_pathways"]), "overview": adata.uns["paths_overview"],
       "pathways": {k: {kk: sorted(map(str, vv)) if kk in ("receptors","ligands") else vv for kk, vv in v.items()} for k, v in adata.uns["pathways"].items()},
       "symnmf": {k: (np.asarray(v["eigenvalues"]), int(v["optimal"])) for k, v in adata.uns["SymNMF"].items()},
       "fraction": dict(adata.uns["fraction_mat"]), "perm": adata.uns["permutation_test"],
       "TF": {k: sorted(v) for k, v in adata.uns["TF"].items()}, "ccc": adata.uns["ccc_mat"][p0],
       "MI": adata.uns["CCC_similarity_map"],
       "obs": {c: np.asarray(adata.obs[c]) for c in adata.obs.columns}}
# plotting smoke tests
plots = {}
calls = {
 "pathways_overview": lambda: ri.pl.pathways_overview(adata, top=5, savefig=False, showfig=False),
 "plot_mode_gap": lambda: ri.pl.plot_mode_gap(adata, p0, savefig=False, showfig=False),
 "mode_violin": lambda: ri.pl.mode_violin(adata, p0 + "_modes", savefig=False, showfig=False),
 "scatter2D": lambda: ri.pl.scatter2D(adata, p0 + "_modes", savefig=False, showfig=False),
 "single_pathway_heterogeneity": lambda: ri.pl.single_pathway_heterogeneity(adata, p0, KEY, savefig=False, showfig=False),
 "heatmap_one_pathway": lambda: ri.pl.heatmap_one_pathway(adata, p0, KEY, savefig=False, showfig=False),
 "pathway_heterogeneity_summary": lambda: ri.pl.pathway_heterogeneity_summary(adata, key=KEY, savefig=False, showfig=False),
 "state_heterogeneity_summary": lambda: ri.pl.state_heterogeneity_summary(adata, KEY, savefig=False, showfig=False),
 "redundancy": lambda: ri.pl.redundancy(adata, savefig=False, showfig=False),
 "pathway_hierarchy": lambda: ri.pl.pathway_hierarchy(adata, savefig=False, showfig=False),
 "pattern_summary": lambda: ri.pl.pattern_summary(adata, KEY, panel_per_row=5, savefig=False, showfig=False),
 "pattern_mouse": lambda: ri.pl.pattern(adata, 0, key=KEY, human=HUMAN, ntop=5, savefig=False, showfig=False),
}
for name, f in calls.items():
    try: f(); plots[name] = "ok"
    except Exception as e: plots[name] = f"{type(e).__name__}: {e}"
    plt.close("all")
res["plots"] = plots
pickle.dump(res, open(out, "wb"))
print("selected:", res["selected"]); print("plots:", plots)
