
import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _no_autoshow():
    """Plots are not shown in tests (non-interactive backend); individual tests may override."""
    import matplotlib.pyplot as plt

    import scrich as ri

    ri.settings.autoshow = False
    yield
    ri.settings.autoshow = True
    plt.close("all")


@pytest.fixture(scope="session")
def exfinder_dir():
    """Folder with the external database files, or skip the test.

    The files are downloaded from Zenodo if needed (or read from ``SCRICH_DATA_DIR``).
    """
    import scrich as ri

    try:
        ri.db.fetch_all()
    except Exception as err:  # no internet, Zenodo down, ...
        pytest.skip(f"exFinder database files not available: {err}")
    return ri.db.data_dir()


@pytest.fixture
def adata_mouse():
    """Small synthetic mouse dataset with mouse ligand/receptor genes, Ms/Mu layers and a kNN graph."""
    import scrich as ri

    rng = np.random.default_rng(0)
    lig = ri.db.load_pathway_genes("ligand", "mouse")
    rec = ri.db.load_pathway_genes("receptor", "mouse")
    genes = set(lig["ligand"])
    for g in rec["receptor"]:
        genes.update(g.split("_"))
    genes = sorted(genes)
    n_types, per = 3, 60
    ncell, ngene = n_types * per, len(genes)
    types = np.repeat([f"T{i}" for i in range(n_types)], per)
    mu = rng.gamma(0.6, 1.0, ngene) + rng.gamma(0.8, 1.5, (n_types, ngene))[np.repeat(np.arange(n_types), per)] * (
        rng.random((ncell, ngene)) < 0.3
    )
    X = np.log1p(rng.poisson(mu)).astype(np.float32)
    emb = rng.normal(size=(ncell, 5)) + np.eye(n_types, 5)[np.repeat(np.arange(n_types), per)] * 4
    conn = kneighbors_graph(emb, 15, mode="connectivity").tocsr()
    Ms = np.asarray(conn @ X / 15, dtype=np.float32)
    a = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame({"celltype": types}, index=[f"c{i}" for i in range(ncell)]),
        var=pd.DataFrame(index=genes),
    )
    a.layers["Ms"], a.layers["Mu"] = Ms, 0.5 * Ms
    a.obsp["distances"] = kneighbors_graph(emb, 15, mode="distance").tocsr()
    a.obsp["connectivities"] = conn
    return a
