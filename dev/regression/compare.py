"""Compare two pickles written by run_pipeline.py.  Usage: python compare.py OLD.pkl NEW.pkl"""
import pickle, sys, numpy as np
a = pickle.load(open(sys.argv[1], "rb")); b = pickle.load(open(sys.argv[2], "rb"))
def cmp(x, y, path):
    if isinstance(x, dict):
        assert set(map(str, x)) == set(map(str, y)), f"{path}: keys differ"
        for k in x: cmp(x[k], y[k], f"{path}.{k}")
    elif isinstance(x, (list, tuple)) and not (len(x) and isinstance(x[0], (int, float, np.number))):
        assert len(x) == len(y), f"{path}: length {len(x)} vs {len(y)}"
        for i, (u, v) in enumerate(zip(x, y)): cmp(u, v, f"{path}[{i}]")
    else:
        u, v = np.asarray(x), np.asarray(y)
        if u.dtype.kind in "fiu" and v.dtype.kind in "fiu":
            ok = np.allclose(u, v, rtol=1e-5, atol=1e-8)
            dev = np.nanmax(np.abs(u - v)) if u.size else 0
            if not ok: print(f"DIFF {path}: max abs diff {dev:.3g}"); diffs.append(path)
        else:
            if not np.array_equal(u.astype(str), v.astype(str)): print(f"DIFF {path}"); diffs.append(path)
diffs = []
for k in a:
    if k == "plots": continue
    cmp(a[k], b[k], k)
print("numeric/label outputs compared:", [k for k in a if k != "plots"])
print("differences:", diffs if diffs else "none")
print("plots old:", {k: v for k, v in a["plots"].items() if v != "ok"} or "all ok")
print("plots new:", {k: v for k, v in b["plots"].items() if v != "ok"} or "all ok")
# Mode labels can be permuted between library versions; compare partitions with the adjusted Rand index.
from sklearn.metrics import adjusted_rand_score
for p in a["selected"]:
    if p + "_modes" in b["obs"]:
        print(f"{p}: optimal modes {a['symnmf'][p][1]} vs {b['symnmf'][p][1]}, "
              f"ARI of mode labels = {adjusted_rand_score(a['obs'][p + '_modes'], b['obs'][p + '_modes']):.3f}")
