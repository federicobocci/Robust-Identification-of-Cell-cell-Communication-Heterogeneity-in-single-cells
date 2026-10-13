# Plotting

All `scrich.pl` functions follow the scanpy conventions:

- Plots are **shown** by default and **never saved automatically**. Set `ri.settings.autoshow = False` to
  stop showing plots by default (e.g. in scripts or on a cluster).
- With `show=False` the function returns the axes (a list of axes for multi-panel plots; the plotly figure
  for alluvial plots), so the plot can be modified or saved:

  ```python
  ax = ri.pl.single_pathway_heterogeneity(adata, "IGF", "celltype", show=False)
  ax.set_title("IGF")
  ax.figure.savefig("IGF_heterogeneity.pdf", bbox_inches="tight")
  ```

- `return_fig=True` returns the figure.
- Single-panel plots accept `ax=` to draw into an existing figure:

  ```python
  import matplotlib.pyplot as plt

  fig, axs = plt.subplots(1, 2, figsize=(8, 3))
  ri.pl.plot_mode_gap(adata, "IGF", ax=axs[0], show=False)
  ri.pl.plot_mode_gap(adata, "VEGF", ax=axs[1], show=False)
  fig.savefig("mode_gaps.pdf")
  ```

- Embedding plots take `basis=` (default `'umap'`, i.e. `adata.obsm['X_umap']`).

Alluvial plots use plotly: save them with `fig.write_html("plot.html")` or, with kaleido installed,
`fig.write_image("plot.pdf")`.

## Arguments of scRICH 0.1

The arguments `savefig`, `showfig`, `figname`, `format` and `dpi` still work but raise a
`FutureWarning` and will be removed in a future version:

| scRICH 0.1 | Now |
|---|---|
| `showfig=True` | default (or `show=True`) |
| `savefig=True, figname="x.pdf"` | `ax = ...(show=False); ax.figure.savefig("x.pdf")` |
| `savefig=False, showfig=True` | default |
