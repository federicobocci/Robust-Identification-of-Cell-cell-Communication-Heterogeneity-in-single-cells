# scRICH

**Robust Identification of Cell-cell Communication Heterogeneity in single cells**

scRICH identifies how cells of the same type differ in the way they send and receive signals. For each
cell-cell communication (CCC) pathway it groups cells into *communication modes* (e.g. senders,
receivers, inactive cells) from the expression of the pathway's ligands, receptors and, optionally,
downstream genes; scores how heterogeneous every cell type is; builds CCC matrices between cell states;
and finds *signaling patterns* shared across pathways.

```{image} _static/scrich_schematic.png
:alt: scRICH schematic
:width: 100%
```

scRICH works on {class}`anndata.AnnData` objects and follows the [scanpy](https://scanpy.readthedocs.io)
conventions: analysis tools are in `scrich.tl`, plots in `scrich.pl`.

```python
import scrich as ri

ri.tl.pathways_overview(adata, species="mouse")
ri.tl.all_path_sim(adata, "celltype", pathways_to_use=10, species="mouse")
ri.tl.permutation_testing(adata, key="celltype")
ri.pl.pathway_heterogeneity_summary(adata, key="celltype")
```

::::{grid} 2
:gutter: 2

:::{grid-item-card} Installation
:link: installation
:link-type: doc
Install scRICH with pip and the optional extras.
:::

:::{grid-item-card} Getting started
:link: getting-started
:link-type: doc
The analysis steps and where their results are stored.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Worked examples on scRNA-seq and spatial data.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc
All functions of `scrich.tl`, `scrich.pl` and `scrich.db`.
:::
::::

## Citation

If you use scRICH, please cite:

> Bocci F. *et al.* Robust identification of cell-cell communication heterogeneity in single cells.
> *Cell Systems* (2026). <https://www.cell.com/cell-systems/fulltext/S2405-4712(26)00185-7>

The code version used in the paper is archived on Zenodo:
[10.5281/zenodo.20394264](https://doi.org/10.5281/zenodo.20394264).

```{toctree}
:hidden:
:maxdepth: 2

installation
getting-started
databases
plotting
tutorials/index
api/index
changelog
```
