# API reference

```python
import scrich as ri
```

```{eval-rst}
.. currentmodule:: scrich

Tools: ``tl``
-------------

Pathways
~~~~~~~~

.. autosummary::
   :toctree: generated/

   tl.pathways_overview
   tl.select_pathways
   tl.find_genes

Communication modes and heterogeneity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   tl.all_path_sim
   tl.single_path_sim
   tl.consensus_clustering
   tl.permutation_testing
   tl.rename_modes
   tl.rename_cells

Signaling patterns and pathway similarity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   tl.find_sign_patterns
   tl.pairwise_MI

Cell-cell communication matrix and downstream genes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   tl.compute_ccc_matrix
   tl.import_database

Plotting: ``pl``
----------------

All plots take ``show`` and ``return_fig``; single-panel plots also take ``ax``. See :doc:`/plotting`.

Pathways and communication modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   pl.pathways_overview
   pl.plot_mode_gap
   pl.mode_violin
   pl.scatter2D
   pl.pathway_umap
   pl.feature_plot

Heterogeneity
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   pl.single_pathway_heterogeneity
   pl.pathway_heterogeneity_summary
   pl.state_heterogeneity_summary
   pl.heterogeneity_heatmap
   pl.heatmap_one_pathway
   pl.mode_composition

Signaling patterns and pathway similarity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   pl.pattern_summary
   pl.pattern
   pl.redundancy
   pl.pathway_hierarchy

Alluvial plots (plotly)
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   pl.alluvial_onepath
   pl.alluvial_twopath
   pl.alluvial_pattern

Cell-cell communication matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   pl.chord_diagram
   pl.coarse_grained_map

RNA velocity and pseudotime
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   pl.plot_maps
   pl.pattern_plot
   pl.expr_map
   pl.top_players_map
   pl.sign_prob_plot
   pl.pattern_prob_plot
   pl.state_dist_pst

Databases: ``db``
-----------------

See :doc:`/databases` for how the database files are stored and downloaded.

.. autosummary::
   :toctree: generated/

   db.load_pathway_genes
   db.load_exfinder
   db.fetch_all
   db.data_dir
   db.set_data_dir
   db.check_species
   db.clear_memory_cache

Settings
--------

.. autosummary::
   :toctree: generated/

   set_verbosity

.. autodata:: scrich.settings
   :annotation:

   Global settings. ``scrich.settings.autoshow`` (default ``True``): show plots when ``show`` is not given.
```
