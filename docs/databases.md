# Databases

scRICH uses two kinds of databases, accessed through {mod}`scrich.db`.

**Pathway ligands and receptors** (CellChatDB, human and mouse, ~50 kB) are included in the package and
used by {func}`scrich.tl.pathways_overview` and {func}`scrich.tl.find_genes`.

**exFinder tables** (ligand -> receptor, receptor -> transcription factor, transcription factor ->
target, and the CellChat interaction table; human and mouse, ~21 MB) are too large for the package. They
are only needed by {func}`scrich.tl.import_database` (downstream genes of pathways). They are hosted on
Zenodo, DOI [10.5281/zenodo.23319307](https://doi.org/10.5281/zenodo.23319307), and are downloaded
automatically on first use, checked against SHA256 checksums, and cached. No setup is needed.

If you use them, please also cite the original exFinder and CellChat publications.

## Where the files are stored

By default in the operating system's cache folder (e.g. `~/Library/Caches/scrich` on macOS,
`~/.cache/scrich` on Linux). To use another folder, for example a shared folder on a cluster whose
compute nodes have no internet access (run `fetch_all` once on a machine with internet):

```bash
export SCRICH_DATA_DIR=/path/to/scrich_db
```

or, in Python:

```python
import scrich as ri
ri.db.set_data_dir("/path/to/scrich_db")
ri.db.fetch_all()   # download or verify all files now
```

If the folder already contains the files, nothing is downloaded.

## Loading the tables

```python
lig = ri.db.load_pathway_genes("ligand", species="mouse")   # pathway -> ligand
rtf = ri.db.load_exfinder("rtf", species="human")           # receptor -> transcription factor
```

The Parquet files were produced from the exFinder data with `scripts/convert_exfinder_to_parquet.py` in
the repository.

## Module reference

```{eval-rst}
.. automodule:: scrich.db
   :no-members:
```

The functions of `scrich.db` are listed in the {doc}`API reference <api/index>`.
