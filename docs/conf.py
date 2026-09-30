"""Sphinx configuration for the scRICH documentation."""

import json
import re
import shutil
from datetime import date
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent

# -- Project -----------------------------------------------------------------
project = "scRICH"
author = "Federico Bocci"
copyright = f"{date.today().year}, {author}"
try:
    release = _version("scrich")
except PackageNotFoundError:
    release = "dev"
version = release

# -- Tutorials and images live outside docs/: copy them in at build time -----
_tut_dir = HERE / "tutorials"
for nb in sorted((ROOT / "tutorials").glob("tutorial-*.ipynb")):
    slug = re.sub(r"[^a-z0-9]+", "-", nb.stem.lower()).strip("-")
    content = json.loads(nb.read_text())
    # every page needs a title: add one if the notebook does not start with a level-1 heading
    has_title = any(c["cell_type"] == "markdown" and "".join(c["source"]).lstrip().startswith("# ")
                    for c in content["cells"])
    if not has_title:
        title = nb.stem.split("-", 2)[-1].replace("_", " ").replace("-", " ")
        content["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}"]})
    (_tut_dir / f"{slug}.ipynb").write_text(json.dumps(content))
(HERE / "_static").mkdir(exist_ok=True)
shutil.copyfile(ROOT / "misc" / "GA.png", HERE / "_static" / "scrich_schematic.png")

# -- General -----------------------------------------------------------------
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}

# -- API documentation (NumPy-style docstrings) -------------------------------
autosummary_generate = True
autodoc_typehints = "none"          # types are documented in the docstrings
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "scvelo": ("https://scvelo.readthedocs.io/en/stable/", None),
}

# -- Notebooks (tutorials are shown with their saved outputs, not executed) ---
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "dollarmath", "deflist"]
myst_heading_anchors = 3
# tutorials use H1 -> H3 headings and contain interactive plotly outputs
suppress_warnings = ["myst.header", "mystnb.unknown_mime_type"]
nb_mime_priority_overrides = [("html", "application/vnd.plotly.v1+json", None)]

# -- HTML ----------------------------------------------------------------------
html_theme = "furo"
html_title = f"scRICH {release}"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/federicobocci/Robust-Identification-of-Cell-cell-Communication-Heterogeneity-in-single-cells/",
    "source_branch": "main",
    "source_directory": "docs/",
}
