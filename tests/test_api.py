import pytest

import scrich as ri


def test_version():
    assert isinstance(ri.__version__, str)


def test_public_api_is_importable():
    for name in ri.tl.__all__:
        assert callable(getattr(ri.tl, name)), name
    for name in ri.pl.__all__:
        assert callable(getattr(ri.pl, name)), name
    for name in ["load_pathway_genes", "load_exfinder", "fetch_all", "data_dir", "set_data_dir"]:
        assert callable(getattr(ri.db, name)), name


def test_optional_dependencies_are_lazy():
    """import scrich must work without the optional extras; using them gives an install hint."""
    import importlib.util

    for mod, extra, call in [
        ("plotly", "alluvial", lambda: ri.plots.flows.go.Figure),
        ("scvelo", "velocity", lambda: ri.plots.velocity.scv.pl),
        ("d3blocks", "chord", lambda: ri.plots.ccc._d3blocks.D3Blocks),
    ]:
        if importlib.util.find_spec(mod) is None:
            with pytest.raises(ImportError, match=f"scrich\\[{extra}\\]"):
                call()
