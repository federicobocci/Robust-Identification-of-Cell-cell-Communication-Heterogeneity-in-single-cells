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
