import pytest

import scrich as ri


@pytest.mark.parametrize("species", ["human", "mouse", "Mouse"])
@pytest.mark.parametrize("kind", ["ligand", "receptor"])
def test_bundled_tables(kind, species):
    df = ri.db.load_pathway_genes(kind, species)
    assert list(df.columns) == ["pathway", kind]
    assert len(df) > 400


def test_bad_species():
    with pytest.raises(ValueError, match="Unsupported species"):
        ri.db.load_pathway_genes("ligand", "zebrafish")


def test_bad_layer():
    with pytest.raises(ValueError, match="Unknown layer"):
        ri.db.load_exfinder("foo")


def test_returns_copy():
    a = ri.db.load_pathway_genes("ligand", "human")
    a.loc[0, "pathway"] = "CHANGED"
    assert ri.db.load_pathway_genes("ligand", "human").loc[0, "pathway"] != "CHANGED"


def test_missing_local_files_give_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ri.db, "ZENODO_RECORD", None)
    ri.db.set_data_dir(tmp_path)
    try:
        with pytest.raises(FileNotFoundError, match="SCRICH_DATA_DIR"):
            ri.db.load_exfinder("rtf", "human")
    finally:
        ri.db.set_data_dir(None)


def test_checksum_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(ri.db, "ZENODO_RECORD", None)
    (tmp_path / "rtf_layer2_human.parquet").write_bytes(b"not a real file")
    ri.db.set_data_dir(tmp_path)
    try:
        with pytest.raises(ValueError, match="Checksum mismatch"):
            ri.db.load_exfinder("rtf", "human")
    finally:
        ri.db.set_data_dir(None)


def test_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRICH_DATA_DIR", str(tmp_path))
    assert ri.db.data_dir() == tmp_path


@pytest.mark.network
@pytest.mark.parametrize("layer", ["cellchat", "lr", "rtf", "tft"])
def test_exfinder_tables(exfinder_dir, layer):
    df = ri.db.load_exfinder(layer, "mouse")
    if layer == "cellchat":
        assert {"pathway_name", "ligand", "receptor"} <= set(df.columns)
    else:
        assert list(df.columns) == ["from", "to", "source"]
    assert len(df) > 1000
