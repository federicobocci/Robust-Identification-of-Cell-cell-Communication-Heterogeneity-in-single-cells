"""Registry of the external (non-bundled) database files used by scRICH.

The files are Parquet conversions of the exFinder / CellChat tables
(see ``scripts/convert_exfinder_to_parquet.py``). They are too large for the
Python package itself and are hosted on Zenodo instead
(DOI `10.5281/zenodo.23319307 <https://doi.org/10.5281/zenodo.23319307>`_).

To publish a new database version:

1. Upload the ``*.parquet`` files to a Zenodo record.
2. Set :data:`ZENODO_RECORD` to that record id and :data:`DB_VERSION` accordingly.
3. Update the checksums below (``sha256sum *.parquet``).
"""

from __future__ import annotations

#: Version label of the database release. Also used as the cache sub-folder.
DB_VERSION = "v1"

#: Zenodo record id hosting the files (https://zenodo.org/records/23319307).
ZENODO_RECORD: str | None = "23319307"

#: DOI of the database release, for citation.
ZENODO_DOI = "10.5281/zenodo.23319307"

#: File name -> ``"sha256:<hash>"``.
REGISTRY: dict[str, str] = {
    "cellchat_interactions_human.parquet": "sha256:4f1f8ef61b779d63e79d7af4bedaf011c4f7a7a5c02b3f11e8467b4cb374c732",
    "cellchat_interactions_mouse.parquet": "sha256:135edefdd551240258055bac7ffa2e8640af9a4f540f1bf0a21a2ac1a99d52db",
    "lr_layer1_human.parquet": "sha256:30a941620868133539a3690b3ab25e29cabbb590cff057bbe020bee948604ffc",
    "lr_layer1_mouse.parquet": "sha256:06808d2c2b5cca6c7cab1faed25dd5ba860554730d7d1a492cf3b026aa42fb46",
    "rtf_layer2_human.parquet": "sha256:fc9f9f7a65131386fd982dd6f56d153ac0bdefe463eead147e864034f3cf0319",
    "rtf_layer2_mouse.parquet": "sha256:acca1680b01b8dde04bf94de0b8f8b1648e9ed7a8137bcd7b6c150f0a66d1976",
    "tft_layer3_human.parquet": "sha256:15630fa037e22102d6b0a9d3785cb82ac2bc2def7a64a5a9099de339d3651cf8",
    "tft_layer3_mouse.parquet": "sha256:1a97e57b667f2660e5c1872b09979b4b1707b25c66103882c1b1aa626441355e",
}
