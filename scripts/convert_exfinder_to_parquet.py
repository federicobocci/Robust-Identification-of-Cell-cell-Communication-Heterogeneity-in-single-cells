"""Convert the exFinder / CellChat CSV exports to the Parquet files used by ``scrich.db``.

The CSVs are produced from the exFinder R data (``*.rda``) with ``write.csv`` and
contain a row-index column and a redundant ``path`` column (``from*to``). Both are
dropped; ``from``, ``to`` and ``source`` are stored as categoricals with zstd
compression (~600 MB of CSV -> ~21 MB of Parquet, identical content).

Usage
-----
    python scripts/convert_exfinder_to_parquet.py /path/to/exfinder_database/data OUT_DIR

Afterwards update the checksums in ``src/scrich/db/_registry.py`` with the
printed values (or ``sha256sum OUT_DIR/*.parquet``).
"""

import sys
from pathlib import Path

import pandas as pd
import pooch

NAMES = {
    "interaction_input_CellChatDB": "cellchat_interactions",
    "LR_layer1": "lr_layer1",
    "RTF_layer2": "rtf_layer2",
    "TFT_layer3": "tft_layer3",
}


def main(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for species in ("human", "mouse"):
        for stem, new in NAMES.items():
            df = pd.read_csv(src / f"{stem}_{species}.csv", index_col=0, keep_default_na=False, na_values=[""])
            if "path" in df:
                assert (df["path"] == df["from"] + "*" + df["to"]).all()
                df = df.drop(columns="path")
                for col in ("from", "to", "source"):
                    df[col] = df[col].astype("category")
            target = out / f"{new}_{species}.parquet"
            df.to_parquet(target, compression="zstd", index=False)
            print(f'    "{target.name}": "sha256:{pooch.file_hash(str(target))}",')


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
