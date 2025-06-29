from pathlib import Path

import pandas as pd


def load(path: Path) -> pd.DataFrame | None:
    # load existing array, if any
    existing_array = None
    if path.exists():
        existing_array = pd.read_feather(path)
        existing_array.set_index(["filename", "frame_no"], inplace=True)

        # make sure all column names are strings so they roundtrip correctly
        existing_array.columns = [str(x) for x in existing_array.columns]  # type: ignore

    return existing_array
