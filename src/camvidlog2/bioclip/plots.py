from pathlib import Path

import pandas as pd
import seaborn as sns
import seaborn.objects as so

sns.set_theme(style="darkgrid")


def plot_distances(distances: pd.DataFrame, outpath: str | Path) -> None:
    # verify distances has structure and type expected
    if "distance" not in distances.columns:
        raise ValueError("Missing column 'distance'")

    (
        so.Plot(distances, x="distance")
        .add(so.Bars(), so.Hist(stat="percent"))
        .save(str(outpath))
    )
