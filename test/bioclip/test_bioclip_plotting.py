from pathlib import Path

import pandas as pd

from camvidlog2.bioclip.plots import plot_distances


def test_plot_distances(tmp_path: Path):
    distances = pd.DataFrame(
        {"distance": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    )
    result_path = tmp_path / "results.png"
    plot_distances(distances, result_path)

    assert result_path.exists()
    assert result_path.is_file()
