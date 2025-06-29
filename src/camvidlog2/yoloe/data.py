from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def create(
    video: Path, trackings: Iterable[tuple[int, np.ndarray]], names: Iterable[str]
) -> pd.DataFrame:
    # output data frame has the following columns:
    # - filename: the path to the video file
    # - frame_no: the frame number in the video
    # - x1,y1,x2,y2: the bounding box coordinates
    # - conf: the confidence score of the detection
    # - class: the class label of the detection
    # - tracker: the tracking ID of the detection

    # Create the DataFrame with empty columns
    array = pd.DataFrame(
        {
            "filename": pd.Series(dtype=pd.StringDtype()),
            "frame_no": pd.Series(dtype=pd.UInt32Dtype()),
            "x1": pd.Series(dtype=pd.UInt32Dtype()),
            "y1": pd.Series(dtype=pd.UInt32Dtype()),
            "x2": pd.Series(dtype=pd.UInt32Dtype()),
            "y2": pd.Series(dtype=pd.UInt32Dtype()),
            "conf": pd.Series(dtype=pd.Float32Dtype()),
            "class": pd.Series(
                pd.Categorical([], categories=list(names), ordered=False)
            ),
            "tracker": pd.Series(dtype=pd.UInt32Dtype()),
        }
    )

    # make sure all column names are strings so they roundtrip correctly
    array.columns = [str(x) for x in array.columns]  # type: ignore
    return array
