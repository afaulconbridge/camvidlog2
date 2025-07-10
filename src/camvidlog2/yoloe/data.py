from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def create(
    video: Path, trackings: Iterable[pd.DataFrame], names: Iterable[str]
) -> pd.DataFrame:
    # output data frame has the following columns:
    # - filename: the path to the video file
    # - frame_no: the frame number in the video
    # - x1,y1,x2,y2: the bounding box coordinates
    # - tracker: the tracking ID of the detection
    # - [name]: a column for each class with confidence score

    columns = {
        "frame_no": pd.Series(dtype=pd.UInt32Dtype()),
        "x1": pd.Series(dtype=pd.UInt32Dtype()),
        "y1": pd.Series(dtype=pd.UInt32Dtype()),
        "x2": pd.Series(dtype=pd.UInt32Dtype()),
        "y2": pd.Series(dtype=pd.UInt32Dtype()),
        "tracker": pd.Series(dtype=pd.UInt32Dtype()),
    }
    for name in names:
        columns[str(name)] = pd.Series(dtype=pd.Float32Dtype())

    # Create the DataFrame with empty columns
    array = pd.DataFrame(columns)

    for tracking in trackings:
        array = pd.concat([array, tracking])
    array.insert(0, "filename", str(video))

    return array
