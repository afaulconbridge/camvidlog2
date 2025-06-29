from pathlib import Path

import pandas as pd

from camvidlog2.vid import generate_frames_cv2
from camvidlog2.yoloe.ai import generate_tracked_bboxes


def test_generate_tracked_bboxes_with_real_video(video_path: Path):
    frames = (f for _, f in generate_frames_cv2(video_path))
    result = list(
        generate_tracked_bboxes(
            [next(frames), next(frames)],
            "yoloe-11l-seg.onnx",
            ["deer", "bird", "hedgehog", "otter", "giraffe"],
        )
    )
    assert len(result) == 2
    for df in result:
        assert isinstance(df, pd.DataFrame)
        assert set(
            ["frame_no", "x1", "y1", "x2", "y2", "conf", "class", "tracker"]
        ).issubset(df.columns)
        # Check that 'class' column is categorical with correct categories
        assert isinstance(df["class"], pd.CategoricalDtype)
        assert set(df["class"].cat.categories) == set(
            ["deer", "bird", "hedgehog", "otter", "giraffe"]
        )
