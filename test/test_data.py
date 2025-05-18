from pathlib import Path

from camvidlog2.data import (
    EmbeddingGroup,
    FrameEmbedding,
    StringEmbedding,
    load_embedding_group_json,
)


def test_load_embedding_group_json(tmp_path: Path) -> None:
    data = """{
    "items":[
        {
            "source": "string",
            "query": "hello world"
        },
        {
            "source": "frame",
            "filepath": "example.mp4",
            "frame_no": 1
        }
    ]
}
"""
    tmp_path.mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "data.json", "w") as f:
        f.write(data)

    loaded_group = load_embedding_group_json(tmp_path / "data.json")

    assert isinstance(loaded_group, EmbeddingGroup)
    assert len(loaded_group.items) == 2
    # Verify the string embedding contents
    assert loaded_group.items[0].source == "string"
    assert loaded_group.items[0].query == "hello world"
    # Verify the frame embedding contents
    assert loaded_group.items[1].source == "frame"
    # Verify the frame embedding contents
    assert loaded_group.items[1].source == "frame"
    assert loaded_group.items[1].filepath == Path("example.mp4")
    assert loaded_group.items[1].frame_no == 1
    assert isinstance(loaded_group.items[0], StringEmbedding)
    assert isinstance(loaded_group.items[1], FrameEmbedding)
