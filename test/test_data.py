from pathlib import Path

from camvidlog2.data import EmbeddingGroup, load_embedding_group_json


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
