import typer
from camvidlog2.data import load as data_load
from camvidlog2.data import create
from camvidlog2.ai import get_video_embeddings, get_string_embedding
from camvidlog2.vid import generate_frames_cv2
from pathlib import Path
import pandas as pd

app = typer.Typer()


@app.command()
def load(videos: list[str]):
    # load existing array, if any
    existing_array = data_load(Path("tmp.feather"))
    for video in videos:
        if (
            existing_array is not None
            and existing_array["filename"].str.contains(video).any()
        ):
            print("File already loaded")
            print(video)
            continue

        video_path = Path(video)
        if not video_path.exists():
            print("file does not exist")
            print(video)
            continue

        array = create(
            video_path, get_video_embeddings(generate_frames_cv2(video_path))
        )

        # add new data to existing array
        if existing_array is None:
            existing_array = array
        else:
            existing_array = pd.concat([existing_array, array])

        # save array (new or existing) to disk
        existing_array.to_feather("tmp.feather")
        print("saved file")


@app.command()
def query(query: str):
    df = data_load(Path("tmp.feather"))
    if not df:
        raise ValueError("Unable to load database")
    embedding = get_string_embedding(query)
    df_embeddings = df.drop(columns=["filename", "frame_no"])

    # dot product of unit vectors is the alignment between them (1 = equal, 0 = perpendicular)
    df["distances"] = df_embeddings.dot(embedding)

    # drop embedding columns and cleanup
    df = df[["filename", "frame_no", "distances"]]
    df.reset_index(drop=True, inplace=True)

    # get the rows that are the closest match
    grouped = df.loc[df.groupby("filename")["distances"].idxmax()]

    grouped.sort_values(by="distances", ascending=False, inplace=True)

    for i, row in enumerate(grouped.itertuples()):
        print(row)
        if i >= 10:
            break


if __name__ == "__main__":
    app()
