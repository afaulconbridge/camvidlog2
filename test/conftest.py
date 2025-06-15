import os
import os.path
import socket
import subprocess
import time
from pathlib import Path

import ffmpeg
import pytest

from camvidlog2.ai import get_video_embeddings
from camvidlog2.data import create
from camvidlog2.vid import generate_frames_cv2


@pytest.fixture(name="data_directory", scope="session")
def fixture_data_directory() -> Path:
    data_path = Path(os.path.dirname(os.path.realpath(__file__))) / "data"
    assert data_path.exists()
    assert data_path.is_dir()
    return data_path


@pytest.fixture(name="video_path", scope="session")
def fixture_video_path(data_directory: Path) -> Path:
    video_path = data_directory / "test.mp4"
    assert video_path.exists()
    assert video_path.is_file()
    return video_path


@pytest.fixture(name="video_embeddings_path", scope="session")
def fixture_video_embeddings_path(video_path: Path) -> Path:
    # create a database
    db_path = video_path.parent / "test.feather"
    # support re-use in other tests and runs
    # may cause a race condition in rare circumstances
    if not db_path.exists():
        # add embeddings to database
        df = create(video_path, get_video_embeddings(generate_frames_cv2(video_path)))
        # save database
        df.to_feather(db_path)

    return db_path


@pytest.fixture(name="rtsp_server", scope="session")
def fixture_rtsp_server(video_path: Path, tmp_path_factory: pytest.TempPathFactory):
    """
    Pytest fixture that starts an RTSP server using go2rtc on an avaliable port.

    This fixture:
    1. Defines the input file.
    2. Chooses a random port and checks if it's free.
    3. Constructs the RTSP server address.
    4. Constructs the go2rtc config file.
    5. Starts the go2rtc process in the background.
    6. Waits for a short period to allow the server to start.
    7. Yields the RTSP server address to the test function.
    8. After the test function completes, it stops the go2rtc process.

    Returns:
        str: The RTSP server address.
    """

    # Ask OS for free port to use
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]

    rtsp_address = f"rtsp://localhost:{port}/test"

    # Create go2rtc.yaml dynamically
    config_path = tmp_path_factory.mktemp("go2rtc") / "go2rtc.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"""
rtsp:
  listen: ":{port}"

streams:
  test: "ffmpeg:{video_path}#loop"

""")

    # Start go2rtc
    go2rtc_command = [
        "./go2rtc",  # Adjust path to go2rtc executable if needed
        "--config",
        str(config_path),
    ]

    process = subprocess.Popen(go2rtc_command)
    try:

        def is_rtsp_stream_ready(rtsp_address):
            """
            Checks if the RTSP stream is ready by attempting to read from it.
            """
            result = None
            try:
                # Use ffmpeg to probe the stream.  A quick probe is usually sufficient.
                result = ffmpeg.probe(rtsp_address, timeout=1)
                return True
            except ffmpeg.Error as e:
                # If probing fails, the stream isn't ready.
                print(f"Stream not ready: {e}")  # Print the error for debugging
                print(e.stderr)
                print(result)
                return False
            except subprocess.TimeoutExpired as e:
                print(f"Unexpected error: {e}")
                print(result)
                return False

        # Poll the RTSP server until it's ready
        max_attempts = 60
        delay = 1
        for attempt in range(max_attempts):
            if is_rtsp_stream_ready(rtsp_address):
                print("RTSP stream is ready.")
                break
            print(
                f"Attempt {attempt + 1}/{max_attempts}: Stream not ready. Waiting {delay} seconds..."
            )
            time.sleep(delay)
        else:
            raise RuntimeError(
                "RTSP stream did not become ready within the allowed time."
            )

        yield rtsp_address
    finally:
        # Stop the ffmpeg process always
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
