import os
import os.path
import random
import socket
import subprocess
import time
from pathlib import Path

import ffmpeg
from pytest import TempPathFactory, fixture

from camvidlog2.ai import get_video_embeddings
from camvidlog2.data import create
from camvidlog2.vid import generate_frames_cv2


@fixture(name="data_directory", scope="session")
def fixture_data_directory() -> Path:
    data_path = Path(os.path.dirname(os.path.realpath(__file__))) / "data"
    assert data_path.exists()
    assert data_path.is_dir()
    return data_path


@fixture(name="video_path", scope="session")
def fixture_video_path(data_directory: Path) -> Path:
    video_path = data_directory / "test.mp4"
    assert video_path.exists()
    assert video_path.is_file()
    return video_path


@fixture(name="video_embeddings_path", scope="session")
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


@fixture(name="rtsp_server", scope="session")
def fixture_rtsp_server(video_path: Path, tmp_path_factory: TempPathFactory):
    """
    Pytest fixture that starts an RTSP server using ffmpeg-python with a randomized port.

    This fixture:
    1. Defines the input file.
    2. Chooses a random port and checks if it's free.
    3. Constructs the RTSP server address.
    4. Constructs the ffmpeg command to stream the input file over RTSP using ffmpeg-python.
    5. Starts the ffmpeg process in the background.
    6. Waits for a short period to allow the server to start.
    7. Yields the RTSP server address to the test function.
    8. After the test function completes, it stops the ffmpeg process.

    Returns:
        str: The RTSP server address.
    """

    def is_port_free(port):
        """Checks if a port is free."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("localhost", port))
            return True
        except socket.error:
            return False
        finally:
            sock.close()

    # Choose a random port and check if it's free
    for _ in range(10):  # Try up to 10 times to find a free port
        random_port = random.randint(
            10000, 65535
        )  # Choose a port in the dynamic/private range
        if is_port_free(random_port):
            break
    else:
        raise Exception("Could not find a free port after 10 attempts")

    rtsp_address = f"rtsp://localhost:{random_port}/test"

    # Create go2rtc.yaml dynamically
    config_path = tmp_path_factory.mktemp("go2rtc") / "go2rtc.yaml"
    with open(config_path, "w") as f:
        f.write(f"""
rtsp:
  listen: ":{random_port}"

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
            try:
                # Use ffmpeg to probe the stream.  A quick probe is usually sufficient.
                ffmpeg.probe(rtsp_address, timeout=1)
                return True
            except ffmpeg.Error as e:
                # If probing fails, the stream isn't ready.
                print(f"Stream not ready: {e}")  # Print the error for debugging
                return False
            except Exception as e:
                print(f"Unexpected error: {e}")
                return False

        # Poll the RTSP server until it's ready
        max_attempts = 60
        delay = 1
        for attempt in range(max_attempts):
            if is_rtsp_stream_ready(rtsp_address):
                print("RTSP stream is ready.")
                break
            else:
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
        process.wait()
