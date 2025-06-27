import os
import os.path
import shutil
import socket
import subprocess
import time
from pathlib import Path

import docker
import ffmpeg
import pytest

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
        # Stop the go2rtc process always
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def wait_for_port(host: str, port: int, timeout: float = 10.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"service on {host}:{port} did not start in time.")


@pytest.fixture(scope="session", name="docker_client")
def fixture_docker_client():
    return docker.from_env()


@pytest.fixture(scope="session", name="rtsp_server_docker")
def fixture_rtsp_server_docker(
    video_path: Path,
    docker_client: docker.DockerClient,
    tmp_path_factory: pytest.TempPathFactory,
):
    """
    Spins up go2rtc in Docker for testing RTSP streaming, serving a test video.
    Requires Docker to be installed and running.
    Yields the RTSP server URL.
    """

    # Prepare config and test video in a temp directory
    workdir = tmp_path_factory.mktemp("go2rtc_docker")
    config_path = workdir / "go2rtc.yaml"
    video_dest = workdir / "test.mp4"

    config_path.write_text("""streams:
  test: "ffmpeg:/input/test.mp4#loop"
""")
    shutil.copy(str(video_path), str(video_dest))

    container = docker_client.containers.run(
        "alexxit/go2rtc:latest",
        name="pytest-go2rtc-rtsp",
        remove=True,
        detach=True,
        ports={
            "8554/tcp": None,  # map to a random host port
            "1984/tcp": 1984,
        },
        volumes={
            str(workdir): {"bind": "/config", "mode": "ro"},
            str(workdir): {"bind": "/input", "mode": "ro"},
        },
        # command=["--config", "/config/go2rtc.yaml"],
    )

    try:
        # Inspect to get the host port
        container.reload()
        port_info = container.attrs["NetworkSettings"]["Ports"]["8554/tcp"][0]
        host = port_info["HostIp"]
        port = int(port_info["HostPort"])

        wait_for_port(host, port, timeout=10)

        rtsp_url = f"rtsp://localhost:{port}/test"

        # probe using ffmpeg to ensure the stream is ready
        for _ in range(300):
            try:
                ffmpeg.probe(rtsp_url, timeout=10)
                break
            except (ffmpeg.Error, subprocess.TimeoutExpired):
                time.sleep(1)
        else:
            raise RuntimeError("go2rtc RTSP stream did not become ready in time.")

        yield rtsp_url
    finally:
        container.stop(timeout=5)
        # Fetch and print the logs
        logs = container.logs().decode("utf-8")
        print(logs)


@pytest.fixture(name="mqtt_broker", scope="session")
def fixture_mqtt_broker_docker(
    docker_client: docker.DockerClient,
    tmp_path_factory: pytest.TempPathFactory,
):
    """
    Spins up an Eclipse Mosquitto MQTT broker in Docker for testing.
    Requires Docker to be installed and running.
    Yields broker connection info as a dict.
    """
    workdir = tmp_path_factory.mktemp("mosquitto_mqtt")
    config_path = workdir / "mosquitto.conf"
    config_path.write_text("""listener 1883 0.0.0.0
allow_anonymous true
""")

    container = docker_client.containers.run(
        "eclipse-mosquitto:2.0",
        name="pytest-mosquitto-mqtt",
        remove=True,
        detach=True,
        ports={"1883/tcp": None},  # map to a random host port
        volumes={
            str(config_path): {"bind": "/mosquitto/config/mosquitto.conf", "mode": "ro"}
        },
    )
    try:
        # Inspect to get the host port
        container.reload()
        port_info = container.attrs["NetworkSettings"]["Ports"]["1883/tcp"][0]
        host = port_info["HostIp"]
        port = int(port_info["HostPort"])

        wait_for_port(host, port, timeout=10)
        yield {"host": host, "port": port}
    finally:
        # Fetch and print the logs
        logs = container.logs().decode("utf-8")
        print(logs)
        container.stop(timeout=3)
