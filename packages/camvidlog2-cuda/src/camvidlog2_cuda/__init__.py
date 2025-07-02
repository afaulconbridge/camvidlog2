import os

from camvidlog2.cli import app


def main() -> None:
    env_val = os.environ.get("CVL2_ONNX_PROVIDERS")
    if not env_val:
        os.environ["CVL2_ONNX_PROVIDERS"] = "CUDAExecutionProvider"

    app()
