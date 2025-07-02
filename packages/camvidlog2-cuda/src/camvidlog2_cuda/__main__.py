import os

from camvidlog2.cli import app

if __name__ == "__main__":
    env_val = os.environ.get("CVL2_ONNX_PROVIDERS")
    if not env_val:
        os.environ["CVL2_ONNX_PROVIDERS"] = "CUDAExecutionProvider"
    app()
