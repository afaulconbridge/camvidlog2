from camvidlog2.cli import app
from camvidlog2.yoloe.ai import ProvidersList


def main() -> None:
    # modify the providers list object in place
    ProvidersList.providers.clear()
    ProvidersList.providers.append("CUDAExecutionProvider")

    app()
