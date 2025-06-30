from camvidlog2.cli import app
from camvidlog2.yoloe.ai import ProvidersList

if __name__ == "__main__":
    # modify the providers list object in place
    ProvidersList.providers.clear()
    ProvidersList.providers.append("CUDAExecutionProvider")
    app()
