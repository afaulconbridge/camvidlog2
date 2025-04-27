#!/bin/bash
set -euxo pipefail

# preload the models without using the module
# useful for building docker images and similar

uv run python -c "import open_clip;open_clip.create_model_and_transforms(\"hf-hub:imageomics/bioclip\")"
