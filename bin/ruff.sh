#!/bin/bash
set -euxo pipefail

# sort imports
uv run ruff check --select I --fix

uv run ruff format
uv run ruff check --fix
