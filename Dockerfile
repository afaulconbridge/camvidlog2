# Based on https://github.com/astral-sh/uv-docker-example

FROM ghcr.io/astral-sh/uv:0.8.13-python3.13-bookworm-slim

WORKDIR /app

# Pre-compile when building the image
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies (excluding dev ones)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=README.md,target=README.md \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Preload models etc
RUN --mount=type=bind,source=bin/preload.sh,target=bin/preload.sh \
    --mount=type=cache,target=/root/.cache/huggingface \
    uv run bin/preload.sh

# Add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv` as its not needed any more
ENTRYPOINT ["camvidlog2"]
