# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dockerfile using uv environment.

ARG TARGETPLATFORM
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

FROM ${BASE_IMAGE}

# Set the DEBIAN_FRONTEND environment variable to avoid interactive prompts during apt operations.
ENV DEBIAN_FRONTEND=noninteractive

# Install packages
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        git \
        git-lfs \
        tree \
        wget

# Install uv: https://docs.astral.sh/uv/getting-started/installation/
# https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile
COPY --from=ghcr.io/astral-sh/uv:0.8.12 /uv /uvx /usr/local/bin/
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy
# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Install just: https://just.systems/man/en/pre-built-binaries.html
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin --tag 1.42.4

WORKDIR /workspace

# Install the project's dependencies using the lockfile and settings
ARG CUDA_NAME=cu128
ENV CUDA_NAME=${CUDA_NAME}
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    --mount=type=bind,source=packages,target=packages \
    uv sync --locked --no-install-project --extra=${CUDA_NAME}

# Copy the code into the container if in standalone mode. Otherwise, just install the dependencies at runtime.
# We mount the source code to /tmp and copy it to /workspace if in standalone mode.
ARG STANDALONE
RUN --mount=type=bind,source=.,target=/tmp/workspace \
   if [ "$STANDALONE" = "true" ] ; then cp -r /tmp/workspace/* /workspace && just install ${CUDA_NAME} && rm -rf /workspace/.git ; else echo "Run just install to install all the dependencies at runtime" ; fi

# Place executables in the environment at the front of the path
ENV PATH="/workspace/.venv/bin:$PATH"

# Install Ray for Lilypad workload orchestration. Ray is not in uv.lock (it's Applied-internal),
# so we install it separately. The entrypoint uses --inexact to prevent uv sync from pruning it.
RUN uv pip install "ray[default]==2.50.1.7" --extra-index-url https://ursa.pypi.applied.dev/simple

# click 8.3.x _Sentinel enum uses object() values that fail Python 3.10's deepcopy identity
# reconstruction; Ray's add_command_alias calls copy.deepcopy() at import time and crashes.
# Downgrade to 8.2.x which has proper __deepcopy__ support.
RUN uv pip install "click==8.2.1"

# Install Lilypad SDK for cross-region boto caching utilities.
RUN uv pip install "lilypad-py==2.27.0" --extra-index-url https://ursa.pypi.applied.dev/simple

# Skip uv sync at container start — all deps are already installed above (STANDALONE=true bakes
# everything in at build time). This prevents uv sync from downgrading click and breaking Ray.
ENV SKIP_UV_SYNC=true

ENTRYPOINT ["/workspace/bin/entrypoint.sh"]

CMD ["/bin/bash"]
