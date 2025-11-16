#!/bin/bash

git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl --recursive
cd nemo-rl
uv venv

source .venv/bin/activate
export UV_HTTP_TIMEOUT=120
uv sync
