#!/bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

uv venv

source .venv/bin/activate

uv pip install -r requirements_evals.txt

uv pip install --upgrade datasets openai
