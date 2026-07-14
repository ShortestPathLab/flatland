#!/bin/bash
set -e # stop on error
set -x # echo commands


FLATLAND_BASEDIR=$(dirname "$BASH_SOURCE")/..
cd ${FLATLAND_BASEDIR}

# uv reads .python-version, provisions the interpreter itself and builds the
# virtualenv from uv.lock. Install uv first: https://docs.astral.sh/uv/
uv sync --locked

uv run pytest -v
uv run jupyter notebook &
