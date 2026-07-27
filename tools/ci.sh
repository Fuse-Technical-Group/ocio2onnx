#!/usr/bin/env bash
# The CI gate: static analysis and the test suite, in one pass.
#
# Runs under Python 3.12 — 3.13+ has no opencolorio wheel. `uv` resolves and
# caches the environment, so this is the same command locally and in CI.
set -euo pipefail

cd "$(dirname "$0")/.."

UV="${UV:-uv}"
RUN=("$UV" run --python 3.12 --extra verify --extra dev)

"${RUN[@]}" ruff check .
"${RUN[@]}" ruff format --check .
"${RUN[@]}" pytest "$@"
