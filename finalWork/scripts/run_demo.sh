#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
python3 -m moneygraph demo --out results/demo --size 2248 --stability
