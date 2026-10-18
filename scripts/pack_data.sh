#!/usr/bin/env bash
# Bundle code + final splits into one archive to copy to the GPU box:
#   bash scripts/pack_data.sh            ->  myjev_bundle.tar.gz
# On the box:  mkdir myjev && tar xzf myjev_bundle.tar.gz -C myjev && cd myjev && bash scripts/setup_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."
tar czf myjev_bundle.tar.gz \
  --exclude='__pycache__' --exclude='*.pyc' \
  pyproject.toml myjev scripts tests docs data/final_v3
ls -lh myjev_bundle.tar.gz
