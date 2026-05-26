#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  printf 'Usage: %s <manifest-path> [extra args...]\n' "$0" >&2
  exit 2
fi

manifest_path="$1"
shift

uv run python scripts/run_artifact_smoke.py --manifest "$manifest_path" "$@"
