#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  printf 'Usage: %s <manifest-path> [extra args...]\n' "$0" >&2
  exit 2
fi

manifest_path="$1"
shift

script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

cd "$repo_root"
uv run python "$script_dir/run_artifact_smoke.py" --manifest "$manifest_path" "$@"
