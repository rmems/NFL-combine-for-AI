#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  printf 'Usage: %s <manifest-path> [extra args...]\n' "$0" >&2
  exit 2
fi

manifest_path="$1"
shift

if realpath -m / >/dev/null 2>&1; then
  manifest_path="$(realpath -m "$manifest_path")"
else
  manifest_path="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$manifest_path")"
fi

script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

cd "$repo_root"
uv run python "$script_dir/run_artifact_smoke.py" --manifest "$manifest_path" "$@"
