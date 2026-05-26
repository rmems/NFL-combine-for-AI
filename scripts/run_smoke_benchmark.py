from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmarks.runner import run_benchmarks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny smoke benchmark with dummy data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write report outputs (default: reports).",
    )
    parser.add_argument(
        "--formats",
        default="json,csv",
        help="Comma-separated list of output formats (json,csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the run.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        config_path = tmpdir_path / "smoke.json"
        dataset_path = tmpdir_path / "smoke.jsonl"

        with dataset_path.open("w") as f:
            f.write(json.dumps({"prompt": "The cat sat on the ", "reference": "mat"}) + "\n")
            f.write(json.dumps({"prompt": "She walked to the ", "reference": "store"}) + "\n")
            f.write(json.dumps({"prompt": "He sat on the ", "reference": "chair"}) + "\n")

        config = {
            "run_name": "smoke-benchmark",
            "seed": args.seed,
            "model": {"backend": "mock", "name": "smoke-model", "revision": "local"},
            "quantization": ["fp16", "awq"],
            "datasets": [
                {
                    "name": "smoke",
                    "source": "jsonl",
                    "path": str(dataset_path),
                    "split": "validation",
                    "max_samples": 3,
                }
            ],
        }
        with config_path.open("w") as f:
            json.dump(config, f)

        formats = [value.strip() for value in args.formats.split(",") if value.strip()]
        metadata = run_benchmarks(
            config_path=config_path,
            output_dir=args.output_dir,
            formats=formats,
            seed_override=args.seed,
        )
        print(f"Smoke benchmark completed: {metadata.run_id}")


if __name__ == "__main__":
    main()
