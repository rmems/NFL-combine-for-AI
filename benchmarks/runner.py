from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.datasets import DatasetSpec, LoadedDataset, default_dataset_registry
from benchmarks.metrics import MetricsAccumulator, MetricsSummary
from benchmarks.models import (
    ModelSpec,
    QuantizationProfile,
    build_model_adapter,
    default_quantization_registry,
    scoped_seed,
)
from benchmarks.reporting import metrics_to_row, write_csv, write_json


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    run_name: str
    seed: int
    git_commit: str
    git_branch: str
    host: str
    os: str
    cpu: str
    cpu_count: int | None
    timestamp: str


@dataclass(frozen=True)
class DatasetResult:
    dataset: str
    split: str
    sample_count: int
    quantization: QuantizationProfile
    metrics: MetricsSummary


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_git_info() -> tuple[str, str]:
    commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        .strip()
    )
    branch = (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True)
        .strip()
    )
    return commit, branch


def build_metadata(run_name: str, seed: int) -> RunMetadata:
    commit, branch = get_git_info()
    return RunMetadata(
        run_id=f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{commit[:7]}",
        run_name=run_name,
        seed=seed,
        git_commit=commit,
        git_branch=branch,
        host=platform.node(),
        os=platform.platform(),
        cpu=platform.processor() or platform.machine(),
        cpu_count=os.cpu_count(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def load_datasets(config: dict[str, Any], base_path: Path) -> list[LoadedDataset]:
    registry = default_dataset_registry()
    datasets = []
    for raw in config["datasets"]:
        spec = DatasetSpec.from_dict(raw)
        if spec.path:
            path = Path(spec.path)
            if not path.is_absolute():
                spec = replace(spec, path=str((base_path / path).resolve()))
        loader = registry.loader_for(spec.source)
        datasets.append(loader.load(spec))
    return datasets


def run_benchmarks(
    config_path: Path,
    output_dir: Path,
    formats: list[str],
    seed_override: int | None = None,
) -> RunMetadata:
    config = load_config(config_path)
    run_name = config.get("run_name", "benchmark-run")
    seed = seed_override if seed_override is not None else config.get("seed", 0)
    metadata = build_metadata(run_name, seed)

    datasets = load_datasets(config, config_path.parent)
    model_spec = ModelSpec.from_dict(config["model"])
    quantization_names = config.get("quantization") or ["fp16"]

    registry = default_quantization_registry()
    results: list[DatasetResult] = []

    for quant_name in quantization_names:
        profile = registry.get(quant_name)
        adapter = build_model_adapter(model_spec, profile)

        for dataset in datasets:
            scoped = scoped_seed(seed, model_spec.name, quant_name, dataset.spec.name)
            rng = random.Random(scoped)
            accumulator = MetricsAccumulator()

            for record in dataset.records:
                prediction = adapter.predict(record, rng)
                accumulator.add(record, prediction)

            total_time = (
                accumulator.token_count / profile.speed_tps
                if profile.speed_tps
                else 0.0
            )
            metrics = accumulator.summary(total_time, profile.vram_gb)
            results.append(
                DatasetResult(
                    dataset=dataset.spec.name,
                    split=dataset.spec.split,
                    sample_count=len(dataset.records),
                    quantization=profile,
                    metrics=metrics,
                )
            )

    write_reports(output_dir, formats, metadata, model_spec, results)
    return metadata


def write_reports(
    output_dir: Path,
    formats: list[str],
    metadata: RunMetadata,
    model_spec: ModelSpec,
    results: list[DatasetResult],
) -> None:
    rows = []
    for result in results:
        row = {
            "run_id": metadata.run_id,
            "run_name": metadata.run_name,
            "seed": metadata.seed,
            "git_commit": metadata.git_commit,
            "git_branch": metadata.git_branch,
            "host": metadata.host,
            "os": metadata.os,
            "cpu": metadata.cpu,
            "cpu_count": metadata.cpu_count,
            "timestamp": metadata.timestamp,
            "model_name": model_spec.name,
            "model_backend": model_spec.backend,
            "model_revision": model_spec.revision,
            "dataset": result.dataset,
            "split": result.split,
            "sample_count": result.sample_count,
            "quantization": result.quantization.name,
            "precision": result.quantization.precision,
            "quantization_format": result.quantization.format,
            "quantization_bits": result.quantization.bits,
            **metrics_to_row(result.metrics),
        }
        rows.append(row)

    payload = {
        "run": {
            "run_id": metadata.run_id,
            "run_name": metadata.run_name,
            "seed": metadata.seed,
            "git_commit": metadata.git_commit,
            "git_branch": metadata.git_branch,
            "host": metadata.host,
            "os": metadata.os,
            "cpu": metadata.cpu,
            "cpu_count": metadata.cpu_count,
            "timestamp": metadata.timestamp,
            "model": {
                "name": model_spec.name,
                "backend": model_spec.backend,
                "revision": model_spec.revision,
            },
        },
        "results": rows,
    }

    run_id = metadata.run_id
    if "json" in formats:
        write_json(output_dir / "json" / f"{run_id}.json", payload)
    if "csv" in formats:
        write_csv(output_dir / "csv" / f"{run_id}.csv", rows)
