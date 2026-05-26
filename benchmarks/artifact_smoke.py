from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.datasets import DatasetSpec, JsonlDatasetLoader
from benchmarks.metrics import MetricsAccumulator
from benchmarks.models import ModelSpec, build_model_adapter, default_quantization_registry
from benchmarks.reporting import write_csv, write_json
from benchmarks.runner import build_metadata
from nfl_combine_for_ai.manifest import (
    ArtifactFormat,
    ArtifactStatus,
    GeneratedArtifact,
    ModelManifest,
    load_manifest,
)


@dataclass(frozen=True)
class ArtifactSelection:
    status: str
    source_format: str
    runtime_format: str | None
    quantization_name: str | None
    generated_format: str | None
    artifact_path: str | None
    failure_reason: str | None = None


def _resolve_path(base_path: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_path / path).resolve()


def _runnable_generated_artifact(base_path: Path, generated: list[GeneratedArtifact]) -> ArtifactSelection | None:
    for artifact in generated:
        if artifact.status not in (ArtifactStatus.SUCCESS, ArtifactStatus.PARTIAL):
            continue
        if artifact.format not in (ArtifactFormat.AWQ, ArtifactFormat.GPTQ):
            continue
        resolved = _resolve_path(base_path, artifact.path)
        if resolved and resolved.exists():
            return ArtifactSelection(
                status="success",
                source_format="generated",
                runtime_format=f"generated_{artifact.format.value}",
                quantization_name=artifact.format.value,
                generated_format=artifact.format.value,
                artifact_path=str(resolved),
            )
    return None


def select_artifact_for_smoke(manifest: ModelManifest, base_path: Path) -> ArtifactSelection:
    generated = _runnable_generated_artifact(base_path, manifest.generated_artifacts)
    if generated is not None:
        return generated

    source = manifest.source_artifact
    source_path = _resolve_path(base_path, source.path)

    if source.format == ArtifactFormat.GGUF:
        if source_path and source_path.exists():
            return ArtifactSelection(
                status="success",
                source_format=source.format.value,
                runtime_format="gguf",
                quantization_name="gguf",
                generated_format=None,
                artifact_path=str(source_path),
            )
        return ArtifactSelection(
            status="failed",
            source_format=source.format.value,
            runtime_format=None,
            quantization_name=None,
            generated_format=None,
            artifact_path=str(source_path) if source_path else None,
            failure_reason="GGUF source artifact path does not exist",
        )

    if source.format in (ArtifactFormat.SAFETENSORS, ArtifactFormat.HF):
        if (source_path and source_path.exists()) or source.hf_repo_id:
            return ArtifactSelection(
                status="success",
                source_format=source.format.value,
                runtime_format="safetensors_hf",
                quantization_name="fp16",
                generated_format=None,
                artifact_path=str(source_path) if source_path else None,
            )
        return ArtifactSelection(
            status="failed",
            source_format=source.format.value,
            runtime_format=None,
            quantization_name=None,
            generated_format=None,
            artifact_path=str(source_path) if source_path else None,
            failure_reason="HF/Safetensors source is not loadable: missing local path and hf_repo_id",
        )

    return ArtifactSelection(
        status="failed",
        source_format=source.format.value,
        runtime_format=None,
        quantization_name=None,
        generated_format=None,
        artifact_path=str(source_path) if source_path else None,
        failure_reason=f"Unsupported source format for smoke run: {source.format.value}",
    )


def _load_smoke_dataset(dataset_path: Path, max_samples: int) -> tuple[str, list[Any]]:
    loader = JsonlDatasetLoader()
    spec = DatasetSpec(
        name="lambada-smoke",
        source="jsonl",
        path=str(dataset_path),
        split="validation",
        max_samples=max_samples,
    )
    loaded = loader.load(spec)
    return loaded.spec.name, loaded.records


def _peak_vram_gb_from_metadata(metadata_vram_gb: float, gpu_memory_mb: list[int | None]) -> float | None:
    seen = [value for value in gpu_memory_mb if value is not None]
    if seen:
        return max(seen) / 1024.0
    return metadata_vram_gb


def _build_failure_payload(
    metadata: Any,
    manifest: ModelManifest,
    manifest_path: Path,
    selection: ArtifactSelection,
) -> dict[str, Any]:
    gpu_names = metadata.telemetry.system.gpu_names or []
    return {
        "run": {
            "run_id": metadata.run_id,
            "run_name": metadata.run_name,
            "timestamp": metadata.timestamp,
        },
        "result": {
            "status": "failed",
            "model_id": manifest.model_name,
            "source_format": selection.source_format,
            "runtime_format": selection.runtime_format,
            "generated_format": selection.generated_format,
            "artifact_path": selection.artifact_path,
            "gpu": gpu_names[0] if gpu_names else None,
            "cuda": metadata.telemetry.system.cuda_version,
            "failure_reason": selection.failure_reason,
            "manifest_path": str(manifest_path),
        },
    }


def _build_success_row(
    metadata: Any,
    manifest: ModelManifest,
    manifest_path: Path,
    selection: ArtifactSelection,
    profile: Any,
    sample_count: int,
    perplexity: float,
) -> dict[str, Any]:
    gpu_names = metadata.telemetry.system.gpu_names or []
    gpu_metrics = metadata.telemetry.gpu_metrics or []
    peak_vram_gb = _peak_vram_gb_from_metadata(
        profile.vram_gb,
        [metric.memory_used_mb for metric in gpu_metrics],
    )
    return {
        "run_id": metadata.run_id,
        "status": "success",
        "timestamp": metadata.timestamp,
        "model_id": manifest.model_name,
        "model_family": manifest.model_family,
        "manifest_path": str(manifest_path),
        "source_format": selection.source_format,
        "runtime_format": selection.runtime_format,
        "generated_format": selection.generated_format,
        "quantization": selection.quantization_name,
        "artifact_path": selection.artifact_path,
        "gpu": gpu_names[0] if gpu_names else None,
        "cuda": metadata.telemetry.system.cuda_version,
        "throughput_tps": profile.speed_tps,
        "generation_tokens": 512,
        "peak_vram_gb": peak_vram_gb,
        "perplexity": perplexity,
        "dataset": "lambada-smoke",
        "sample_count": sample_count,
        "failure_reason": None,
    }


def _build_failure_row(
    metadata: Any,
    manifest: ModelManifest,
    manifest_path: Path,
    selection: ArtifactSelection,
) -> dict[str, Any]:
    gpu_names = metadata.telemetry.system.gpu_names or []
    return {
        "run_id": metadata.run_id,
        "status": "failed",
        "timestamp": metadata.timestamp,
        "model_id": manifest.model_name,
        "model_family": manifest.model_family,
        "manifest_path": str(manifest_path),
        "source_format": selection.source_format,
        "runtime_format": selection.runtime_format,
        "generated_format": selection.generated_format,
        "quantization": selection.quantization_name,
        "artifact_path": selection.artifact_path,
        "gpu": gpu_names[0] if gpu_names else None,
        "cuda": metadata.telemetry.system.cuda_version,
        "throughput_tps": None,
        "generation_tokens": 512,
        "peak_vram_gb": None,
        "perplexity": None,
        "dataset": "lambada-smoke",
        "sample_count": 0,
        "failure_reason": selection.failure_reason,
    }


def _write_failure_outputs(
    output_dir: Path,
    formats: list[str],
    metadata: Any,
    manifest: ModelManifest,
    manifest_path: Path,
    selection: ArtifactSelection,
) -> dict[str, Any]:
    row = _build_failure_row(metadata, manifest, manifest_path, selection)
    payload = _build_failure_payload(metadata, manifest, manifest_path, selection)
    run_id = metadata.run_id
    if "json" in formats:
        write_json(output_dir / "json" / f"{run_id}.artifact-smoke.json", payload)
    if "csv" in formats:
        write_csv(output_dir / "csv" / f"{run_id}.artifact-smoke.csv", [row])
    return payload


def run_artifact_smoke(
    manifest_path: Path,
    output_dir: Path,
    formats: list[str],
    dataset_path: Path | None = None,
    max_samples: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    metadata = build_metadata(f"artifact-smoke-{manifest.model_name}", seed)
    selection = select_artifact_for_smoke(manifest, manifest_path.parent)

    if selection.status != "success":
        return _write_failure_outputs(output_dir, formats, metadata, manifest, manifest_path, selection)

    try:
        dataset = dataset_path or Path("configs/datasets/lambada.sample.jsonl")
        _, records = _load_smoke_dataset(dataset, max_samples)
        profile = default_quantization_registry().get(selection.quantization_name or "fp16")
        adapter = build_model_adapter(
            ModelSpec(backend="mock", name=manifest.model_name, revision=manifest.source_artifact.hf_revision),
            profile,
        )

        accumulator = MetricsAccumulator()
        rng_seed = int(time.time()) ^ seed
        import random

        rng = random.Random(rng_seed)
        for record in records:
            prediction = adapter.predict(record, rng)
            accumulator.add(record, prediction)

        total_time = accumulator.token_count / profile.speed_tps if profile.speed_tps else 0.0
        metrics = accumulator.summary(total_time, profile.vram_gb)
        row = _build_success_row(
            metadata,
            manifest,
            manifest_path,
            selection,
            profile,
            len(records),
            metrics.perplexity,
        )
        payload = {
            "run": {
                "run_id": metadata.run_id,
                "run_name": metadata.run_name,
                "timestamp": metadata.timestamp,
                "gpu": metadata.telemetry.system.gpu_names,
                "cuda": metadata.telemetry.system.cuda_version,
            },
            "result": row,
        }

        run_id = metadata.run_id
        if "json" in formats:
            write_json(output_dir / "json" / f"{run_id}.artifact-smoke.json", payload)
        if "csv" in formats:
            write_csv(output_dir / "csv" / f"{run_id}.artifact-smoke.csv", [row])
        return payload
    except Exception as exc:
        failed = ArtifactSelection(
            status="failed",
            source_format=selection.source_format,
            runtime_format=selection.runtime_format,
            quantization_name=selection.quantization_name,
            generated_format=selection.generated_format,
            artifact_path=selection.artifact_path,
            failure_reason=str(exc),
        )
        return _write_failure_outputs(output_dir, formats, metadata, manifest, manifest_path, failed)
