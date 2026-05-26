from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class ArtifactFormat(str, Enum):
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    HF = "hf"
    AWQ = "awq"
    GPTQ = "gptq"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    MYELIN = "myelin"


class ArtifactStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    PLANNED = "planned"
    SKIPPED = "skipped"


class SourceArtifact(BaseModel):
    format: ArtifactFormat
    path: str | None = None
    hf_repo_id: str | None = None
    hf_revision: str | None = None
    url: str | None = None
    checksum_sha256: str | None = Field(None, alias="checksum_sha256")
    parameter_count: int | None = None
    moe_layout: dict[str, Any] | None = None
    notes: str | None = None


class GeneratedArtifact(BaseModel):
    format: ArtifactFormat
    status: ArtifactStatus
    path: str | None = None
    checksum_sha256: str | None = Field(None, alias="checksum_sha256")
    quantization_method: str | None = None
    calibration_dataset: str | None = None
    bits: int | None = None
    group_size: int | None = None
    backend_compatibility: list[str] | None = None
    notes: str | None = None


class BackendCompatibility(BaseModel):
    gguf: bool = False
    awq: bool = False
    gptq: bool = False
    myelin_accelerator: bool = False


class SAAQMetadata(BaseModel):
    routing_entropy: float | None = None
    spike_density: float | None = None
    experiment_id: str | None = None


class BenchmarkLinkage(BaseModel):
    nfl_combine_run_id: str | None = None
    nfl_combine_config_path: str | None = None


class ModelManifest(BaseModel):
    manifest_version: str = "1.0.0"
    model_name: str
    model_family: str | None = None
    source_artifact: SourceArtifact
    generated_artifacts: list[GeneratedArtifact] = []
    backend_compatibility: BackendCompatibility | None = None
    saaq_metadata: SAAQMetadata | None = None
    benchmark_linkage: BenchmarkLinkage | None = None


def load_manifest(path: Path) -> ModelManifest:
    """Load and validate a model manifest from a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return ModelManifest.model_validate(raw)


def load_manifest_from_string(text: str) -> ModelManifest:
    """Load and validate a model manifest from a JSON string."""
    raw = json.loads(text)
    return ModelManifest.model_validate(raw)


def dispatch_artifact(manifest: ModelManifest) -> str:
    """Return a dispatch tag based on the source artifact format."""
    if manifest.generated_artifacts:
        first_gen = manifest.generated_artifacts[0]
        if first_gen.status in (ArtifactStatus.SUCCESS, ArtifactStatus.PARTIAL, ArtifactStatus.PLANNED):
            return f"generated_{first_gen.format.value}"
    source_format = manifest.source_artifact.format
    if source_format == ArtifactFormat.GGUF:
        return "gguf"
    if source_format in (ArtifactFormat.SAFETENSORS, ArtifactFormat.HF):
        return "safetensors_hf"
    return "unknown"


__all__ = [
    "ArtifactFormat",
    "ArtifactStatus",
    "SourceArtifact",
    "GeneratedArtifact",
    "BackendCompatibility",
    "SAAQMetadata",
    "BenchmarkLinkage",
    "ModelManifest",
    "load_manifest",
    "load_manifest_from_string",
    "dispatch_artifact",
    "ValidationError",
]
