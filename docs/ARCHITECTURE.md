# Architecture

## Manifest Ingestion

`NFL-combine-for-AI` consumes model manifests produced by `magere-brug`.

### Manifest format (JSON)

- `manifest_version`: fixed `"1.0.0"`
- `model_name`: display name of the model
- `model_family`: optional family tag (e.g. `llama2`, `llama3`, `grok`)
- `source_artifact`: describes the original/local artifact
  - `format`: `gguf`, `safetensors`, `hf`, `pytorch`, `onnx`, `myelin`
  - `path`: local filesystem path
  - `hf_repo_id` / `hf_revision`: HuggingFace source
  - `url`: generic remote URL
  - `checksum_sha256`: optional integrity hash
  - `parameter_count`: optional parameter count
  - `moe_layout`: optional MoE metadata dict
- `generated_artifacts`: list of conversion/quantization results
  - `format`: target format
  - `status`: `success`, `failed`, `partial`, `planned`, `skipped`
  - `path`, `checksum_sha256`, `quantization_method`, `calibration_dataset`, `bits`, `group_size`
  - `backend_compatibility`: list of backends (e.g. `["llama.cpp", "vllm", "myelin-accelerator"]`)
- `backend_compatibility`: boolean flags for `gguf`, `awq`, `gptq`, `myelin_accelerator`
- `saaq_metadata`: optional experiment metadata for SAAQ/routing research
  - `routing_entropy`, `spike_density`, `experiment_id`
- `benchmark_linkage`: optional link to an `NFL-combine-for-AI` run
  - `nfl_combine_run_id`, `nfl_combine_config_path`

### Dispatch logic

`dispatch_artifact(manifest)` returns a routing tag used by the benchmark harness:

1. If a `generated_artifact` exists with status `success`, `partial`, or `planned`, tag is `generated_<format>`.
2. Otherwise, tag by source format: `gguf` or `safetensors_hf`.
3. Fallback: `unknown`.

### Validation

`load_manifest(path)` and `load_manifest_from_string(text)` use Pydantic to validate schema. Invalid manifests raise `pydantic.ValidationError` with a clear message.

### Example manifests

Committed examples live in `configs/manifests/`:
- `gguf.sample.json` — local GGUF artifact
- `safetensors_hf.sample.json` — HF checkout with planned AWQ/GPTQ
- `grok_planning.sample.json` — future MoE + SAAQ planning

## Telemetry and Hardware Metrics

The benchmark harness collects cross-platform hardware telemetry at the start of every run.

### `benchmarks/telemetry.py`

- `SystemSnapshot` — CPU counts, memory, GPU names/driver, platform, Python version
- `GPUMetrics` — per-GPU utilization, memory, temperature, power, clocks (via `pynvml`, optional)
- `RoutingMetrics` — neuromorphic/SAAQ fields: `routing_entropy`, `spike_density`, `latent_stability`, `dv_dt_reductions`, `event_rate`
- `TelemetrySnapshot` — combines system, GPU, and routing metrics into one artifact

All GPU collection is done through `pynvml` / `nvidia-ml-py`. No CUDA kernels live in this repo.

### Upstream artifact compatibility

- `CorinthCanalArtifact` — consumes SAAQ / telemetry JSON files from `corinth-canal`
- `MyelinAcceleratorArtifact` — consumes benchmark JSON files from `myelin-accelerator`
- `merge_upstream_artifacts(base, corinth=..., myelin=...)` — overlays upstream metrics onto the local telemetry snapshot

A benchmark config may reference upstream artifacts:

```json
{
  "telemetry": {
    "corinth_canal_path": "configs/telemetry/corinth.json",
    "myelin_accelerator_path": "configs/telemetry/myelin.json"
  }
}
```

### Report output

Telemetry fields are flattened with a `telemetry_` prefix in both JSON and CSV reports:
- `telemetry_sys_cpu_count_logical`, `telemetry_sys_memory_total_gb`, etc.
- `telemetry_routing_entropy`, `telemetry_spike_density`, etc.
- `telemetry_kernel_occupancy`, `telemetry_vram_bandwidth_gbps`

A dedicated `reports/telemetry/<run_id>.telemetry.json` file is also emitted per run.

## Artifact Smoke Run

Issue #3 adds a single-command manifest-driven smoke benchmark path for Vultr sprint validation.

### Entrypoints

- Shell wrapper: `scripts/run_smoke_benchmark.sh <manifest-path> [extra args...]`
- Python CLI: `scripts/run_artifact_smoke.py --manifest <path>`

Example:

```bash
./scripts/run_smoke_benchmark.sh \
  configs/manifests/safetensors_hf.sample.json \
  --output-dir /tmp/artifact-smoke
```

### Dispatch rules

- Prefer generated `AWQ` / `GPTQ` artifacts only when status is `success` or `partial` and the local artifact path exists.
- Otherwise run `GGUF` sources when the local path exists.
- Otherwise run `HF` / `Safetensors` sources when a local path exists or `hf_repo_id` is present.
- Unsupported or missing artifacts produce a structured failure report instead of crashing without output.

### Outputs

- JSON: `reports/json/<run_id>.artifact-smoke.json`
- CSV: `reports/csv/<run_id>.artifact-smoke.csv`

Each smoke report includes:

- `model_id`
- `source_format`
- `runtime_format`
- `generated_format`
- `quantization`
- `gpu`
- `cuda`
- `throughput_tps`
- `peak_vram_gb`
- `perplexity`
- `failure_reason` when the smoke run cannot execute

## CI / Actions

See `.github/workflows/`:
- `ci.yml` — pytest + smoke benchmark help on every PR/push
- `manifest-ingestion.yml` — validates example manifests and dispatch logic on PRs
- `mini-eval-smoke.yml` — runs tiny fake benchmark with dummy data on PRs
- `benchmark-smoke.yml` — manual-only workflow to run a small benchmark and upload artifacts
