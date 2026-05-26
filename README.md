# NFL-combine-for-AI

Neutral benchmark harness for AI model quantization experiments.

Owns:
- evaluation scripts
- benchmark configs
- metric collection
- CSV/JSON/Markdown reports

Does not own:
- quantization kernels
- model-specific quantization recipes

## Quickstart

Run the sample benchmark harness with the mock backend:

```bash
python scripts/benchmark.py --config configs/benchmark.sample.json
```

Reports are written to `reports/json` and `reports/csv` by default.

Run a single-command artifact smoke benchmark from a `magere-brug` manifest:

```bash
./scripts/run_smoke_benchmark.sh configs/manifests/safetensors_hf.sample.json
```
