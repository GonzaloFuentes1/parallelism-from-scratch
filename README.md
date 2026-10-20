# parallelism-from-scratch

[![ci](https://github.com/GonzaloFuentes1/parallelism-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/GonzaloFuentes1/parallelism-from-scratch/actions/workflows/ci.yml)

Readable DDP, ZeRO, tensor and pipeline parallelism for a tiny Llama, with exact parity tests on CPU and DCP resharding.

## Development

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```
