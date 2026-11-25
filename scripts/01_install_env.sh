#!/usr/bin/env bash
# Create two isolated virtualenvs and install the pinned requirements.
#
#   .venv-quant : llmcompressor  -> 02_quantize.py
#   .venv-vllm  : vLLM           -> 03_perplexity.py / 04_prefill_decode.py / 05_vllm_profiler.py
#
# They must be separate: llmcompressor 0.15.0 pins compressed-tensors==0.20.1,
# while vllm 0.34.0 pins compressed-tensors==0.17.0.
#
# Usage:
#   bash scripts/01_install_env.sh            # install both
#   bash scripts/01_install_env.sh quant      # only the quantization env
#   bash scripts/01_install_env.sh vllm       # only the vLLM env
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root (requirements-*.txt and .venv-* live here)
TARGET="${1:-all}"

install_env () {
  local name="$1" req="$2"
  echo "=== Creating .venv-${name} from ${req} ==="
  python3 -m venv ".venv-${name}"
  ".venv-${name}/bin/python" -m pip install -U pip
  ".venv-${name}/bin/python" -m pip install -r "${req}"
}

case "${TARGET}" in
  quant) install_env quant requirements-quant.txt ;;
  vllm)  install_env vllm  requirements-vllm.txt  ;;
  all)   install_env quant requirements-quant.txt
         install_env vllm  requirements-vllm.txt  ;;
  *) echo "Usage: $0 [all|quant|vllm]"; exit 1 ;;
esac

echo
echo "=== Installed versions ==="
if [[ "${TARGET}" != "vllm" ]]; then
  .venv-quant/bin/python - <<'PY'
import torch, transformers, datasets, llmcompressor, compressed_tensors
print("[quant] torch", torch.__version__, "| transformers", transformers.__version__,
      "| datasets", datasets.__version__, "| llmcompressor", llmcompressor.__version__,
      "| compressed-tensors", compressed_tensors.__version__)
PY
fi
if [[ "${TARGET}" != "quant" ]]; then
  .venv-vllm/bin/python - <<'PY'
import torch, transformers, vllm, compressed_tensors
print("[vllm ] torch", torch.__version__, "| transformers", transformers.__version__,
      "| vllm", vllm.__version__, "| compressed-tensors", compressed_tensors.__version__)
PY
fi
