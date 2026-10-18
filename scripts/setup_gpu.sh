#!/usr/bin/env bash
# One-shot setup on a fresh Ubuntu cloud GPU box. Run from the repo root:  bash scripts/setup_gpu.sh
# Env: FLASH_ATTN=1 to also build flash-attn (only useful with --no-pack); CUDA_WHEEL=cu124 (default cu128).
set -euo pipefail
cd "$(dirname "$0")/.."

nvidia-smi
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install "torch==2.14.1" --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL:-cu128}"
# Pinned to the versions the packing/attention-mask test passed with locally.
uv pip install "transformers==5.18.0" "peft==0.21.2" "accelerate==1.15.0" datasets huggingface_hub hf_transfer fastapi uvicorn pytest tqdm
uv pip install -e . --no-deps
if [ "${FLASH_ATTN:-0}" = "1" ]; then uv pip install flash-attn --no-build-isolation || echo "flash-attn failed (optional)"; fi

export HF_HUB_ENABLE_HF_TRANSFER=1
for m in Qwen/Qwen3-1.7B-Base Qwen/Qwen3-4B-Base; do
  python -c "from huggingface_hub import snapshot_download as d; print(d('$m', allow_patterns=['*.json','*.safetensors','*.txt','*.model','merges.txt']))"
done

python - <<'EOF'
import torch, transformers, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__, "peft", peft.__version__)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name} {p.total_memory / 2**30:.0f} GiB, bf16={torch.cuda.is_bf16_supported()}")
EOF
python -m pytest tests/test_serialize.py -q
echo "setup done: source .venv/bin/activate"
