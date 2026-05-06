# Conda Environment

Use one main server environment for training and evaluation. Keep the local laptop environment separate and lighter.

## Server Training Environment

Target hardware:

- Linux server.
- 1x H100 80GB preferred.
- CUDA 12.2 driver/module is acceptable.
- BF16 training.

Create the environment:

```bash
conda env create -f environment.yml
conda activate drivevlm-lite
python -m pip install -e ".[dev]"
python scripts/00_check_env.py
```

If your cluster uses CUDA modules and the available module is CUDA 12.2, that is fine:

```bash
module load cuda/12.2
```

The conda file still uses `pytorch-cuda=12.1` on purpose. PyTorch does not usually publish a separate conda selector for every CUDA minor version, and CUDA 12.1 runtime packages work on a CUDA 12.2-capable driver. The project pins `transformers>=4.57.0` because Qwen3-VL requires recent Transformers support. The base model config uses SDPA by default because it is reliable.

If the server has a newer driver and you prefer the newer PyTorch CUDA build, use the official PyTorch selector and switch to `pytorch-cuda=12.4` only after confirming it installs cleanly on the cluster. Do not use an unofficial CUDA 12.2 PyTorch package for this project.

Optional H100 speedup:

```bash
python -m pip install flash-attn --no-build-isolation
```

Only install `flash-attn` after PyTorch is installed and importable. If it fails to build because the cluster CUDA module and PyTorch CUDA runtime do not match exactly, keep using SDPA and continue.

## Why Python 3.10

Python 3.10 is the safest choice for CUDA ML stacks. It avoids edge-case incompatibilities that still appear with newer Python versions in training libraries, quantization packages, and compiled extensions.

## Package Roles

| Package | Role |
|---|---|
| `torch`, `torchvision`, `pytorch-cuda` | CUDA training/runtime |
| `transformers` | Qwen3-VL loading and generation |
| `trl` | SFT / later DPO trainer |
| `peft` | LoRA adapters |
| `accelerate` | single/multi-GPU launch |
| `bitsandbytes` | QLoRA / low-memory experiments |
| `qwen-vl-utils[decord]` | Qwen-VL image/video preprocessing helpers |
| `datasets` | JSONL and HF dataset loading |
| `gradio` | demo |
| `matplotlib`, `seaborn`, `scikit-learn` | reports and metrics |

## Expected Checks

After activating the environment:

```bash
python scripts/00_check_env.py
```

Expected on the H100 server:

```text
torch: ok
transformers: ok
trl: ok
peft: ok
datasets: ok
accelerate: ok
gradio: ok
cuda available: True
gpu: NVIDIA H100 ...
```

## Local Demo Environment

The RTX 5070 8GB laptop should not be used for training. Use it only for a small Gradio demo.

For local demo, prefer an INT4/AWQ/GGUF deployment path after the server model is ready. Do not solve local quantized deployment before the server baseline and LoRA SFT are complete.

Minimum local needs:

- Python 3.10.
- Gradio.
- A quantized model runtime.
- A few demo images.

Keep full training data and checkpoints on the server.
