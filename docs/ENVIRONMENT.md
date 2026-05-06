# Conda Environment

Use one server environment for training and evaluation. Keep the local laptop
environment separate and lighter.

## Server Environment

Create a plain Python environment first:

```bash
conda create -n drivevlm-lite python=3.10 pip -y
conda activate drivevlm-lite
```

Install PyTorch with CUDA support using pip, not conda:

```bash
python -m pip install -U pip
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install the project dependencies:

```bash
python -m pip install -r requirements-e0.txt -c constraints-torch-cu121.txt
python -m pip install -e . --no-deps
python scripts/00_check_env.py
```

Do not use `python -m pip install -e ".[dev]"` on the server. It asks pip to
resolve dependencies from `pyproject.toml` and can replace the CUDA PyTorch
wheel. Use `--no-deps` for the editable project install.

Later milestones add their own requirements:

```bash
python -m pip install -r requirements-train.txt -c constraints-torch-cu121.txt
python -m pip install -r requirements-report.txt -c constraints-torch-cu121.txt
python -m pip install -r requirements-demo.txt -c constraints-torch-cu121.txt
```

If a broken environment already exists:

```bash
conda env remove -n drivevlm-lite
```

The `environment.yml` file is intentionally minimal. It only creates a Python
3.10 + pip environment:

```bash
conda env create -f environment.yml
```

For server work, the explicit commands above are preferred because failures are
easier to diagnose.

## CUDA Note

If the cluster has a CUDA 12.2 module or driver, using the PyTorch `cu121` wheel
is still acceptable. PyTorch publishes wheels for common CUDA runtime targets
such as 12.1 and 12.4, not every minor CUDA module version.

Use SDPA first. Install FlashAttention only after the baseline pipeline works:

```bash
python -m pip install flash-attn --no-build-isolation
```

## Expected Check

```bash
python scripts/00_check_env.py
```

Expected on the H100 server:

```text
E0 required packages:
torch: ok
transformers: ok
datasets: ok
huggingface_hub: ok
pillow: ok
pyyaml: ok
qwen-vl-utils: ok
cuda available: True
gpu: NVIDIA H100 ...
```

It is fine if `accelerate`, `trl`, `peft`, `bitsandbytes`, `gradio`, and
`wandb` are missing before the corresponding later milestone.
