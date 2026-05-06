# Conda Environment

Use one server environment for training and evaluation. Keep the local laptop
environment separate and lighter.

## Server Environment

Create a plain Python environment first:

```bash
conda create -n drivevlm-lite python=3.10 pip -y
conda activate drivevlm-lite
```

Install PyTorch with CUDA support:

```bash
conda install -y pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install the project dependencies:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
python scripts/00_check_env.py
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

If the cluster has a CUDA 12.2 module or driver, using `pytorch-cuda=12.1` is
still acceptable. PyTorch publishes conda runtime packages for common CUDA
targets such as 12.1 and 12.4, not every minor CUDA module version.

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

