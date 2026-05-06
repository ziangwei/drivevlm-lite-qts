from __future__ import annotations

import importlib.util
import platform


def exists(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def main() -> None:
    print(f"Python: {platform.python_version()}")
    required = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("huggingface_hub", "huggingface_hub"),
        ("PIL", "pillow"),
        ("yaml", "pyyaml"),
        ("qwen_vl_utils", "qwen-vl-utils"),
    ]
    optional = [
        ("accelerate", "accelerate"),
        ("trl", "trl"),
        ("peft", "peft"),
        ("bitsandbytes", "bitsandbytes"),
        ("gradio", "gradio"),
        ("wandb", "wandb"),
    ]

    print("E0 required packages:")
    for module, package in required:
        print(f"{package}: {'ok' if exists(module) else 'missing'}")

    print("Later-stage optional packages:")
    for module, package in optional:
        print(f"{package}: {'ok' if exists(module) else 'missing'}")

    if exists("torch"):
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
