from __future__ import annotations

import importlib.util
import platform


def exists(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def main() -> None:
    print(f"Python: {platform.python_version()}")
    for module in ["torch", "transformers", "trl", "peft", "datasets", "accelerate", "gradio"]:
        print(f"{module}: {'ok' if exists(module) else 'missing'}")

    if exists("torch"):
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
