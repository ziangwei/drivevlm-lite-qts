from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print("Training entry point is scaffolded.")
    print("Next implementation step: wire TRL SFTTrainer + PEFT LoRA using this config.")
    print(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
