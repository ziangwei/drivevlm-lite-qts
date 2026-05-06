from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", default=7860, type=int)
    args = parser.parse_args()

    print("Gradio demo entry point is scaffolded.")
    print(f"model={args.model}")
    print(f"server={args.server_name}:{args.server_port}")


if __name__ == "__main__":
    main()
