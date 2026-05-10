from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.data.jsonl import read_jsonl


def _message_text(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages", []):
        if message.get("role") == role:
            return str(message.get("content", ""))
    return ""


class VLMSFTDataset(Dataset):
    def __init__(self, path: Path, limit: int = 0):
        rows = read_jsonl(path)
        self.rows = rows[:limit] if limit > 0 else rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        return {
            "sample_id": row.get("sample_id"),
            "images": row.get("images", []),
            "question": _message_text(row, "user"),
            "answer": _message_text(row, "assistant"),
        }


@dataclass
class VLMDataCollator:
    processor: Any

    def _load_images(self, paths: list[str]) -> list[Image.Image]:
        return [Image.open(path).convert("RGB") for path in paths]

    def _messages(
        self,
        question: str,
        answer: str | None,
        images: list[Image.Image],
        add_answer: bool,
    ) -> list[dict[str, Any]]:
        user_content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
        user_content.append({"type": "text", "text": question})
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
        if add_answer and answer is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                }
            )
        return messages

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if len(features) != 1:
            raise ValueError("This first SFT implementation expects per_device_train_batch_size=1.")

        feature = features[0]
        images = self._load_images(feature["images"])
        full_messages = self._messages(feature["question"], feature["answer"], images, add_answer=True)
        prompt_messages = self._messages(feature["question"], None, images, add_answer=False)

        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_inputs = self.processor(text=[full_text], images=images, return_tensors="pt")
        prompt_inputs = self.processor(text=[prompt_text], images=images, return_tensors="pt")

        labels = full_inputs["input_ids"].clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:, :prompt_len] = -100
        if "attention_mask" in full_inputs:
            labels[full_inputs["attention_mask"] == 0] = -100

        full_inputs["labels"] = labels
        return full_inputs


def _training_args(config: dict[str, Any], output_dir: Path):
    from transformers import TrainingArguments

    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": config.get("num_train_epochs", 1),
        "per_device_train_batch_size": config.get("per_device_train_batch_size", 1),
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 16),
        "learning_rate": config.get("learning_rate", 1e-4),
        "warmup_ratio": config.get("warmup_ratio", 0.03),
        "bf16": bool(config.get("bf16", True) and torch.cuda.is_available()),
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
        "logging_steps": config.get("logging_steps", 10),
        "save_steps": config.get("save_steps", 500),
        "save_total_limit": config.get("save_total_limit", 2),
        "remove_unused_columns": False,
        "report_to": [],
        "dataloader_num_workers": 0,
        "optim": config.get("optim", "adamw_torch"),
    }
    eval_steps = config.get("eval_steps", 0)
    if eval_steps:
        kwargs["eval_steps"] = eval_steps
        try:
            return TrainingArguments(eval_strategy="steps", **kwargs)
        except TypeError:
            return TrainingArguments(evaluation_strategy="steps", **kwargs)
    try:
        return TrainingArguments(eval_strategy="no", **kwargs)
    except TypeError:
        return TrainingArguments(evaluation_strategy="no", **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--train-file", default=None, type=Path)
    parser.add_argument("--eval-file", default=None, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--max-train-samples", default=0, type=int)
    parser.add_argument("--max-eval-samples", default=0, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=None, type=int)
    parser.add_argument("--learning-rate", default=None, type=float)
    parser.add_argument("--num-train-epochs", default=None, type=float)
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--dry-run-collator", action="store_true")
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.gradient_accumulation_steps is not None:
        config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.num_train_epochs is not None:
        config["num_train_epochs"] = args.num_train_epochs
    if args.gradient_checkpointing is not None:
        config["gradient_checkpointing"] = args.gradient_checkpointing

    model_config_path = Path(config["model_config"])
    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))

    model_name = args.model or model_config["model_name_or_path"]
    train_file = args.train_file or Path(config["train_file"])
    eval_file = args.eval_file or Path(config["eval_file"])
    output_dir = args.output_dir or Path(config["output_dir"])

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and config.get("bf16", True) else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        dtype=dtype,
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False

    if config.get("use_lora", True):
        lora_config = LoraConfig(
            r=config.get("lora_rank", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=config.get("lora_target_modules"),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_dataset = VLMSFTDataset(train_file, limit=args.max_train_samples)
    eval_dataset = VLMSFTDataset(eval_file, limit=args.max_eval_samples)
    collator = VLMDataCollator(processor)
    print(f"train_file={train_file}")
    print(f"eval_file={eval_file}")
    print(f"requested_max_train_samples={args.max_train_samples}")
    print(f"actual_train_samples={len(train_dataset)}")
    print(f"actual_eval_samples={len(eval_dataset)}")
    print(f"gradient_accumulation_steps={config.get('gradient_accumulation_steps', 16)}")
    print(f"learning_rate={config.get('learning_rate', 1e-4)}")
    print(f"gradient_checkpointing={config.get('gradient_checkpointing', True)}")
    if args.max_train_samples > 0 and len(train_dataset) < args.max_train_samples:
        print(
            "WARNING: requested max train samples is larger than the train JSONL. "
            "Regenerate the selected training JSONL for a larger run."
        )

    if args.dry_run_collator:
        batch = collator([train_dataset[0]])
        print({key: tuple(value.shape) for key, value in batch.items() if hasattr(value, "shape")})
        return

    trainer = Trainer(
        model=model,
        args=_training_args(config, output_dir),
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) else None,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"Saved LoRA SFT output to {output_dir}")


if __name__ == "__main__":
    main()
