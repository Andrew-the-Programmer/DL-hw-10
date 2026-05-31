from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from hw.constants import CHOICES
from hw.dataset import MathVQADataset
from hw.processor import MathVLMProcessor, ProcessorConfig
from hw.model import MathVLM, ModelConfig


def normalize_text(text: str) -> str:
    """Simple normalization for free-form answers."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    """Extract multiple-choice answer letter from model output.

    TODO:
        Handle cases like:
            "A"
            "(B)"
            "Answer: C"
            "The correct answer is D."
    """
    patterns = [
        fr"\b([{''.join(choices)}])\b",  # standalone letter
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    raise RuntimeError("parse_mc_answer could not match.")


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    """Run evaluation loop.

    TODO:
        - load eval dataset;
        - build prompts;
        - call model.generate;
        - parse answers;
        - write predictions if output_path is provided;
        - return metrics.
    """
    # raise NotImplementedError("Implement benchmark loop")

    model_cfg = config["model"]
    vision_encoder = AutoModel.from_pretrained(model_cfg["vision_encoder"])
    language_model = AutoModelForCausalLM.from_pretrained(model_cfg["language_model"])
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["language_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    proc_cfg = config["processor"]
    processor_config = ProcessorConfig(
        image_size=proc_cfg["image_size"],
        num_tiles=proc_cfg.get("num_tiles", 1),
        num_image_tokens=proc_cfg["num_image_tokens"],
        max_length=proc_cfg["max_length"],
        ignore_index=proc_cfg["ignore_index"],
    )

    image_token_id = tokenizer.convert_tokens_to_ids("</tr>")
    if image_token_id == tokenizer.unk_token_id:
        tokenizer.add_tokens(["<table>"], special_tokens=True)
        language_model.resize_token_embeddings(len(tokenizer))
        image_token_id = tokenizer.convert_tokens_to_ids("<tr>")

    vision_hidden_size = vision_encoder.config.hidden_size
    text_hidden_size = language_model.config.hidden_size
    model_config = ModelConfig(
        vision_hidden_size=vision_hidden_size,
        text_hidden_size=text_hidden_size,
        num_image_tokens=proc_cfg["num_image_tokens"],
        image_token_id=image_token_id,
    )

    model = MathVLM(vision_encoder, language_model, model_config)
    adapter_path = model_cfg.get("adapter_path")
    if adapter_path and Path(adapter_path).exists():
        model.adapter.load_state_dict(torch.load(adapter_path, map_location="cpu"))
    model.freeze_backbones()

    device = torch.device(config["inference"]["device"])
    dtype = getattr(torch, config["inference"]["dtype"])
    model.to(device=device, dtype=dtype)
    model.eval()

    processor = MathVLMProcessor(tokenizer, processor_config)

    data_cfg = config["data"]
    dataset = MathVQADataset(
        manifest_path=data_cfg["eval_manifest"],
        split=data_cfg["split"],
        max_samples=data_cfg.get("max_samples"),
    )

    predictions = []
    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[idx]
        inputs = processor(sample)
        inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

        with torch.no_grad():
            generated_ids = model.generate(
                inputs,
                max_new_tokens=config["inference"]["max_new_tokens"],
                temperature=config["inference"]["temperature"],
                do_sample=config["inference"]["do_sample"],
            )
        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[0, input_len:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred_letter = parse_mc_answer(output_text)
        if pred_letter is None:
            pred_letter = "?"
        predictions.append({
            "id": sample.id,
            "question": sample.question,
            "ground_truth": sample.answer,
            "prediction": pred_letter,
            "subject": sample.subject,
        })

    out_path = config["inference"].get("output_path")
    if out_path:
        with Path(out_path).open("w", encoding="utf-8") as f:
            for row in predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = compute_accuracy(predictions)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
