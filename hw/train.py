from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from hw.dataset import MathVQADataset
from hw.processor import MathVLMProcessor, ProcessorConfig
from hw.model import MathVLM, ModelConfig

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from torch.utils.data import Dataset, DataLoader


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one optimization step and return scalar loss.

    TODO:
        - model.train();
        - forward;
        - ensure finite loss;
        - backward;
        - optimizer.step();
        - optimizer.zero_grad();
    """
    # raise NotImplementedError("Implement train_one_step")

    model.train()
    optimizer.zero_grad()
    outputs = model(batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    return loss.item()


def run_training(config: dict[str, Any], fast_train: bool = False) -> None:
    """Main training entry point.

    TODO:
        - instantiate dataset, processor, model;
        - create DataLoader;
        - support max_steps and fast_train;
        - save adapter/checkpoint if configured.
    """
    # raise NotImplementedError("Implement run_training")

    trainer_cfg = config["trainer"]
    device = torch.device(trainer_cfg["device"])
    dtype = getattr(torch, trainer_cfg["dtype"])

    data_cfg = config["data"]
    dataset = MathVQADataset(
        manifest_path=data_cfg["train_manifest"],
        split=data_cfg["split"],
        max_samples=data_cfg.get("max_samples"),
    )

    model_cfg = config["model"]
    vision_encoder = AutoModel.from_pretrained(model_cfg["vision_encoder"])
    language_model = AutoModelForCausalLM.from_pretrained(model_cfg["language_model"])
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["language_model"])
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    image_token_id = tokenizer.convert_tokens_to_ids("</tr>")
    if image_token_id == tokenizer.unk_token_id:
        tokenizer.add_tokens(["<tr>"], special_tokens=True)
        language_model.resize_token_embeddings(len(tokenizer))
        image_token_id = tokenizer.convert_tokens_to_ids("<tr>")

    processor_cfg = config["processor"]
    processor_cfg = ProcessorConfig(
        image_size=processor_cfg["image_size"],
        num_tiles=processor_cfg.get("num_tiles", 1),
        tile_overlap=processor_cfg.get("tile_overlap", 0.0),
        num_image_tokens=processor_cfg["num_image_tokens"],
        max_length=processor_cfg["max_length"],
        ignore_index=processor_cfg["ignore_index"],
    )

    model_config = ModelConfig(
        vision_hidden_size=vision_encoder.config.hidden_size,
        text_hidden_size=language_model.config.hidden_size,
        num_image_tokens=processor_cfg.num_image_tokens,
        image_token_id=image_token_id,
    )

    model = MathVLM(vision_encoder, language_model, model_config)
    model.freeze_backbones()
    model.to(device=device, dtype=dtype)

    processor = MathVLMProcessor(tokenizer, processor_cfg)

    local_batch_size = trainer_cfg["local_batch_size"]
    dataloader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=True,
        collate_fn=processor.collate,
        num_workers=trainer_cfg.get("num_workers", 0),
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=trainer_cfg["learning_rate"],
        weight_decay=trainer_cfg["weight_decay"],
    )

    grad_accum_steps = (
        trainer_cfg.get("global_batch_size", local_batch_size)
        // local_batch_size
    )
    max_steps = trainer_cfg["max_steps"]
    if fast_train:
        max_steps = min(max_steps, 3)

    accumulated_loss = 0.0
    model.train()
    for step, batch in enumerate(dataloader):
        if step >= max_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        if "pixel_values" in batch:
            batch["pixel_values"] = batch["pixel_values"].to(dtype=dtype)

        loss_val = train_one_step(model, batch, optimizer)
        accumulated_loss += loss_val

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Step {step}/{max_steps}, loss: {loss_val:.4f}")

    save_checkpoint_path = trainer_cfg.get("save_checkpoint_path", None)
    if save_checkpoint_path is not None:
        save_checkpoint_path = Path(save_checkpoint_path)
        save_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.adapter.state_dict(), save_checkpoint_path)
        print(f"Saved adapter checkpoint to {save_checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()
