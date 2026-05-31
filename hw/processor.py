from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample

import torch.nn.functional as F

import numpy as np


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size].

        TODO:
            - convert to RGB;
            - resize/crop/pad;
            - split into tiles if num_tiles > 1;
            - normalize to float tensor.
        """
        # raise NotImplementedError("Implement image preprocessing")
        image = image.convert("RGB")
        n = int(self.config.num_tiles**0.5)
        image_size = self.config.image_size
        full_size = n * image_size
        image = image.resize((full_size, full_size), Image.Resampling.BILINEAR)
        img_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(img_np)
        image_tensor = image_tensor.view(n, image_size, n, image_size, 3)
        image_tensor = image_tensor.permute(0, 2, 1, 3, 4)
        image_tensor = image_tensor.reshape(
            self.config.num_tiles, image_size, image_size, 3
        )
        image_tensor = image_tensor.permute(0, 3, 1, 2)
        return image_tensor

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build a text prompt WITHOUT visual tokens (they will be inserted later)."""
        options_text = "\n".join(sample.options)
        prompt = f"Question: {sample.question}\nOptions:\n{options_text}\nAnswer:"
        if include_answer:
            prompt += f" {sample.answer}"
        return prompt

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels with explicit visual token insertion."""
        prompt_text = self.build_prompt(sample, include_answer=False)
        full_text = self.build_prompt(sample, include_answer=True)

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        if image_token_id == self.tokenizer.unk_token_id:
            raise RuntimeError(f"Image token {IMAGE_TOKEN} not in tokenizer vocabulary")

        visual_ids = [image_token_id] * self.config.num_image_tokens
        full_ids = visual_ids + full_ids
        prompt_ids = visual_ids + prompt_ids

        prompt_len = len(prompt_ids)
        labels = full_ids.copy()
        labels[:prompt_len] = [self.config.ignore_index] * prompt_len

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values.

        TODO:
            - pad input_ids with tokenizer.pad_token_id;
            - pad attention_mask with 0;
            - pad labels with ignore_index;
            - stack pixel_values into [B, T, 3, H, W].
        """
        # raise NotImplementedError("Implement collate_fn")

        max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids_pad = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for item in batch:
            seq_len = item["input_ids"].size(0)
            pad_len = max_len - seq_len

            input_ids = F.pad(item["input_ids"], (0, pad_len), value=input_ids_pad)
            padded_input_ids.append(input_ids)

            attn_mask = F.pad(item["attention_mask"], (0, pad_len), value=0)
            padded_attention_mask.append(attn_mask)

            labels = F.pad(item["labels"], (0, pad_len), value=self.config.ignore_index)
            padded_labels.append(labels)

        return {
            "input_ids": torch.stack(padded_input_ids, dim=0),
            "attention_mask": torch.stack(padded_attention_mask, dim=0),
            "labels": torch.stack(padded_labels, dim=0),
            "pixel_values": torch.stack(
                [item["pixel_values"] for item in batch], dim=0
            ),
        }
