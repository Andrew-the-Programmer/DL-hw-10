from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class ModelConfig:
    vision_hidden_size: int
    text_hidden_size: int
    num_image_tokens: int
    image_token_id: int


class VisionToTextAdapter(nn.Module):
    """Maps vision encoder hidden states to LLM embedding space."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        num_image_tokens: int,
    ) -> None:
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.num_image_tokens = num_image_tokens

        # TODO: replace with a small projection network.
        # Recommended: LayerNorm -> Linear -> GELU -> Linear.
        # raise NotImplementedError("Implement VisionToTextAdapter.__init__")

        self.pool = nn.AdaptiveAvgPool1d(num_image_tokens)
        self.mlp = nn.Sequential(
            nn.LayerNorm(vision_hidden_size),
            nn.Linear(vision_hidden_size, vision_hidden_size * 2),
            nn.GELU(),
            nn.Linear(vision_hidden_size * 2, text_hidden_size),
        )

    def forward(self, vision_hidden_states: torch.Tensor) -> torch.Tensor:
        """Return visual embeddings [B, num_image_tokens, text_hidden_size].
        vision_hidden_states: [B, L, D]
        """
        # NOTE: missed TODO
        # raise NotImplementedError("Implement VisionToTextAdapter.forward")

        x = vision_hidden_states.permute(0, 2, 1)  # [B, D, L]
        x = self.pool(x)  # [B, D, num_image_tokens]
        x = x.permute(0, 2, 1)  # [B, num_image_tokens, D]
        x = self.mlp(x)  # [B, num_image_tokens, text_hidden_size]
        return x


def merge_visual_embeddings(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    visual_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Replace embeddings at <image> token positions with visual embeddings.

    Args:
        input_embeds: [B, L, D] text embeddings.
        input_ids: [B, L] token ids.
        visual_embeds: [B, K, D] visual embeddings.
        image_token_id: token id used as visual placeholder.

    Returns:
        Tensor [B, L, D] with visual embeddings inserted.

    Assumption for public tests:
        each row has exactly K positions where input_ids == image_token_id.
    """
    # NOTE: missed TODO
    # raise NotImplementedError("Implement visual/text embedding merge")

    B, L, D = input_embeds.shape
    K = visual_embeds.shape[1]

    merged = input_embeds.clone()

    for b in range(B):
        mask = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]
        mask = mask[:K]
        merged[b, mask] = visual_embeds[b]

    return merged


class MathVLM(nn.Module):
    """Thin wrapper around vision encoder, adapter and language model.

    In Track A/B, vision encoder and LLM should be frozen; adapter trainable.
    """

    def __init__(
        self, vision_encoder: nn.Module, language_model: nn.Module, config: ModelConfig
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.config = config
        self.adapter = VisionToTextAdapter(
            vision_hidden_size=config.vision_hidden_size,
            text_hidden_size=config.text_hidden_size,
            num_image_tokens=config.num_image_tokens,
        )

    def freeze_backbones(self) -> None:
        """Freeze vision encoder and language model parameters."""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = False

    def get_merged_embeds(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """helper function."""

        B, T, C, H, W = batch["pixel_values"].shape
        pixel_values = batch["pixel_values"].view(B * T, C, H, W)

        vision_outputs = self.vision_encoder(pixel_values)
        vision_hidden = vision_outputs.last_hidden_state  # [B*T, L, D]

        vision_hidden = vision_hidden.view(
            B, T, -1, self.config.vision_hidden_size
        )  # [B, T, L, D]
        vision_hidden = vision_hidden.mean(dim=1)  # [B, L, D]

        visual_embeds = self.adapter(
            vision_hidden
        )  # [B, num_image_tokens, text_hidden_size]
        input_embeds = self.language_model.get_input_embeddings()(batch["input_ids"])

        merged_embeds = merge_visual_embeddings(
            input_embeds, batch["input_ids"], visual_embeds, self.config.image_token_id
        )
        return merged_embeds

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        """Forward pass with loss.

        TODO:
            - encode images;
            - map to visual embeddings;
            - get text input embeddings;
            - merge visual/text embeddings;
            - call language_model with inputs_embeds, attention_mask, labels.
        """
        # NOTE: missed TODO
        # raise NotImplementedError("Implement MathVLM.forward")

        merged_embeds = self.get_merged_embeds(batch)

        outputs = self.language_model(
            inputs_embeds=merged_embeds,
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs

    @torch.no_grad()
    def generate(
        self, batch: dict[str, torch.Tensor], **generation_kwargs: Any
    ) -> torch.Tensor:
        """Generate answer token ids."""
        # NOTE: missed TODO
        # raise NotImplementedError("Implement MathVLM.generate")

        merged_embeds = self.get_merged_embeds(batch)

        default_kwargs = {"max_new_tokens": 64, "temperature": 0.0, "do_sample": False}
        default_kwargs.update(generation_kwargs)

        generated_ids = self.language_model.generate(
            inputs_embeds=merged_embeds,
            attention_mask=batch["attention_mask"],
            **default_kwargs,
        )
        return generated_ids
