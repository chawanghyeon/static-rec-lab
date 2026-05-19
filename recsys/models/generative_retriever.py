"""Transformer 기반 Generative Retrieval 모델."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import cast

import torch
from torch import nn

from recsys.models.dataset import BOS_TOKEN_ID, PAD_ITEM_INDEX, PAD_TOKEN_ID


@dataclass(frozen=True)
class GenerativeRetrieverConfig:
    """GenerativeRetriever 모델 설정."""

    item_vocab_size: int
    semantic_vocab_size: int
    semantic_id_length: int
    max_history_length: int = 50
    d_model: int = 64
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    pad_item_index: int = PAD_ITEM_INDEX
    pad_token_id: int = PAD_TOKEN_ID
    bos_token_id: int = BOS_TOKEN_ID

    def to_dict(self) -> dict[str, int | float]:
        """checkpoint 저장용 dict로 변환한다."""
        return asdict(self)


class GenerativeRetriever(nn.Module):
    """History item sequence를 target Semantic ID token sequence로 변환한다."""

    def __init__(self, config: GenerativeRetrieverConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.item_embedding = nn.Embedding(
            config.item_vocab_size,
            config.d_model,
            padding_idx=config.pad_item_index,
        )
        self.token_embedding = nn.Embedding(
            config.semantic_vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.history_position_embedding = nn.Embedding(
            config.max_history_length,
            config.d_model,
        )
        self.target_position_embedding = nn.Embedding(
            config.semantic_id_length,
            config.d_model,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers,
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers,
        )
        self.output_projection = nn.Linear(config.d_model, config.semantic_vocab_size)

    def forward(
        self,
        history_item_ids: torch.Tensor,
        history_padding_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Semantic token logits를 반환한다."""
        if history_item_ids.ndim != 2:
            msg = "history_item_ids는 [batch, history_length] tensor여야 합니다."
            raise ValueError(msg)
        if decoder_input_ids.ndim != 2:
            msg = "decoder_input_ids는 [batch, target_length] tensor여야 합니다."
            raise ValueError(msg)
        if history_item_ids.shape != history_padding_mask.shape:
            msg = "history_item_ids와 history_padding_mask shape가 같아야 합니다."
            raise ValueError(msg)
        history_length = history_item_ids.shape[1]
        target_length = decoder_input_ids.shape[1]
        if history_length > self.config.max_history_length:
            msg = (
                "history length가 model max_history_length를 초과했습니다: "
                f"{history_length} > {self.config.max_history_length}"
            )
            raise ValueError(msg)
        if target_length != self.config.semantic_id_length:
            msg = (
                "decoder target length가 semantic_id_length와 다릅니다: "
                f"{target_length} != {self.config.semantic_id_length}"
            )
            raise ValueError(msg)

        src = self.item_embedding(history_item_ids) + self._position_embeddings(
            self.history_position_embedding,
            batch_size=history_item_ids.shape[0],
            sequence_length=history_length,
            device=history_item_ids.device,
        )
        tgt = self.token_embedding(decoder_input_ids) + self._position_embeddings(
            self.target_position_embedding,
            batch_size=decoder_input_ids.shape[0],
            sequence_length=target_length,
            device=decoder_input_ids.device,
        )
        target_mask = _causal_mask(target_length, decoder_input_ids.device)
        memory = self.encoder(
            src,
            src_key_padding_mask=history_padding_mask,
        )
        hidden = self.decoder(
            tgt,
            memory,
            tgt_mask=target_mask,
            tgt_is_causal=True,
            memory_key_padding_mask=history_padding_mask,
        )
        return cast(torch.Tensor, self.output_projection(hidden))

    @staticmethod
    def _position_embeddings(
        embedding: nn.Embedding,
        *,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        positions = torch.arange(sequence_length, device=device).unsqueeze(0)
        return cast(torch.Tensor, embedding(positions).expand(batch_size, sequence_length, -1))


def _validate_config(config: GenerativeRetrieverConfig) -> None:
    if config.item_vocab_size <= config.pad_item_index:
        msg = "item_vocab_size가 pad_item_index보다 커야 합니다."
        raise ValueError(msg)
    if config.semantic_vocab_size <= max(config.pad_token_id, config.bos_token_id):
        msg = "semantic_vocab_size가 special token id보다 커야 합니다."
        raise ValueError(msg)
    if config.semantic_id_length < 1:
        msg = "semantic_id_length는 1 이상이어야 합니다."
        raise ValueError(msg)
    if config.max_history_length < 1:
        msg = "max_history_length는 1 이상이어야 합니다."
        raise ValueError(msg)
    if config.d_model < 1:
        msg = "d_model은 1 이상이어야 합니다."
        raise ValueError(msg)
    if config.d_model % config.num_heads != 0:
        msg = "d_model은 num_heads로 나누어 떨어져야 합니다."
        raise ValueError(msg)


def _causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.full((sequence_length, sequence_length), float("-inf"), device=device),
        diagonal=1,
    )
