"""Torch device 선택 helper."""

from __future__ import annotations

import torch


def resolve_torch_device(value: str) -> torch.device:
    """CLI/API device 설정값을 실제 torch.device로 변환한다."""
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        return torch.device("cuda")
    if value == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
