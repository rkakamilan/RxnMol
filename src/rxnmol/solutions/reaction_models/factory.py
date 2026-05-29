"""Factory for reaction model adapters."""

import logging
from typing import Optional

import torch

from .transformer_v1 import TransformerV1Model
from .transformer_v2 import TransformerV2Model

logger = logging.getLogger(__name__)


def load_reaction_model(config, device: Optional[str] = None):
    """Load a reaction model adapter based on config."""
    device = device or getattr(config, "device", None) or "cuda"
    device = _resolve_device(device)

    provider = getattr(config, "provider", "transformer_v1")
    provider = provider.strip() if isinstance(provider, str) else provider

    if provider == "transformer_v1":
        return TransformerV1Model(config, device)
    if provider == "transformer_v2":
        return TransformerV2Model(config, device)

    raise ValueError(f"Unsupported reaction_model.provider: {provider}")


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")
        return "cpu"
    return device
