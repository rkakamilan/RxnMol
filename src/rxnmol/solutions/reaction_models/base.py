"""Reaction model interface for CSA integration."""

from abc import ABC, abstractmethod
from typing import List


class ReactionModel(ABC):
    """Minimal interface expected by CSA solution specs."""

    @abstractmethod
    def predict(self, input_text: str) -> str:
        """Predict a product for a single reaction input."""
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, input_texts: List[str]) -> List[str]:
        """Predict products for a batch of reaction inputs."""
        raise NotImplementedError
