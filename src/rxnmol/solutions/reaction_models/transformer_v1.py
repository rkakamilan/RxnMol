"""Adapter for the legacy reaction transformer (reaction_model.py)."""

from pathlib import Path
import logging
import os
from typing import List

from .base import ReactionModel

logger = logging.getLogger(__name__)


class TransformerV1Model(ReactionModel):
    """Adapter for rxnmol.solutions.reaction_model.RxnPredictor."""

    def __init__(self, config, device: str):
        from ..reaction_model import RxnPredictor

        model_dir = _resolve_model_dir(config)
        if config.checkpoint_path or config.tokenizer_path:
            logger.debug(
                "TransformerV1Model ignores checkpoint_path/tokenizer_path; "
                "use model_dir to point at the legacy model bundle."
            )

        kwargs = {
            "model_dir": str(model_dir),
            "device": device,
            "enforce_cuda": getattr(config, "enforce_cuda", True),
        }
        if getattr(config, "batch_size", None) is not None:
            kwargs["batch_size"] = config.batch_size

        self.predictor = RxnPredictor(**kwargs)

    def predict(self, input_text: str) -> str:
        return self.predictor.predict(input_text)

    def predict_batch(self, input_texts: List[str]) -> List[str]:
        return self.predictor.predict_batch(input_texts)


def _resolve_model_dir(config) -> Path:
    if getattr(config, "model_dir", None):
        return Path(config.model_dir)

    env_model_dir = os.environ.get("RXNMOL_MODEL_DIR")
    if env_model_dir:
        return Path(env_model_dir)

    root = Path(__file__).resolve().parents[4]
    return root / "rxn_smiles_mit"
