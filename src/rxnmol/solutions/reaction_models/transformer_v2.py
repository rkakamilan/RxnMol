"""Adapter for the new transformer repo (tokenizer + transformer modules)."""

from pathlib import Path
import json
import logging
import os
import importlib.util
import sys
import types
from typing import List, Optional, Tuple

import torch

from .base import ReactionModel

logger = logging.getLogger(__name__)

# Minimum batch size to attempt before giving up
_MIN_BATCH_SIZE = 1
# Factor to reduce batch size on OOM
_OOM_REDUCTION_FACTOR = 0.6


class TransformerV2Model(ReactionModel):
    """Adapter for the new reaction transformer (tokenizer + transformer)."""

    def __init__(self, config, device: str):
        # repo_path is optional: if unset, the vendored architecture is used (self-contained).
        repo_path = _resolve_repo_path(config)

        self.device = device
        self.batch_size = config.batch_size if config.batch_size is not None else 128
        self.max_len = config.max_len or 256
        self.beam_size = config.beam_size or 1
        self.length_penalty = config.length_penalty or 1.0

        model_dir = _require_model_dir(config)
        tokenizer_path = _resolve_tokenizer_path(config, model_dir)
        training_config = _load_training_config(model_dir)

        if repo_path is not None:
            _add_repo_to_syspath(repo_path)
            modules = _load_external_modules(repo_path)
        else:
            modules = _load_vendored_modules()
        tokenizer_cls, transformer_cls, decoder_cls = modules

        self.tokenizer = tokenizer_cls.from_pretrained(str(tokenizer_path))
        self.task = config.task or training_config.get("tokenizer_config", {}).get("task", "forward")

        model_type = training_config.get("model_type", "transformer")
        self.model_type = model_type
        # model_config may be under either "model_config" or "model"
        model_config = training_config.get("model_config", {}) or training_config.get("model", {})

        # Ensure vocab + special tokens align with tokenizer.
        model_config = dict(model_config)
        model_config["vocab_size"] = getattr(self.tokenizer, "vocab_size", model_config.get("vocab_size"))
        model_config.setdefault("pad_id", getattr(self.tokenizer, "pad_id", 0))
        model_config.setdefault("bos_id", getattr(self.tokenizer, "bos_id", 1))
        model_config.setdefault("eos_id", getattr(self.tokenizer, "eos_id", 2))
        if "n_kv_heads" not in model_config and "n_heads" in model_config:
            model_config["n_kv_heads"] = model_config["n_heads"]
        if model_config.get("d_ff") is None and model_config.get("d_model") is not None:
            d_ff = int(8 * model_config["d_model"] / 3)
            model_config["d_ff"] = ((d_ff + 63) // 64) * 64

        model_path = _resolve_model_path(config, model_dir)
        try:
            state_obj = torch.load(model_path, map_location="cpu", weights_only=False)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Failed to load model checkpoint due to missing module in pickle. "
                "If your model.pt was saved with custom classes, prefer exporting a plain "
                "state_dict or set checkpoint_path to a Lightning checkpoint that includes "
                "state_dict only."
            ) from exc

        self.model = _build_model(
            model_type=self.model_type,
            model_config=model_config,
            transformer_cls=transformer_cls,
            decoder_cls=decoder_cls,
        )
        state_dict = _extract_state_dict(state_obj)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Store model config for batch size estimation
        self._model_config = model_config
        self._user_batch_size = config.batch_size  # None if auto
        self._auto_configure_batch_size(config)
        self._maybe_compile(config, device)
        self._log_tokenizer_debug()

    def predict(self, input_text: str) -> str:
        return self.predict_batch([input_text])[0]

    def predict_batch(self, input_texts: List[str]) -> List[str]:
        if not input_texts:
            return []

        all_predictions: List[str] = []
        current_batch_size = self.batch_size
        i = 0

        while i < len(input_texts):
            batch = input_texts[i:i + current_batch_size]
            try:
                predictions = self._predict_batch_single(batch)
                all_predictions.extend(predictions)
                i += current_batch_size

                # Clear cache periodically
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "cpu":
                    import gc
                    gc.collect()

            except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
                # Check if it's actually an OOM error
                if not _is_oom_error(e):
                    raise

                # Reduce batch size and retry
                new_batch_size = max(_MIN_BATCH_SIZE, int(current_batch_size * _OOM_REDUCTION_FACTOR))

                if new_batch_size < current_batch_size:
                    logger.warning(
                        "OOM at batch_size=%d, reducing to %d",
                        current_batch_size, new_batch_size
                    )
                    current_batch_size = new_batch_size
                    # Update instance batch size to avoid repeated OOMs
                    self.batch_size = current_batch_size
                    # Clear CUDA cache before retry
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                else:
                    # Already at minimum batch size, cannot reduce further
                    logger.error(
                        "OOM even at minimum batch_size=%d. Consider reducing max_len or using CPU.",
                        _MIN_BATCH_SIZE
                    )
                    raise

        return all_predictions

    def _maybe_compile(self, config, device: str):
        if not getattr(config, "compile", False):
            return
        if device != "cuda" or not torch.cuda.is_available():
            return

        backend = getattr(config, "compile_backend", "inductor")
        mode = getattr(config, "compile_mode", "reduce-overhead")

        try:
            self.model = _compile_model(self.model, backend, mode)
            logger.info("Compiled transformer_v2 model with backend=%s", backend)
        except Exception as exc:
            logger.warning("Model compile failed (%s); running uncompiled.", exc)

    def _log_tokenizer_debug(self):
        if not logger.isEnabledFor(logging.DEBUG):
            return
        try:
            use_task_tokens = getattr(self.tokenizer, "use_task_tokens", None)
            use_role_tokens = getattr(self.tokenizer, "use_role_tokens", None)
            logger.debug(
                "Tokenizer config: task=%s use_task_tokens=%s use_role_tokens=%s",
                self.task,
                use_task_tokens,
                use_role_tokens,
            )
            example = "C.C"
            encoded = self.tokenizer.encode_reaction(
                reactants=example,
                products=None,
                task=self.task,
                model_type="transformer" if self.model_type != "decoder_only" else "decoder_only",
                max_src_length=self.max_len,
                max_length=self.max_len,
                truncation=True,
                return_tensors=None,
            )
            logger.debug("Tokenizer example reactants='%s' encoded=%s", example, encoded)
        except Exception as exc:
            logger.debug("Tokenizer debug logging failed: %s", exc)

    def _auto_configure_batch_size(self, config):
        """Auto-configure batch size based on GPU memory buckets.

        The predict_batch method will automatically reduce batch size further if OOM occurs.
        """
        if config.batch_size is not None:
            return

        if self.device != "cuda" or not torch.cuda.is_available():
            # CPU: be very conservative
            self.batch_size = max(1, 8 // max(1, self.beam_size))
            logger.info("Auto-configured CPU batch_size=%d", self.batch_size)
            return

        props = torch.cuda.get_device_properties(0)
        total_mem_gb = props.total_memory / 1e9

        if total_mem_gb < 12:  # e.g. 11GB (2080Ti)
            self.batch_size = 1024
        elif total_mem_gb < 26:  # e.g. 24GB (3090/4090/A5000)
            self.batch_size = 2048
        else:  # e.g. 48GB (A6000) or 80GB (A100)
            self.batch_size = 4096

        logger.info(
            "Auto-configured batch_size=%d (GPU: %.1fGB memory bucket)",
            self.batch_size,
            total_mem_gb,
        )

    @torch.inference_mode()
    def _predict_batch_single(self, reactants_list: List[str]) -> List[str]:
        if self.model_type == "decoder_only":
            encoded_list = []
            for reactants in reactants_list:
                encoded = self.tokenizer.encode_reaction(
                    reactants=reactants,
                    products=None,
                    task=self.task,
                    model_type="decoder_only",
                    max_length=self.max_len,
                    max_src_length=self.max_len,
                    truncation=True,
                    return_tensors=None,
                )
                encoded_list.append(encoded)

            max_len = max(len(e["input_ids"]) for e in encoded_list)
            input_ids = torch.full(
                (len(encoded_list), max_len),
                self.tokenizer.pad_id,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros(
                (len(encoded_list), max_len),
                dtype=torch.bool,
                device=self.device,
            )
            for i, enc in enumerate(encoded_list):
                seq = enc["input_ids"]
                input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
                attention_mask[i, :len(seq)] = True

            # Calculate max_new_tokens to avoid exceeding model's max_seq_len
            prompt_len = input_ids.shape[1]
            model_max_seq = self._model_config.get("max_seq_len", 512)
            max_new_tokens = min(self.max_len, model_max_seq - prompt_len)
            max_new_tokens = max(1, max_new_tokens)  # Ensure at least 1 token

            if self.beam_size > 1:
                sequences, _ = self.model.parallel_beam_search(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    beam_size=self.beam_size,
                    length_penalty=self.length_penalty,
                    eos_id=self.tokenizer.eos_id,
                )
            else:
                sequences = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    top_k=1,
                    eos_id=self.tokenizer.eos_id,
                )

            predictions = []
            for seq in sequences:
                seq_ids = seq.tolist()
                if self.tokenizer.sep_id in seq_ids:
                    seq_ids = seq_ids[seq_ids.index(self.tokenizer.sep_id) + 1:]
                predictions.append(
                    self.tokenizer.decode(seq_ids, skip_special=True, stop_at_eos=True)
                )
            return predictions

        # Encoder-decoder transformer
        encoded_list = []
        for reactants in reactants_list:
            encoded = self.tokenizer.encode_reaction(
                reactants=reactants,
                products=None,
                task=self.task,
                model_type="transformer",
                max_src_length=self.max_len,
                truncation=True,
                return_tensors=None,
            )
            encoded_list.append(encoded["src"])

        max_len = max(len(seq) for seq in encoded_list)
        src = torch.full(
            (len(encoded_list), max_len),
            self.tokenizer.pad_id,
            dtype=torch.long,
            device=self.device,
        )
        src_mask = torch.zeros(
            (len(encoded_list), max_len),
            dtype=torch.bool,
            device=self.device,
        )
        for i, seq in enumerate(encoded_list):
            src[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
            src_mask[i, :len(seq)] = True

        if self.beam_size > 1:
            predictions = []
            for i in range(len(reactants_list)):
                seq, _ = self.model.beam_search(
                    src[i:i+1],
                    src_mask[i:i+1],
                    max_len=self.max_len,
                    beam_size=self.beam_size,
                    length_penalty=self.length_penalty,
                )
                pred = self.tokenizer.decode(seq[0].tolist(), skip_special=True, stop_at_eos=True)
                predictions.append(pred)
        else:
            sequences = self.model.generate(
                src,
                src_mask,
                max_len=self.max_len,
                top_k=1,
            )
            predictions = [
                self.tokenizer.decode(seq.tolist(), skip_special=True, stop_at_eos=True)
                for seq in sequences
            ]

        return predictions


def _resolve_repo_path(config) -> Optional[Path]:
    """Resolve the external reaction-model repo, or None to use the vendored architecture."""
    repo_path = getattr(config, "repo_path", None)
    if repo_path:
        return Path(repo_path)

    env_repo = os.environ.get("RXNMOL_REACTION_MODEL_REPO")
    if env_repo:
        return Path(env_repo)

    return None


def _load_vendored_modules():
    """Load the architecture from the vendored ``_arch`` package (no external repo needed).

    Also alias the vendored modules under their original ``src.*`` import paths so that
    checkpoints pickled against the training repo (e.g. ``model.pt`` holding a ``src.config``
    object) unpickle without the external repository on ``sys.path``.
    """
    from . import _arch
    from ._arch import tokenizer, transformer, decoder_model, config

    if "src" not in sys.modules:
        src_pkg = types.ModuleType("src")
        src_pkg.__path__ = []  # mark as package
        sys.modules["src"] = src_pkg
    sys.modules.setdefault("src.config", config)
    sys.modules.setdefault("src.tokenizer", tokenizer)
    sys.modules.setdefault("src.transformer", transformer)
    sys.modules.setdefault("src.decoder_model", decoder_model)

    return tokenizer.SMILESTokenizer, transformer.Transformer, decoder_model.CausalTransformer


def _compile_model(model, backend: str, mode: str):
    try:
        return torch.compile(model, backend=backend, mode=mode)
    except TypeError:
        return torch.compile(model, backend=backend)


def _require_model_dir(config) -> Path:
    model_dir = Path(config.model_dir) if getattr(config, "model_dir", None) else None
    if model_dir is None:
        raise ValueError("transformer_v2 requires model_dir pointing to a run directory.")
    if not model_dir.exists():
        raise FileNotFoundError(f"transformer_v2 model_dir not found: {model_dir}")
    return model_dir


def _resolve_tokenizer_path(config, model_dir: Path) -> Path:
    tokenizer_path = Path(config.tokenizer_path) if getattr(config, "tokenizer_path", None) else None
    if tokenizer_path is None:
        tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"transformer_v2 tokenizer.json not found in {model_dir}. "
            "Set reaction_model.tokenizer_path explicitly."
        )
    return tokenizer_path


def _load_training_config(model_dir: Path) -> dict:
    training_path1 = model_dir / "training_config.json" # older name
    if training_path1.exists():
        with open(training_path1, "r") as handle:
            return json.load(handle)
    training_path2 = model_dir / "config.resolved.json" 
    if training_path2.exists():
        with open(training_path2, "r") as handle:
            return json.load(handle)
    
    raise FileNotFoundError(
        f"transformer_v2 config.resolved.json nor training_config.json found in {model_dir}"
        )


def _resolve_model_path(config, model_dir: Path) -> Path:
    checkpoint_path = Path(config.checkpoint_path) if getattr(config, "checkpoint_path", None) else None
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"transformer_v2 checkpoint_path not found: {checkpoint_path}")
        return checkpoint_path

    model_path = model_dir / "model.pt"
    if model_path.exists():
        return model_path

    raise ValueError(
        "transformer_v2 requires model.pt in model_dir or an explicit checkpoint_path."
    )


def _load_external_modules(repo_path: Path):
    repo_path = repo_path.resolve()
    src_dir = repo_path / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"Expected src/ under repo_path: {repo_path}")

    tokenizer_mod = _load_module("rxnmol_ext_tokenizer", src_dir / "tokenizer.py")
    transformer_mod = _load_module("rxnmol_ext_transformer", src_dir / "transformer.py")
    decoder_mod = _load_module("rxnmol_ext_decoder", src_dir / "decoder_model.py")

    return (
        getattr(tokenizer_mod, "SMILESTokenizer"),
        getattr(transformer_mod, "Transformer"),
        getattr(decoder_mod, "CausalTransformer"),
    )


def _add_repo_to_syspath(repo_path: Path):
    repo_path = repo_path.resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _build_model(model_type: str, model_config: dict, transformer_cls, decoder_cls):
    if model_type in {"encoder_decoder", "seq2seq"}:
        model_type = "transformer"

    common_kwargs = dict(
        vocab_size=model_config["vocab_size"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_kv_heads=model_config["n_kv_heads"],
        n_layers=model_config["n_layers"],
        d_ff=model_config["d_ff"],
        max_seq_len=model_config["max_seq_len"],
        dropout=model_config.get("dropout", 0.1),
        pad_id=model_config.get("pad_id", 0),
        bos_id=model_config.get("bos_id", 1),
        eos_id=model_config.get("eos_id", 2),
        rope_base=model_config.get("rope_base", 10000.0),
    )

    if model_type == "decoder_only":
        return decoder_cls(**common_kwargs)
    return transformer_cls(**common_kwargs)


def _extract_state_dict(state_obj):
    if isinstance(state_obj, dict):
        if "model_state_dict" in state_obj:
            return state_obj["model_state_dict"]
        if "state_dict" in state_obj:
            state_dict = state_obj["state_dict"]
            if any(k.startswith("model.") for k in state_dict.keys()):
                return {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
            return state_dict
    return state_obj


def _is_oom_error(exc: Exception) -> bool:
    """Check if an exception is an out-of-memory error."""
    if isinstance(exc, MemoryError):
        return True
    # CUDA OOM
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    # RuntimeError with OOM message (older PyTorch versions)
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda" in msg and "alloc" in msg:
            return True
        # CPU OOM
        if "cannot allocate" in msg or "memory" in msg:
            return True
    return False
