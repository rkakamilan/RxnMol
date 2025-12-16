"""
SynFlowNet reward objective functions.

This module provides objectives from the SynFlowNet paper and related MPNN-based models.
It integrates:
- SEH (Soluble Epoxide Hydrolases) binding proxy
- CB1 (Cannabinoid Receptor 1) binding affinity (VIP36 target)
- QED and SA Score (via SynFlowNet utils or RDKit)
- Vina docking (optional)

It handles lazy loading of heavy models to support multiprocessing.
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import numpy as np
import torch
import torch_geometric.data as gd
from rdkit import Chem
from rdkit.Chem import QED
from torch import Tensor

from .base import MolObjective

logger = logging.getLogger(__name__)

# =============================================================================
# SynFlowNet / MPNN Model Wrapper
# =============================================================================

# Add the synflownet package to the path if it's not already installed
# TODO: Later we add SynFlowNet as a proper dependency and import from there
SYNFLOWNET_PATH = Path("/home/alatoo/projects/fragments/molfinder-rxn-synflownet/synflownet/src")
if SYNFLOWNET_PATH.exists():
    sys.path.insert(0, str(SYNFLOWNET_PATH))

# TODO: remove try except
try:
    from synflownet.models import bengio2021flow
    from synflownet.models.bengio2021flow import MPNNet, mol2graph
    from synflownet.utils import sascore
    HAS_SYNFLOWNET = True
except ImportError as e:
    logger.warning(f"Error importing SynFlowNet modules: {e}")
    logger.warning("SynFlowNet dependencies not found. sEH and CB1 objectives unavailable.")
    HAS_SYNFLOWNET = False
    raise e


class SynFlowNetModelWrapper:
    """
    Wrapper for SynFlowNet MPNN models (SEH, CB1, etc.).
    Handles lazy loading and batch inference with OOM recovery.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None,
                 is_seh_original: bool = False, max_batch_size = None, min_batch_size: int = 32):
        """
        Args:
            model_path: Path to model checkpoint (for CB1/custom models)
            device: 'cuda' or 'cpu'
            is_seh_original: If True, load the original SEH proxy model from SynFlowNet
            max_batch_size: Maximum number of graphs per batch (reduced on OOM)
            min_batch_size: Minimum batch size before falling back to sequential
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_seh_original = is_seh_original
        # self.max_batch_size = max_batch_size
        # set batch size based on device and its memory 
        if max_batch_size is not None:
            self.max_batch_size = max_batch_size
        else:
            if self.device == 'cuda':
                total_mem = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1e9  # in GB
                if total_mem <= 15:
                    self.max_batch_size = 512
                elif total_mem <= 25:
                    self.max_batch_size = 1024
                elif total_mem <= 50:
                    self.max_batch_size = 2048
                else:
                    self.max_batch_size = 3048
                
            else:
                # CPU default
                self.max_batch_size = 64

        self.min_batch_size = min_batch_size
        self._model = None
        self._normalization_params = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return self._model

        
        if self.is_seh_original:
            # Load original SEH proxy
            logger.info(f"Loading original sEH model from Bengio2021Flow")
            self._model = bengio2021flow.load_original_model()
            self._model.to(self.device)
            self._model.eval()
        else:
            # Load custom MPNN checkpoint (e.g. CB1)
            if not self.model_path or not Path(self.model_path).exists():
                 # Fallback for CB1 if path not provided
                 default_cb1 = '/home/alatoo/projects/fragments/docking/prediction_model/models/mpnn_zscore_baseline.pt'
                 if Path(default_cb1).exists():
                     self.model_path = default_cb1
                 else:
                     raise FileNotFoundError(f"Model path not found: {self.model_path}")

            logger.info(f"Loading Proxy model from: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Extract config
            state_dict = checkpoint['model_state_dict']
            dim = state_dict.get('lin0.weight', torch.empty(64)).shape[0]
            num_conv_steps = 12 # Default
            
            self._model = MPNNet(
                num_feat=71, 
                num_vec=0, 
                dim=dim, 
                num_conv_steps=num_conv_steps, 
                num_out_per_mol=1
            )
            self._model.load_state_dict(state_dict)
            self._model.to(self.device)
            self._model.eval()
            
            self._normalization_params = checkpoint.get('normalization_params')

        return self._model

    @staticmethod
    def _get_cuda_mem_stats() -> Optional[Dict[str, float]]:
        """Get CUDA memory statistics for logging."""
        if not torch.cuda.is_available():
            return None
        try:
            idx = torch.cuda.current_device()
            return {
                "allocated": torch.cuda.memory_allocated(idx) / 1e9,
                "reserved": torch.cuda.memory_reserved(idx) / 1e9,
                "total": torch.cuda.get_device_properties(idx).total_memory / 1e9,
            }
        except Exception:
            return None

    def _predict_batch_single(self, graphs: List, model) -> Tensor:
        """
        Process a single batch of graphs through the model.

        Args:
            graphs: List of torch_geometric Data objects
            model: Loaded MPNN model

        Returns:
            Tensor of predictions
        """
        batch = gd.Batch.from_data_list(graphs).to(self.device)
        with torch.no_grad():
            preds = model(batch).reshape((-1,)).cpu()
            if self.is_seh_original:
                preds = preds / 8
                preds = preds.clip(1e-4, 100)
        return preds

    def _predict_with_oom_retry(self, graphs: List, model) -> List[float]:
        """
        Predict with automatic OOM recovery and batch chunking.

        On CUDA OOM:
        1. Clear CUDA cache
        2. Reduce batch size by 30%
        3. Retry with smaller chunks
        4. Update self.max_batch_size for future calls

        Args:
            graphs: List of torch_geometric Data objects
            model: Loaded MPNN model

        Returns:
            List of predictions (floats)
        """
        if not graphs:
            return []

        current_batch_size = min(self.max_batch_size, len(graphs))

        # For small batches, run directly
        if len(graphs) <= self.min_batch_size:
            try:
                preds = self._predict_batch_single(graphs, model)
                return [float(p) for p in preds]
            except Exception as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    logger.warning(f"OOM even for small batch ({len(graphs)} graphs), falling back to sequential")
                    return self._predict_sequential(graphs, model)
                raise

        while current_batch_size >= self.min_batch_size:
            try:
                logger.debug(
                    f"MPNN predict attempt: {len(graphs)} graphs @ batch_size={current_batch_size}"
                )

                all_preds = []
                for i in range(0, len(graphs), current_batch_size):
                    chunk = graphs[i:i + current_batch_size]
                    preds = self._predict_batch_single(chunk, model)
                    all_preds.extend([float(p) for p in preds])

                return all_preds

            except Exception as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    logger.warning(
                        f"CUDA OOM at batch_size={current_batch_size} for {len(graphs)} graphs: {e}"
                    )

                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    mem_msg = ""
                    mem_stats = self._get_cuda_mem_stats()
                    if mem_stats:
                        mem_msg = (
                            f" (alloc={mem_stats['allocated']:.2f}GB, "
                            f"reserved={mem_stats['reserved']:.2f}/{mem_stats['total']:.2f}GB)"
                        )

                    # Reduce batch size by 30%
                    new_batch_size = int(current_batch_size * 0.7)
                    logger.warning(
                        f"⚠️ CUDA OOM at batch_size={current_batch_size} for {len(graphs)} graphs{mem_msg}; "
                        f"reducing to {new_batch_size} and retrying..."
                    )

                    # Update instance batch size for future calls
                    if new_batch_size >= self.min_batch_size:
                        self.max_batch_size = new_batch_size
                        logger.info(f"Updated MPNN max_batch_size to {new_batch_size}")

                    current_batch_size = new_batch_size
                else:
                    # Not an OOM error, re-raise
                    logger.error(f"MPNN batch prediction failed (non-OOM): {e}")
                    raise

        # If we get here, even min_batch_size failed - fall back to sequential
        logger.warning(
            f"⚠️ Batch prediction failed at min_batch_size={self.min_batch_size}, "
            f"falling back to sequential processing"
        )
        return self._predict_sequential(graphs, model)

    def _predict_sequential(self, graphs: List, model) -> List[float]:
        """Fall back to processing graphs one at a time."""
        results = []
        for g in graphs:
            try:
                preds = self._predict_batch_single([g], model)
                results.append(float(preds[0]))
            except Exception as e:
                logger.error(f"Single graph prediction failed: {e}")
                results.append(0.0)
        return results

    def predict(self, inputs: Union[str, Chem.Mol, List[Union[str, Chem.Mol]]]) -> Union[float, List[float]]:
        """Predict reward/affinity for SMILES or RDKit Mol inputs."""
        model = self._load_model()

        is_single = isinstance(inputs, (str, Chem.Mol))
        if is_single:
            inputs = [inputs]

        # Convert to graphs
        graphs = []
        valid_indices = []

        for i, inp in enumerate(inputs):
            try:
                if isinstance(inp, str):
                    mol = Chem.MolFromSmiles(inp)
                else:
                    mol = inp
                if mol:
                    # Add Hs for MPNN
                    mol = Chem.AddHs(mol)
                    g = mol2graph(mol)
                    graphs.append(g)
                    valid_indices.append(i)
            except Exception:
                pass

        if not graphs:
            return 0.0 if is_single else [0.0] * len(inputs)

        # Batch inference with OOM recovery
        preds = self._predict_with_oom_retry(graphs, model)

        # Map back to original indices
        results = [0.0] * len(inputs)
        for idx, pred in zip(valid_indices, preds):
            results[idx] = pred

        return results[0] if is_single else results


# =============================================================================
# Objectives
# =============================================================================

if HAS_SYNFLOWNET:

    class SEHObjective(MolObjective):
        """
        SEH (Soluble Epoxide Hydrolase) binding affinity prediction.
        Uses the original proxy model from SynFlowNet paper.
        """
        supports_batch = True

        def __init__(self, config=None, **kwargs):
            super().__init__(config=config, **kwargs)
            # Lazy initialization handled by wrapper
            self._wrapper = None

        def compute(self, mol: Chem.Mol) -> float:
            if self._wrapper is None:
                self._wrapper = SynFlowNetModelWrapper(is_seh_original=True)
            # Maximize SEH -> Return negative for CSA minimization
            return -self._wrapper.predict(mol)

        def compute_batch(self, mols: List[Chem.Mol]) -> List[float]:
            if self._wrapper is None:
                self._wrapper = SynFlowNetModelWrapper(is_seh_original=True)
            preds = self._wrapper.predict(mols)
            return [-float(p) for p in preds]


    class CB1ZscoreObjective(MolObjective):
        """
        CB1 (Cannabinoid Receptor 1) binding affinity prediction.
        Target: VIP36
        Uses MPNN model trained on docking scores.
        """
        supports_batch = True

        def __init__(self, config=None, **kwargs):
            super().__init__(config=config, **kwargs)
            # TODO: when we publish the code, we need to update this path
            self.model_path = '/home/alatoo/projects/fragments/docking/prediction_model/models/mpnn_zscore_baseline.pt'          
            self._wrapper = None

        def compute(self, mol: Chem.Mol) -> float:
            if self._wrapper is None:
                self._wrapper = SynFlowNetModelWrapper(model_path=self.model_path, is_seh_original=False)
            # Minimize Z-score -> Return raw value
            return self._wrapper.predict(mol)

        def compute_batch(self, mols: List[Chem.Mol]) -> List[float]:
            if self._wrapper is None:
                self._wrapper = SynFlowNetModelWrapper(model_path=self.model_path, is_seh_original=False)
            preds = self._wrapper.predict(mols)
            return [float(p) for p in preds]

    class CB1RawObjective(MolObjective):
        """
        CB1 (Cannabinoid Receptor 1) binding affinity prediction.
        Target: VIP36
        Uses MPNN model trained on docking scores.
        """
        supports_batch = True

        def __init__(self, config=None, **kwargs):
            super().__init__(config=config, **kwargs)
            # TODO: when we publish the code, we need to update this path
            self.model_path = '/home/alatoo/projects/fragments/docking/prediction_model/models/mpnn_raw_baseline.pt'          
            self._wrapper = None

        def compute(self, mol: Chem.Mol) -> float:
            if self._wrapper is None:
                self._wrapper = SynFlowNetModelWrapper(model_path=self.model_path, is_seh_original=False)
            # Minimize Raw Score -> Return raw value
            return self._wrapper.predict(mol)
        
        def compute_batch(self, mols: List[Chem.Mol]) -> List[float]:
            if self._wrapper is None:
                self._wrapper = SynFlowNetModelWrapper(model_path=self.model_path, is_seh_original=False)
            preds = self._wrapper.predict(mols)
            return [float(p) for p in preds]
            
    
    class CB1MinMaxObjective(MolObjective):
        """
        CB1 (Cannabinoid Receptor 1) binding affinity prediction
        """
        supports_batch = True

        def __init__(self, config=None, **kwargs):
            super().__init__(config=config, **kwargs)
            # TODO: when we publish the code, we need to update this path
            self.model_path = '/home/alatoo/projects/fragments/docking/prediction_model/models/mpnn_minmax_baseline.pt'
            self._wrapper = None

        def compute(self, mol: Chem.Mol) -> float:
            if self._wrapper is None:
                self._wrapper = SynFlowNetModelWrapper(model_path=self.model_path, is_seh_original=False)
            # Maximize MinMax Score -> Return negative
            return -self._wrapper.predict(mol)

        def compute_batch(self, mols: List[Chem.Mol]) -> List[float]:
            if self._wrapper is None:
                self._wrapper = SynFlowNetModelWrapper(model_path=self.model_path, is_seh_original=False)
            preds = self._wrapper.predict(mols)
            return [-float(p) for p in preds]

    class CB1RawSAObjective(MolObjective):
        """
        Combined CB1 Raw docking score × Synthetic Accessibility objective.

        Formula: score = -( (-CB1_raw) × SA_normalized )

        - CB1 Raw: Raw docking score from MPNN proxy (kcal/mol scale).
                   More negative = better binding. Inverted for multiplication.
        - SA: Normalized to [0,1] using SynFlowNet formula: (10 - raw_SA) / 6.5
              Clamped at SA=3.5 (already easy to synthesize).

        The product rewards molecules that bind well AND are synthetically accessible.
        """
        supports_batch = True

        def __init__(self, config=None, **kwargs):
            super().__init__(config=config, **kwargs)
            self.cb1 = CB1RawObjective(config)
            self.sa = SynFlowSAObjective(config)

        def compute(self, mol: Chem.Mol) -> float:
            if not mol:
                return 0.0
            cb1_val = self.cb1.compute(mol)  # Raw value (minimize: more negative = better)
            sa_val = self.sa.compute(mol)    # Negated (maximize: higher SA = easier synthesis)

            # CB1 Raw: more negative = better binding, invert for product
            cb1_pos = -cb1_val
            sa_pos = -sa_val  # Revert SA negation

            return -(cb1_pos * sa_pos)

        def compute_batch(self, mols: List[Chem.Mol]) -> List[float]:
            if not mols:
                return []
            cb1_scores = self.cb1.compute_batch(mols)
            sa_scores = [self.sa.compute(m) for m in mols]
            results = []
            for cb1_val, sa_neg in zip(cb1_scores, sa_scores):
                cb1_pos = -cb1_val
                sa_pos = -sa_neg
                results.append(-(cb1_pos * sa_pos))
            return results

    class CB1ZscoreSAObjective(MolObjective):
        """
        Combined CB1 Z-Score × Synthetic Accessibility objective.

        Formula: score = -( (-CB1_zscore) × SA_normalized )

        - CB1 Z-Score: Standardized docking score (mean=0, std=1).
                       More negative = better binding. Inverted for multiplication.
        - SA: Normalized to [0,1] using SynFlowNet formula: (10 - raw_SA) / 6.5
              Clamped at SA=3.5 (already easy to synthesize).

        The product rewards molecules that bind well AND are synthetically accessible.
        """
        supports_batch = True

        def __init__(self, config=None, **kwargs):
            super().__init__(config=config, **kwargs)
            self.cb1 = CB1ZscoreObjective(config)
            self.sa = SynFlowSAObjective(config)

        def compute(self, mol: Chem.Mol) -> float:
            if not mol:
                return 0.0
            cb1_val = self.cb1.compute(mol)  # Z-score (minimize: more negative = better)
            sa_val = self.sa.compute(mol)    # Negated (maximize)

            # Z-score: more negative = better binding, invert for product
            cb1_pos = -cb1_val
            sa_pos = -sa_val

            return -(cb1_pos * sa_pos)

        def compute_batch(self, mols: List[Chem.Mol]) -> List[float]:
            if not mols:
                return []
            cb1_scores = self.cb1.compute_batch(mols)
            sa_scores = [self.sa.compute(m) for m in mols]
            results = []
            for cb1_val, sa_neg in zip(cb1_scores, sa_scores):
                cb1_pos = -cb1_val
                sa_pos = -sa_neg
                results.append(-(cb1_pos * sa_pos))
            return results

    class CB1MinMaxSAObjective(MolObjective):
        """
        Combined CB1 MinMax × Synthetic Accessibility objective.

        Formula: score = -( CB1_minmax × SA_normalized )

        - CB1 MinMax: Min-max normalized docking score [0,1].
                      Higher = better binding (already in maximize form).
        - SA: Normalized to [0,1] using SynFlowNet formula: (10 - raw_SA) / 6.5
              Clamped at SA=3.5 (already easy to synthesize).

        The product rewards molecules that bind well AND are synthetically accessible.
        Both components are in [0,1] range, so the product is also [0,1].
        """
        supports_batch = True

        def __init__(self, config=None, **kwargs):
            super().__init__(config=config, **kwargs)
            self.cb1 = CB1MinMaxObjective(config)
            self.sa = SynFlowSAObjective(config)

        def compute(self, mol: Chem.Mol) -> float:
            if not mol:
                return 0.0
            cb1_val = self.cb1.compute(mol)  # Negated (maximize: higher = better)
            sa_val = self.sa.compute(mol)    # Negated (maximize)

            # Both are already negated for CSA, revert to positive for product
            cb1_pos = -cb1_val
            sa_pos = -sa_val

            return -(cb1_pos * sa_pos)

        def compute_batch(self, mols: List[Chem.Mol]) -> List[float]:
            if not mols:
                return []
            cb1_scores = self.cb1.compute_batch(mols)
            sa_scores = [self.sa.compute(m) for m in mols]
            results = []
            for cb1_neg, sa_neg in zip(cb1_scores, sa_scores):
                cb1_pos = -cb1_neg
                sa_pos = -sa_neg
                results.append(-(cb1_pos * sa_pos))
            return results


    class SynFlowQEDObjective(MolObjective):
        """QED using RDKit (wrapper for consistency)."""
        def compute(self, mol: Chem.Mol) -> float:
            # Maximize QED -> Return negative
            return -QED.qed(mol) if mol else 0.0


    class SynFlowSAObjective(MolObjective):
        """Synthetic Accessibility Score (normalized 0-1)."""
        def compute(self, mol: Chem.Mol) -> float:
            if not mol:
                return 0.0
            
            raw_score = sascore.calculateScore(mol)
            # Transform to 0-1 (higher is better/easier)
            # ----------------
            # SynFlowNet normalization: 
            # 3.5 acts as a "Saturation Threshold" or "Good Enough" point.
            # Range logic: 
            #   - Raw SA > 10:  Returns 0.0 (Very Hard)
            #   - Raw SA = 3.5: Returns 1.0 (Perfect Score)
            #   - Raw SA < 3.5: Returns 1.0 (Capped at Perfect)
            #
            # Rationale: An SA score < 3.5 is already considered easily synthesizable 
            # in medicinal chemistry. We clamp the reward here to prevent the 
            # algorithm from "gaming the system" by generating trivially simple 
            # molecules (e.g., methane, ethanol) just to get a raw score of 1.0.
            # ----------------
            # https://github.com/mirunacrt/synflownet/blob/main/src/synflownet/tasks/reactions_task.py#L137
            normalized = (10 - raw_score) / (10 - 3.5) 
            
            # Clamp between [0, 1]
            score = max(0.0, min(1.0, normalized))

            # CSA minimizes -> Return negative to make "Higher Score" = "Lower Loss"
            return -score
            

    class SEHQEDObjective(MolObjective):
        """
        Combined SEH and QED objective.
        SynFlowNet Logic: Reward = SEH_score * clamp(QED_score / 0.7)
        """
        supports_batch = True

        def __init__(self, config=None, **kwargs):
            super().__init__(config=config, **kwargs)
            self.seh = SEHObjective(config)
            # We don't strictly need self.qed as an object if we just use RDKit directly
            # but keeping it is fine for consistency.

        def compute(self, mol: Chem.Mol) -> float:
            if not mol:
                return 0.0
            raw_qed = QED.qed(mol)
            
            # 2. Normalize QED (The SynFlow logic)
            # If QED >= 0.7, this becomes 1.0
            qed_norm = min(1.0, raw_qed / 0.7)

            # 3. Get sEH Score 
            # CAUTION: Ensure self.seh returns a NEGATED probability/score 
            # so that -s_neg is a positive [0,1] value.
            s_neg = self.seh.compute(mol)
            s_pos = -s_neg 

            # 4. Combine
            # We want to maximize (s_pos * qed_norm)
            # CSA minimizes, so we return negative
            return -(s_pos * qed_norm)

        def compute_batch(self, mols: List[Chem.Mol]) -> List[float]:
            if not mols:
                return []
            seh_scores = self.seh.compute_batch(mols)  # negated
            results = []
            for s_neg, mol in zip(seh_scores, mols):
                if mol is None:
                    results.append(0.0)
                    continue
                raw_qed = QED.qed(mol)
                qed_norm = min(1.0, raw_qed / 0.7)
                s_pos = -s_neg
                results.append(-(s_pos * qed_norm))
            return results

    class SEHSAObjective(MolObjective):
        """
        Combined SEH and SA objective.
        Returns: SEH * SA_normalized
        """
        supports_batch = True

        def __init__(self, config=None, **kwargs):
            super().__init__(config=config, **kwargs)
            self.seh = SEHObjective(config)
            self.sa = SynFlowSAObjective(config)

        def compute(self, mol: Chem.Mol) -> float:
            s = self.seh.compute(mol) # Already negated
            sa_val = self.sa.compute(mol) # Already negated
            
            # Revert to positive
            s_pos = -s
            sa_pos = -sa_val
            
            # Maximize Product -> Return negative
            return -(s_pos * sa_pos)

        def compute_batch(self, mols: List[Chem.Mol]) -> List[float]:
            if not mols:
                return []
            seh_scores = self.seh.compute_batch(mols)  # negated
            sa_scores = [self.sa.compute(m) for m in mols]  # negated
            results = []
            for s_neg, sa_neg in zip(seh_scores, sa_scores):
                s_pos = -s_neg
                sa_pos = -sa_neg
                results.append(-(s_pos * sa_pos))
            return results

else:
    logger.warning("SynFlowNet dependencies not found. SEH and CB1 objectives unavailable.")
