"""
CSearch docking energy prediction objectives.

This module provides objectives from the CSearch paper for predicting
docking energies to 4 protein targets using GNN models:
- MPro (SARS-CoV-2 Main Protease) - PDB: 6M0K
- BTK (Tyrosine-protein kinase BTK) - PDB: 5P9H
- ALK (Anaplastic Lymphoma Kinase) - PDB: 4MKC
- H1N1_NA (H1N1 Neuraminidase) - PDB: 3TI5

Reference:
    Kim et al. "CSearch: chemical space search via virtual synthesis and
    global optimization" J Cheminform (2024)
    https://github.com/seoklab/CSearch

It handles lazy loading of heavy models to support multiprocessing.

SCALING NOTES (to match CSearch paper Table 1 values):
=======================================================
The CSearch code applies TWO separate scalings to raw GNN predictions:

1. energy_calculation.py (line 100-103):
    https://github.com/seoklab/CSearch/blob/main/opps/energy_calculation.py#L100-L103

   INTENDED behavior (but has a bug due to Python truthy evaluation):
   ```python
   if input_pdbid == '4MKC' or '3TI5' or '5P9H':  # BUG: always True!
       pred_list = list(np.around(pred_list*10, 3))
   else:
       pred_list = list(np.around(pred_list, 3))  # 6M0K: no scaling
   ```
   INTENDED: Only 4MKC, 3TI5, 5P9H get 10x; 6M0K gets 1x (no scaling)
   ACTUAL (due to bug): ALL targets get 10x scaling

2. CSearch.py write_bank() (line 327-330):
   https://github.com/seoklab/CSearch/blob/main/CSearch.py#L327C1-L339C1
   
   ```python
   if self.pdbid == '6M0K':
       x = 10   # MPro: 10x for display
   else:
       x = 100  # Others: 100x for display
   ```
   Applied when writing CSV output to match paper scale.

TOTAL SCALING to match paper Table 1 values:
   Assuming INTENDED behavior (fixing the bug):
   - 6M0K (MPro):     raw × 1  × 10  = 10x   → Paper: -156.0
   - 5P9H (BTK):      raw × 10 × 100 = 1000x → Paper: -199.6
   - 4MKC (ALK):      raw × 10 × 100 = 1000x → Paper: -150.4
   - 3TI5 (H1N1_NA):  raw × 10 × 100 = 1000x → Paper: -148.7

Example (Aspirin CC(=O)Oc1ccccc1C(=O)O):
   - Raw GNN output:   ~-0.42 (MPro), ~-0.047 (BTK)
   - With full scale:  -42 (MPro), -47 (BTK)
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from rdkit import Chem

from .base import MolObjective

logger = logging.getLogger(__name__)

# =============================================================================
# CSearch Path Setup
# =============================================================================

CSEARCH_PATH = Path("/home/alatoo/projects/fragments/CSearch")
CSEARCH_MODELS_PATH = CSEARCH_PATH / "opps" / "save"

# Add CSearch libs to path for imports (not the opps package itself to avoid __init__ issues)
if CSEARCH_PATH.exists():
    # Add the opps/libs directory directly to avoid opps/__init__.py imports
    sys.path.insert(0, str(CSEARCH_PATH / "opps"))

try:
    import dgl
    # Import directly from libs submodule (not via opps package)
    from libs.models import MyModel
    from libs.io_inference import MyDataset, my_collate_fn
    HAS_CSEARCH = True
except ImportError as e:
    logger.warning(f"Error importing CSearch modules: {e}")
    logger.warning("CSearch dependencies not found. Docking energy objectives unavailable.")
    HAS_CSEARCH = False


# =============================================================================
# CSearch GNN Model Wrapper
# =============================================================================

class CSearchModelWrapper:
    """
    Wrapper for CSearch GNN models (MPro, BTK, ALK, H1N1_NA).
    Handles lazy loading and batch inference with automatic batch size adjustment.

    The models predict GalaxyDock3 docking energies using a GCN architecture
    with PMA (Pooling by Multihead Attention) readout.
    """

    # Model architecture config (from CSearch energy_calculation.py)
    MODEL_CONFIG = {
        'model_type': 'gcn',
        'num_layers': 4,
        'hidden_dim': 128,
        'readout': 'pma',
        'dropout_prob': 0.2,
        'out_dim': 2,
    }

    # PDB ID to model path mapping
    MODEL_PATHS = {
        '6M0K': CSEARCH_MODELS_PATH / '6M0K_gcn_128_pma_m2cdo.pth',  # MPro
        '5P9H': CSEARCH_MODELS_PATH / '5P9H_gcn_128_pma_m2cdo.pth',  # BTK
        '4MKC': CSEARCH_MODELS_PATH / '4MKC_gcn_128_pma_m2cdo.pth',  # ALK
        '3TI5': CSEARCH_MODELS_PATH / '3TI5_gcn_128_pma_m2cdo.pth',  # H1N1_NA
    }

    # Full scaling factors to match CSearch paper Table 1 values
    # Combines: energy_calc() scaling (INTENDED) × write_bank() scaling
    # See module docstring for detailed explanation of the two-step scaling
    #
    # energy_calc() INTENDED (line 100-103):
    #   - 6M0K: 1x (no scaling)
    #   - Others (4MKC, 3TI5, 5P9H): 10x
    #
    # write_bank() (line 327-330):
    #   - 6M0K: 10x
    #   - Others: 100x
    #
    # TOTAL = energy_calc × write_bank:
    SCALE_FACTORS = {
        # ignoring bug for 6MOK to match paper values
        '6M0K': 10 * 10,    # MPro:    10 × 10  = 100x   → Paper: -156.0
        '5P9H': 10 * 100,   # BTK:     10 × 100 = 1000x → Paper: -199.6
        '4MKC': 10 * 100,   # ALK:     10 × 100 = 1000x → Paper: -150.4
        '3TI5': 10 * 100,   # H1N1_NA: 10 × 100 = 1000x → Paper: -148.7
    }

    def __init__(
        self,
        pdbid: str,
        device: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        min_batch_size: int = 16,
        num_inference_passes: int = 3,
    ):
        """
        Args:
            pdbid: Target protein PDB ID ('6M0K', '5P9H', '4MKC', '3TI5')
            device: 'cuda' or 'cpu' (auto-detected if None)
            max_batch_size: Maximum graphs per batch (auto-set based on device)
            min_batch_size: Minimum batch size before sequential fallback
            num_inference_passes: Number of forward passes to average (default 3)
        """
        if pdbid not in self.MODEL_PATHS:
            raise ValueError(
                f"Invalid pdbid: {pdbid}. Choose from {list(self.MODEL_PATHS.keys())}"
            )

        self.pdbid = pdbid
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_inference_passes = num_inference_passes
        self._model = None

        # Set batch size based on device
        if max_batch_size is not None:
            self.max_batch_size = max_batch_size
        else:
            if self.device == 'cuda':
                try:
                    total_mem = torch.cuda.get_device_properties(
                        torch.cuda.current_device()
                    ).total_memory / 1e9
                    if total_mem <= 15:
                        self.max_batch_size = 256
                    elif total_mem <= 25:
                        self.max_batch_size = 512
                    elif total_mem <= 50:
                        self.max_batch_size = 1024
                    else:
                        self.max_batch_size = 2048
                except Exception:
                    self.max_batch_size = 256
            else:
                self.max_batch_size = 128  # CPU default

        self.min_batch_size = min_batch_size

    def _load_model(self):
        """Lazy load the GNN model."""
        if self._model is not None:
            return self._model

        model_path = self.MODEL_PATHS[self.pdbid]
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        logger.info(f"Loading CSearch model for {self.pdbid} from {model_path}")

        self._model = MyModel(**self.MODEL_CONFIG)
        self._model = self._model.to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()

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

    def _predict_batch_single(self, mols: List[Chem.Mol], model) -> np.ndarray:
        """
        Process a single batch of molecules through the model.

        Args:
            mols: List of RDKit Mol objects
            model: Loaded GNN model

        Returns:
            numpy array of predictions
        """
        dataset = MyDataset(mols)
        loader = DataLoader(
            dataset=dataset,
            batch_size=len(mols),
            shuffle=False,
            collate_fn=my_collate_fn
        )

        with torch.no_grad():
            for graph in loader:
                graph = graph.to(self.device)

                # Multiple forward passes and average (from CSearch)
                preds_list = []
                for _ in range(self.num_inference_passes):
                    tmp_graph = graph.clone()
                    pred, _ = model(tmp_graph)
                    pred = pred.unsqueeze(-1)
                    preds_list.append(pred)

                preds_tensor = torch.cat(preds_list, dim=-1)
                mean_preds = torch.mean(preds_tensor, dim=-1)
                # Take first output dimension
                predictions = mean_preds[:, 0].cpu().numpy()

        return predictions

    def _predict_with_oom_retry(self, mols: List[Chem.Mol], model) -> List[float]:
        """
        Predict with automatic OOM recovery and batch chunking.

        On CUDA OOM:
        1. Clear CUDA cache
        2. Reduce batch size by 30%
        3. Retry with smaller chunks
        4. Update self.max_batch_size for future calls
        """
        if not mols:
            return []

        current_batch_size = min(self.max_batch_size, len(mols))

        # For small batches, run directly
        if len(mols) <= self.min_batch_size:
            try:
                preds = self._predict_batch_single(mols, model)
                return self._scale_predictions(preds)
            except Exception as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    logger.warning(
                        f"OOM even for small batch ({len(mols)} mols), "
                        "falling back to sequential"
                    )
                    return self._predict_sequential(mols, model)
                raise

        while current_batch_size >= self.min_batch_size:
            try:
                logger.debug(
                    f"CSearch predict: {len(mols)} mols @ batch_size={current_batch_size}"
                )

                all_preds = []
                for i in range(0, len(mols), current_batch_size):
                    chunk = mols[i:i + current_batch_size]
                    preds = self._predict_batch_single(chunk, model)
                    all_preds.extend(preds.tolist())

                return self._scale_predictions(all_preds)

            except Exception as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    logger.warning(
                        f"CUDA OOM at batch_size={current_batch_size} "
                        f"for {len(mols)} mols: {e}"
                    )

                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    mem_stats = self._get_cuda_mem_stats()
                    mem_msg = ""
                    if mem_stats:
                        mem_msg = (
                            f" (alloc={mem_stats['allocated']:.2f}GB, "
                            f"reserved={mem_stats['reserved']:.2f}/"
                            f"{mem_stats['total']:.2f}GB)"
                        )

                    # Reduce batch size by 30%
                    new_batch_size = int(current_batch_size * 0.7)
                    logger.warning(
                        f"⚠️ CUDA OOM at batch_size={current_batch_size}{mem_msg}; "
                        f"reducing to {new_batch_size} and retrying..."
                    )

                    if new_batch_size >= self.min_batch_size:
                        self.max_batch_size = new_batch_size
                        logger.info(f"Updated CSearch max_batch_size to {new_batch_size}")

                    current_batch_size = new_batch_size
                else:
                    logger.error(f"CSearch batch prediction failed (non-OOM): {e}")
                    raise

        # Fall back to sequential
        logger.warning(
            f"⚠️ Batch prediction failed at min_batch_size={self.min_batch_size}, "
            f"falling back to sequential processing"
        )
        return self._predict_sequential(mols, model)

    def _predict_sequential(self, mols: List[Chem.Mol], model) -> List[float]:
        """Fall back to processing molecules one at a time."""
        results = []
        for mol in mols:
            try:
                preds = self._predict_batch_single([mol], model)
                results.append(float(preds[0]))
            except Exception as e:
                logger.error(f"Single molecule prediction failed: {e}")
                results.append(0.0)
        return self._scale_predictions(results)

    def _scale_predictions(self, predictions: Union[np.ndarray, List[float]]) -> List[float]:
        """
        Apply full scaling to match CSearch paper Table 1 values.

        Uses target-specific scale factors that combine:
        - energy_calc() INTENDED scaling (1x for 6M0K, 10x for others)
        - write_bank() display scaling (10x for 6M0K, 100x for others)

        See module docstring for detailed explanation.
        """
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()

        scale = self.SCALE_FACTORS[self.pdbid]
        return [p * scale for p in predictions]

    def predict(
        self,
        inputs: Union[str, Chem.Mol, List[Union[str, Chem.Mol]]]
    ) -> Union[float, List[float]]:
        """
        Predict docking energy for SMILES or RDKit Mol inputs.

        Args:
            inputs: Single SMILES/Mol or list of SMILES/Mol

        Returns:
            Predicted docking energy (lower is better binding)
        """
        model = self._load_model()

        is_single = isinstance(inputs, (str, Chem.Mol))
        if is_single:
            inputs = [inputs]

        # Convert to RDKit Mol objects and filter valid molecules
        # Note: Molecules must have at least one bond for the GNN to process
        mols = []
        valid_indices = []

        for i, inp in enumerate(inputs):
            try:
                if isinstance(inp, str):
                    mol = Chem.MolFromSmiles(inp)
                else:
                    mol = inp
                if mol is not None:
                    # Skip molecules with no bonds (single atoms cause tensor errors)
                    if mol.GetNumBonds() == 0:
                        logger.debug(f"Skipping molecule with no bonds: {Chem.MolToSmiles(mol)}")
                        continue
                    mols.append(mol)
                    valid_indices.append(i)
            except Exception:
                pass

        if not mols:
            return 0.0 if is_single else [0.0] * len(inputs)

        # Batch inference with OOM recovery
        preds = self._predict_with_oom_retry(mols, model)

        # Map back to original indices
        results = [0.0] * len(inputs)
        for idx, pred in zip(valid_indices, preds):
            results[idx] = pred

        return results[0] if is_single else results


# =============================================================================
# Objectives
# =============================================================================

if HAS_CSEARCH:

    class MProObjective(MolObjective):
        """
        SARS-CoV-2 Main Protease (MPro) docking energy prediction.
        PDB: 6M0K

        Lower scores indicate better predicted binding.
        """
        supports_batch = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._wrapper = None

        def compute(self, mol: Chem.Mol) -> float:
            if self._wrapper is None:
                self._wrapper = CSearchModelWrapper(pdbid='6M0K')
            # Lower docking energy is better -> return as-is for minimization
            return self._wrapper.predict(mol)

        def compute_batch(self, mols: List[Chem.Mol]) -> List[float]:
            if self._wrapper is None:
                self._wrapper = CSearchModelWrapper(pdbid='6M0K')
            return self._wrapper.predict(mols)


    class BTKObjective(MolObjective):
        """
        Tyrosine-protein kinase BTK docking energy prediction.
        PDB: 5P9H

        Lower scores indicate better predicted binding.
        """
        supports_batch = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._wrapper = None

        def compute(self, mol: Chem.Mol) -> float:
            if self._wrapper is None:
                self._wrapper = CSearchModelWrapper(pdbid='5P9H')
            return self._wrapper.predict(mol)

        def compute_batch(self, mols: List[Chem.Mol]) -> List[float]:
            if self._wrapper is None:
                self._wrapper = CSearchModelWrapper(pdbid='5P9H')
            return self._wrapper.predict(mols)


    class ALKObjective(MolObjective):
        """
        Anaplastic Lymphoma Kinase (ALK) docking energy prediction.
        PDB: 4MKC

        Lower scores indicate better predicted binding.
        """
        supports_batch = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._wrapper = None

        def compute(self, mol: Chem.Mol) -> float:
            if self._wrapper is None:
                self._wrapper = CSearchModelWrapper(pdbid='4MKC')
            return self._wrapper.predict(mol)

        def compute_batch(self, mols: List[Chem.Mol]) -> List[float]:
            if self._wrapper is None:
                self._wrapper = CSearchModelWrapper(pdbid='4MKC')
            return self._wrapper.predict(mols)


    class H1N1NAObjective(MolObjective):
        """
        H1N1 Neuraminidase docking energy prediction.
        PDB: 3TI5

        Lower scores indicate better predicted binding.
        """
        supports_batch = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._wrapper = None

        def compute(self, mol: Chem.Mol) -> float:
            if self._wrapper is None:
                self._wrapper = CSearchModelWrapper(pdbid='3TI5')
            return self._wrapper.predict(mol)

        def compute_batch(self, mols: List[Chem.Mol]) -> List[float]:
            if self._wrapper is None:
                self._wrapper = CSearchModelWrapper(pdbid='3TI5')
            return self._wrapper.predict(mols)

else:
    logger.warning("CSearch dependencies not found. Docking energy objectives unavailable.")
