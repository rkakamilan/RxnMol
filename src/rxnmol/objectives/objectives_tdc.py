"""
TDC (Therapeutics Data Commons) Objective Functions.

Provides objectives based on TDC oracles with batch support:
- GSK3β: Glycogen Synthase Kinase 3 Beta activity
- DRD2: Dopamine Receptor D2 activity
- JNK3: c-Jun N-terminal Kinase 3 activity
- MPO: Multi-Property Optimization for 7 drugs (from GuacaMol via TDC)

All TDC oracles support batch evaluation via list input.

Reference:
    Huang et al. (2021). Therapeutics Data Commons: Machine Learning Datasets
    and Tasks for Drug Discovery and Development. NeurIPS.
"""

import logging
import math
from typing import List, Dict, Any, Optional

from .base import SmilesObjective

logger = logging.getLogger(__name__)

# ============================================================================
# Oracle Cache (per-process, created on first use)
# ============================================================================

_oracle_cache: Dict[str, Any] = {}


def _get_oracle(name: str):
    """
    Lazy-load TDC oracle with caching.

    Oracles are cached per process to avoid reloading.
    """
    if name not in _oracle_cache:
        try:
            from tdc import Oracle
            logger.info(f"Loading TDC Oracle: {name}")
            _oracle_cache[name] = Oracle(name=name)
        except ImportError:
            raise ImportError(
                "TDC not installed. Install with: pip install PyTDC"
            )
        except Exception as e:
            logger.error(f"Failed to load TDC Oracle '{name}': {e}")
            raise
    return _oracle_cache[name]


# Backward compatibility alias
def get_oracle(name: str):
    """Alias for _get_oracle (backward compatibility)."""
    return _get_oracle(name)


# ============================================================================
# Base TDC Objective
# ============================================================================

class TDCObjective(SmilesObjective):
    """
    Base class for TDC-based objectives.

    TDC oracles return values in [0, 1] where higher = better.
    We negate for CSA minimization.

    Attributes:
        oracle_name: TDC oracle identifier (must be set by subclasses)
        display_name: Human-readable name for logging
        supports_batch: True (TDC oracles accept list input)
    """

    oracle_name: str = None  # Must be set by subclasses
    display_name: str = None  # Human-readable name

    # TDC oracles support batch evaluation
    supports_batch = True

    def __init__(self, config=None, **kwargs):
        if self.oracle_name is None:
            raise ValueError("Subclass must define oracle_name")
        super().__init__(config=config, **kwargs)
        self._oracle = None

    def _get_oracle(self):
        """Lazy load oracle."""
        if self._oracle is None:
            self._oracle = _get_oracle(self.oracle_name)
        return self._oracle

    def compute_from_smiles(self, smiles: str) -> float:
        """
        Evaluate single SMILES.

        TDC returns [0,1] where higher = better.
        We negate for CSA minimization.
        """
        try:
            score = float(self._get_oracle()(smiles))
            return -score  # Negate: maximize TDC → minimize CSA
        except Exception as e:
            logger.debug(f"TDC {self.oracle_name} failed for {smiles}: {e}")
            return 0.0  # Return 0 (neutral) on failure

    def compute_batch_from_smiles(self, smiles_list: List[str]) -> List[float]:
        """
        Batch evaluate SMILES (TDC supports list input).

        This is more efficient than serial evaluation.
        """
        if not smiles_list:
            return []

        try:
            oracle = self._get_oracle()
            scores = oracle(smiles_list)

            # Handle both list and single return
            if not isinstance(scores, (list, tuple)):
                scores = [scores]

            return [-float(s) for s in scores]

        except Exception as e:
            logger.warning(f"TDC batch failed for {self.oracle_name}: {e}, falling back to serial")
            return [self.compute_from_smiles(s) for s in smiles_list]


# ============================================================================
# Single-Target Activity Predictors
# ============================================================================

class GSK3BObjective(TDCObjective):
    """
    GSK3β (Glycogen Synthase Kinase 3 Beta) activity.

    Associated with bipolar disorder and Alzheimer's disease.
    Model: Random Forest on ExCAPE-DB with ECFP6 fingerprints.

    Range: [0, 1], higher = more active
    """
    oracle_name = "GSK3B"
    display_name = "GSK3β Activity"


class DRD2Objective(TDCObjective):
    """
    DRD2 (Dopamine Receptor D2) activity.

    Target for antipsychotic drugs (schizophrenia, Parkinson's).
    Model: SVM with Gaussian kernel on ExCAPE-DB.

    Range: [0, 1], higher = more active
    """
    oracle_name = "DRD2"
    display_name = "DRD2 Activity"


class JNK3Objective(TDCObjective):
    """
    JNK3 (c-Jun N-terminal Kinase 3) activity.

    Stress-activated kinase, target for neurodegenerative diseases.
    Model: Random Forest on ExCAPE-DB with ECFP6 fingerprints.

    Range: [0, 1], higher = more active
    """
    oracle_name = "JNK3"
    display_name = "JNK3 Activity"


# ============================================================================
# Multi-Property Optimization (MPO) Objectives
# ============================================================================

class OsimertinibMPOObjective(TDCObjective):
    """
    Osimertinib MPO - EGFR inhibitor for lung cancer.

    Targets:
    - FCFP4 Tanimoto similarity ≤ 0.8
    - ECFP6 Tanimoto similarity > 0.85
    - TPSA > 100
    - logP < 1
    """
    oracle_name = "osimertinib"
    display_name = "Osimertinib MPO"


class FexofenadineMPOObjective(TDCObjective):
    """
    Fexofenadine MPO - antihistamine optimization.

    Goal: Make fexofenadine less greasy
    Targets:
    - Similar to fexofenadine
    - TPSA > 90
    - logP < 4
    """
    oracle_name = "fexofenadine"
    display_name = "Fexofenadine MPO"


class RanolazineMPOObjective(TDCObjective):
    """
    Ranolazine MPO - antianginal drug.

    Goal: More polar ranolazine with fluorine
    Targets:
    - Similar to ranolazine
    - logP < 7
    - TPSA ≈ 95
    - 1 fluorine atom
    """
    oracle_name = "ranolazine"
    display_name = "Ranolazine MPO"


class PerindoprilMPOObjective(TDCObjective):
    """
    Perindopril MPO - ACE inhibitor for hypertension.

    Targets:
    - Similar to perindopril
    - 2 aromatic rings
    """
    oracle_name = "perindopril"
    display_name = "Perindopril MPO"


class AmlodipineMPOObjective(TDCObjective):
    """
    Amlodipine MPO - calcium channel blocker.

    Targets:
    - Similar to amlodipine
    - 3 total rings
    """
    oracle_name = "amlodipine"
    display_name = "Amlodipine MPO"


class SitagliptinMPOObjective(TDCObjective):
    """
    Sitagliptin MPO - DPP-4 inhibitor for diabetes.

    Targets:
    - Dissimilar to sitagliptin
    - Similar logP and TPSA
    - Formula: C16H15F6N5O
    """
    oracle_name = "sitagliptin"
    display_name = "Sitagliptin MPO"


class ZaleplonMPOObjective(TDCObjective):
    """
    Zaleplon MPO - sedative-hypnotic drug.

    Targets:
    - Similar to zaleplon
    - Different isomeric formula
    - Formula constraint: C19H17N3O2
    """
    oracle_name = "zaleplon"
    display_name = "Zaleplon MPO"


# ============================================================================
# Multi-Target Objectives (Combined)
# ============================================================================

class GSK3B_JNK3Objective(SmilesObjective):
    """
    Combined GSK3β + JNK3 activity (multi-target optimization).

    Returns: geometric_mean(GSK3B, JNK3)
    Both targets relevant for neurodegenerative diseases.
    """
    display_name = "GSK3β × JNK3"
    supports_batch = True

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self._gsk3b = None
        self._jnk3 = None

    def _ensure_oracles(self):
        if self._gsk3b is None:
            self._gsk3b = _get_oracle("GSK3B")
            self._jnk3 = _get_oracle("JNK3")

    def compute_from_smiles(self, smiles: str) -> float:
        self._ensure_oracles()
        try:
            gsk = float(self._gsk3b(smiles))
            jnk = float(self._jnk3(smiles))
            # Geometric mean, negated for minimization
            return -math.sqrt(max(0, gsk) * max(0, jnk))
        except Exception as e:
            logger.debug(f"GSK3B_JNK3 failed for {smiles}: {e}")
            return 0.0

    def compute_batch_from_smiles(self, smiles_list: List[str]) -> List[float]:
        if not smiles_list:
            return []

        self._ensure_oracles()
        try:
            gsk_scores = self._gsk3b(smiles_list)
            jnk_scores = self._jnk3(smiles_list)

            if not isinstance(gsk_scores, (list, tuple)):
                gsk_scores = [gsk_scores]
            if not isinstance(jnk_scores, (list, tuple)):
                jnk_scores = [jnk_scores]

            results = []
            for g, j in zip(gsk_scores, jnk_scores):
                g, j = float(g), float(j)
                results.append(-math.sqrt(max(0, g) * max(0, j)))
            return results

        except Exception as e:
            logger.warning(f"Batch multi-target failed: {e}")
            return [self.compute_from_smiles(s) for s in smiles_list]


class GSK3B_JNK3_QEDObjective(SmilesObjective):
    """
    GSK3β + JNK3 + QED (activity with drug-likeness).

    Returns: geometric_mean(GSK3B, JNK3, QED)
    Balances target activity with drug-like properties.
    """
    display_name = "GSK3β × JNK3 × QED"
    supports_batch = True

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self._gsk3b = None
        self._jnk3 = None

    def _ensure_oracles(self):
        if self._gsk3b is None:
            self._gsk3b = _get_oracle("GSK3B")
            self._jnk3 = _get_oracle("JNK3")

    def compute_from_smiles(self, smiles: str) -> float:
        from rdkit import Chem
        from rdkit.Chem import QED

        self._ensure_oracles()
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0

            gsk = float(self._gsk3b(smiles))
            jnk = float(self._jnk3(smiles))
            qed = QED.qed(mol)

            # Geometric mean of all three, negated
            product = max(0, gsk) * max(0, jnk) * max(0, qed)
            return -math.pow(product, 1/3) if product > 0 else 0.0

        except Exception as e:
            logger.debug(f"GSK3B_JNK3_QED failed for {smiles}: {e}")
            return 0.0

    def compute_batch_from_smiles(self, smiles_list: List[str]) -> List[float]:
        from rdkit import Chem
        from rdkit.Chem import QED

        if not smiles_list:
            return []

        self._ensure_oracles()

        try:
            gsk_scores = self._gsk3b(smiles_list)
            jnk_scores = self._jnk3(smiles_list)

            if not isinstance(gsk_scores, (list, tuple)):
                gsk_scores = [gsk_scores]
            if not isinstance(jnk_scores, (list, tuple)):
                jnk_scores = [jnk_scores]

            results = []
            for smiles, g, j in zip(smiles_list, gsk_scores, jnk_scores):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        results.append(0.0)
                        continue

                    g, j = float(g), float(j)
                    qed = QED.qed(mol)

                    product = max(0, g) * max(0, j) * max(0, qed)
                    score = -math.pow(product, 1/3) if product > 0 else 0.0
                    results.append(score)

                except Exception:
                    results.append(0.0)

            return results

        except Exception as e:
            logger.warning(f"Batch GSK3B_JNK3_QED failed: {e}")
            return [self.compute_from_smiles(s) for s in smiles_list]


class DRD2_QEDObjective(SmilesObjective):
    """
    DRD2 + QED (activity with drug-likeness).

    Returns: geometric_mean(DRD2, QED)
    Balances dopamine receptor activity with drug-like properties.
    """
    display_name = "DRD2 × QED"
    supports_batch = True

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self._drd2 = None

    def _ensure_oracle(self):
        if self._drd2 is None:
            self._drd2 = _get_oracle("DRD2")

    def compute_from_smiles(self, smiles: str) -> float:
        from rdkit import Chem
        from rdkit.Chem import QED

        self._ensure_oracle()
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0

            drd2 = float(self._drd2(smiles))
            qed = QED.qed(mol)

            # Geometric mean, negated
            product = max(0, drd2) * max(0, qed)
            return -math.sqrt(product) if product > 0 else 0.0

        except Exception as e:
            logger.debug(f"DRD2_QED failed for {smiles}: {e}")
            return 0.0

    def compute_batch_from_smiles(self, smiles_list: List[str]) -> List[float]:
        from rdkit import Chem
        from rdkit.Chem import QED

        if not smiles_list:
            return []

        self._ensure_oracle()

        try:
            drd2_scores = self._drd2(smiles_list)

            if not isinstance(drd2_scores, (list, tuple)):
                drd2_scores = [drd2_scores]

            results = []
            for smiles, d in zip(smiles_list, drd2_scores):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        results.append(0.0)
                        continue

                    d = float(d)
                    qed = QED.qed(mol)

                    product = max(0, d) * max(0, qed)
                    score = -math.sqrt(product) if product > 0 else 0.0
                    results.append(score)

                except Exception:
                    results.append(0.0)

            return results

        except Exception as e:
            logger.warning(f"Batch DRD2_QED failed: {e}")
            return [self.compute_from_smiles(s) for s in smiles_list]
