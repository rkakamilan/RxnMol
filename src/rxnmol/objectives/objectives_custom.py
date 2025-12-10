"""
Custom objective functions.

This module provides domain-specific or user-defined objective functions
that don't fit into standard categories.

Currently includes:
- J-score: Custom scoring function (logP - SA - ring_penalty)
- EE/OS score: Excitation energy and oscillator strength (requires RF models)
"""

import sys
import os
import logging

from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem.Descriptors import MolLogP

# Import SA scorer
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
try:
    import sascorer
    HAS_SASCORER = True
except ImportError:
    logging.warning("SA scorer not available. J-score will not work.")
    HAS_SASCORER = False

from .base import MolObjective

logger = logging.getLogger(__name__)


# =============================================================================
# J-Score (Custom Scoring Function)
# =============================================================================

class JScoreObjective(MolObjective):
    """
    J-score: Custom scoring function.

    Formula: J = logP - SA_score - ring_penalty - size_penalty

    Components:
    - logP: Lipophilicity (Wildman-Crippen)
    - SA_score: Synthetic accessibility [1-10]
    - ring_penalty: |num_ring_atoms - 6|
    - size_penalty: 0 if ≤ 38 heavy atoms, else 9999999

    The score is negated so higher J-score = better molecule.

    Use case: Optimizing for high logP, easy synthesis, moderate ring content, and small size.
    """

    def __init__(self, config):
        super().__init__(config)

        if not HAS_SASCORER:
            raise ImportError(
                "SA scorer required for J-score. "
                "Make sure sascorer.py is in RDKit contrib directory."
            )

        # Parameters (can be made configurable via config in future)
        self.n_max_heavy = 38
        self.target_ring_atoms = 6

    def compute(self, mol: Chem.Mol) -> float:
        """
        Compute J-score.

        Returns:
            J-score value (negated, so higher = better)
        """
        # Compute components
        logp = MolLogP(mol)

        # SA score
        try:
            sas = sascorer.calculateScore(mol)
        except Exception as e:
            logger.warning(f"SA score computation failed: {e}")
            # Use a penalty value
            sas = 10.0

        # Ring penalty
        ring_atoms = [atm for atm in mol.GetAtoms() if atm.IsInRing()]
        ring_penalty = abs(len(ring_atoms) - self.target_ring_atoms)

        # Size penalty
        num_heavy = mol.GetNumHeavyAtoms()
        if num_heavy <= self.n_max_heavy:
            size_penalty = 0
        else:
            size_penalty = 9999999

        # Compute J-score
        J_val = logp - sas - ring_penalty - size_penalty

        # Negate so higher = better
        J_val = -J_val

        return J_val


# =============================================================================
# Excitation Energy / Oscillator Strength Score
# =============================================================================

# Note: This objective requires RF models which are not included
# We include it here for documentation but it's not registered by default

class EEOSScoreObjective(MolObjective):
    """
    Excitation Energy and Oscillator Strength scoring.

    This is a custom objective for optimizing molecules based on:
    - Oscillator strength (OS)
    - Excitation energy (EE)
    - Synthetic accessibility
    - Similarity to reference molecule
    - Number of double bonds

    Note: Requires trained Random Forest models (rf_utils.OS_RF_model, rf_utils.EE_RF_model)
          which are not included by default.

    To use this objective:
    1. Ensure RF models are available
    2. Uncomment the @register_objective decorator below
    3. Configure weights in ObjectiveConfig
    """

    def __init__(self, config):
        super().__init__(config)

        # Try to import RF utilities
        try:
            import oldcodes.rf_utils as rf_utils
            self.rf_utils = rf_utils
            self.has_rf = True
        except ImportError:
            logger.error("RF utilities not found. EE/OS score will not work.")
            self.has_rf = False

        # Weights (should be configured via config)
        self.w_os = getattr(config, 'w_os', 1.0)
        self.w_sim = getattr(config, 'w_sim', 1.0)
        self.w_sascore = getattr(config, 'w_sascore', 1.0)
        self.w_energy = getattr(config, 'w_energy', 1.0)
        self.w_double = getattr(config, 'w_double', 1.0)

        # Target values
        self.target_ee = getattr(config, 'target_ee', 0.0)
        self.ref_mol = getattr(config, 'ref_mol', None)

    def compute(self, mol: Chem.Mol) -> float:
        """
        Compute EE/OS score.

        Returns:
            Combined score based on multiple factors
        """
        if not self.has_rf:
            raise RuntimeError("RF models not available for EE/OS score")

        # Convert to Kekule SMILES
        Chem.Kekulize(mol)
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)

        # Predict using RF models
        os = self.rf_utils.predict_rf(self.rf_utils.OS_RF_model, smiles)
        ee = self.rf_utils.predict_rf(self.rf_utils.EE_RF_model, smiles)

        # SA score
        try:
            sa = sascorer.calculateScore(mol)
        except:
            sa = 10.0

        # Similarity to reference
        if self.ref_mol is not None:
            from rdkit import DataStructs
            from rdkit.Chem import AllChem
            fp1 = AllChem.GetMorganFingerprint(mol, 2)
            fp2 = AllChem.GetMorganFingerprint(self.ref_mol, 2)
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        else:
            sim = 0.0

        # Delta excitation energy
        delta_ee = abs(ee - self.target_ee)

        # Number of double bonds
        n_double = len([b for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 2.0])

        # Compute combined score
        score = (
            -self.w_os * os
            - self.w_sim * sim
            + self.w_sascore * sa
            + self.w_energy * delta_ee / (1.0 + delta_ee)
            + self.w_double * n_double
        )

        return -score  # Negate so higher = better


# =============================================================================
# Register available objectives
# =============================================================================

logger.info("Registered 1 custom objective (j_score)")
logger.info("Note: EE/OS score is available but not registered (requires RF models)")
