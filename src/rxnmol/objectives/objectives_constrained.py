"""
Constrained docking objectives: docking score × property penalty factor.

Two penalty types:
- MW penalty: Penalizes molecules above Lipinski MW threshold (500 Da).
  Factor linearly decays from 1.0 at MW ≤ 500 to 0.0 at MW ≥ 800.
- Lipinski penalty: Penalizes Lipinski rule-of-5 violations (MW, logP, HBD, HBA).
  Factor = (4 - n_violations) / 4, so 0 violations → 1.0, 4 violations → 0.0.

Applied to all 8 docking tasks:
  MPNN-Docking: sEH, CB1 (Raw), CB1 (ZScore), CB1 (MinMax)
  GNN-Docking:  MPro, BTK, ALK, H1N1 NA
"""

from typing import List

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen

from .base import MolObjective

# Lazy imports — avoid circular and heavy loading at import time
_SEHObjective = None
_CB1RawObjective = None
_CB1ZscoreObjective = None
_CB1MinMaxObjective = None
_MProObjective = None
_BTKObjective = None
_ALKObjective = None
_H1N1NAObjective = None


def _get_mpnn_classes():
    global _SEHObjective, _CB1RawObjective, _CB1ZscoreObjective, _CB1MinMaxObjective
    if _SEHObjective is None:
        from .objectives_synflownet import (
            SEHObjective, CB1RawObjective, CB1ZscoreObjective, CB1MinMaxObjective,
        )
        _SEHObjective = SEHObjective
        _CB1RawObjective = CB1RawObjective
        _CB1ZscoreObjective = CB1ZscoreObjective
        _CB1MinMaxObjective = CB1MinMaxObjective
    return _SEHObjective, _CB1RawObjective, _CB1ZscoreObjective, _CB1MinMaxObjective


def _get_gnn_classes():
    global _MProObjective, _BTKObjective, _ALKObjective, _H1N1NAObjective
    if _MProObjective is None:
        from .objectives_csearch import (
            MProObjective, BTKObjective, ALKObjective, H1N1NAObjective,
        )
        _MProObjective = MProObjective
        _BTKObjective = BTKObjective
        _ALKObjective = ALKObjective
        _H1N1NAObjective = H1N1NAObjective
    return _MProObjective, _BTKObjective, _ALKObjective, _H1N1NAObjective


# =============================================================================
# Penalty functions
# =============================================================================

# MW penalty: linear decay from 1.0 at MW ≤ 500 to 0.0 at MW ≥ 800
MW_THRESHOLD = 500   # Da — Lipinski limit
MW_UPPER = 800       # Da — zero factor above this


def mw_factor(mol: Chem.Mol) -> float:
    """MW penalty factor ∈ [0, 1]. 1.0 if MW ≤ 500, 0.0 if MW ≥ 800."""
    mw = Descriptors.MolWt(mol)
    return max(0.0, min(1.0, (MW_UPPER - mw) / (MW_UPPER - MW_THRESHOLD)))


def lipinski_factor(mol: Chem.Mol) -> float:
    """Lipinski penalty factor ∈ [0, 1]. 1.0 if no violations, 0.0 if all 4."""
    violations = 0
    if Descriptors.MolWt(mol) > 500:
        violations += 1
    if Crippen.MolLogP(mol) > 5:
        violations += 1
    if Lipinski.NumHDonors(mol) > 5:
        violations += 1
    if Lipinski.NumHAcceptors(mol) > 10:
        violations += 1
    return (4 - violations) / 4.0


# =============================================================================
# Base classes: Docking × Penalty
# =============================================================================

class _MWPenalizedDocking(MolObjective):
    """
    Base: docking_score × mw_factor.

    Subclasses set `_docking_cls` to the specific docking objective.
    The combined score is: -(positive_docking × mw_factor) for CSA minimization.
    """
    supports_batch = True
    _docking_cls = None  # override

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.docking = self._docking_cls(**kwargs)

    def compute(self, mol: Chem.Mol) -> float:
        if not mol:
            return 0.0
        dock_val = self.docking.compute(mol)
        dock_pos = -dock_val   # universal: convert CSA form → positive (higher=better)
        return -(dock_pos * mw_factor(mol))

    def compute_batch(self, mols: List[Chem.Mol]) -> List[float]:
        if not mols:
            return []
        dock_scores = self.docking.compute_batch(mols)
        return [
            -((-d) * mw_factor(m)) if m else 0.0
            for d, m in zip(dock_scores, mols)
        ]


class _LipinskiPenalizedDocking(MolObjective):
    """
    Base: docking_score × lipinski_factor.

    Subclasses set `_docking_cls` to the specific docking objective.
    The combined score is: -(positive_docking × lipinski_factor) for CSA minimization.
    """
    supports_batch = True
    _docking_cls = None  # override

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.docking = self._docking_cls(**kwargs)

    def compute(self, mol: Chem.Mol) -> float:
        if not mol:
            return 0.0
        dock_val = self.docking.compute(mol)
        dock_pos = -dock_val
        return -(dock_pos * lipinski_factor(mol))

    def compute_batch(self, mols: List[Chem.Mol]) -> List[float]:
        if not mols:
            return []
        dock_scores = self.docking.compute_batch(mols)
        return [
            -((-d) * lipinski_factor(m)) if m else 0.0
            for d, m in zip(dock_scores, mols)
        ]


# =============================================================================
# MPNN-Docking × MW
# =============================================================================

class SEHMWObjective(_MWPenalizedDocking):
    """sEH binding × MW penalty. Rewards potent, low-MW ligands."""
    @property
    def _docking_cls(self):
        return _get_mpnn_classes()[0]


class CB1RawMWObjective(_MWPenalizedDocking):
    """CB1 raw docking × MW penalty."""
    @property
    def _docking_cls(self):
        return _get_mpnn_classes()[1]


class CB1ZscoreMWObjective(_MWPenalizedDocking):
    """CB1 z-score docking × MW penalty."""
    @property
    def _docking_cls(self):
        return _get_mpnn_classes()[2]


class CB1MinMaxMWObjective(_MWPenalizedDocking):
    """CB1 min-max docking × MW penalty."""
    @property
    def _docking_cls(self):
        return _get_mpnn_classes()[3]


# =============================================================================
# GNN-Docking × MW
# =============================================================================

class MProMWObjective(_MWPenalizedDocking):
    """MPro docking × MW penalty."""
    @property
    def _docking_cls(self):
        return _get_gnn_classes()[0]


class BTKMWObjective(_MWPenalizedDocking):
    """BTK docking × MW penalty."""
    @property
    def _docking_cls(self):
        return _get_gnn_classes()[1]


class ALKMWObjective(_MWPenalizedDocking):
    """ALK docking × MW penalty."""
    @property
    def _docking_cls(self):
        return _get_gnn_classes()[2]


class H1N1NAMWObjective(_MWPenalizedDocking):
    """H1N1 NA docking × MW penalty."""
    @property
    def _docking_cls(self):
        return _get_gnn_classes()[3]


# =============================================================================
# MPNN-Docking × Lipinski
# =============================================================================

class SEHLipinskiObjective(_LipinskiPenalizedDocking):
    """sEH binding × Lipinski penalty."""
    @property
    def _docking_cls(self):
        return _get_mpnn_classes()[0]


class CB1RawLipinskiObjective(_LipinskiPenalizedDocking):
    """CB1 raw docking × Lipinski penalty."""
    @property
    def _docking_cls(self):
        return _get_mpnn_classes()[1]


class CB1ZscoreLipinskiObjective(_LipinskiPenalizedDocking):
    """CB1 z-score docking × Lipinski penalty."""
    @property
    def _docking_cls(self):
        return _get_mpnn_classes()[2]


class CB1MinMaxLipinskiObjective(_LipinskiPenalizedDocking):
    """CB1 min-max docking × Lipinski penalty."""
    @property
    def _docking_cls(self):
        return _get_mpnn_classes()[3]


# =============================================================================
# GNN-Docking × Lipinski
# =============================================================================

class MProLipinskiObjective(_LipinskiPenalizedDocking):
    """MPro docking × Lipinski penalty."""
    @property
    def _docking_cls(self):
        return _get_gnn_classes()[0]


class BTKLipinskiObjective(_LipinskiPenalizedDocking):
    """BTK docking × Lipinski penalty."""
    @property
    def _docking_cls(self):
        return _get_gnn_classes()[1]


class ALKLipinskiObjective(_LipinskiPenalizedDocking):
    """ALK docking × Lipinski penalty."""
    @property
    def _docking_cls(self):
        return _get_gnn_classes()[2]


class H1N1NALipinskiObjective(_LipinskiPenalizedDocking):
    """H1N1 NA docking × Lipinski penalty."""
    @property
    def _docking_cls(self):
        return _get_gnn_classes()[3]
