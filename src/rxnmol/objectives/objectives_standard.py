"""
Standard RDKit-based objective functions.

This module provides common molecular descriptors and properties from RDKit:
- QED (Quantitative Estimate of Drug-likeness)
- SA Score (Synthetic Accessibility)
- logP (Partition coefficient)
- Molecular weight
- TPSA (Topological Polar Surface Area)
- Number of rotatable bonds
- Number of aromatic rings
- Number of H-bond donors/acceptors
"""

import os
import sys
import logging

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Lipinski, Crippen
from rdkit import RDConfig

from .base import MolObjective

logger = logging.getLogger(__name__)

# Import SA scorer from RDKit contrib at module level
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# TODO: remove the try-except; if there is no sascorer, we want an ImportError
try:
    import sascorer
    HAS_SASCORER = True
except ImportError:
    logging.warning("SA scorer not available. SA score objectives will not work.")
    HAS_SASCORER = False
    sascorer = None


# =============================================================================
# Drug-likeness and Synthetic Accessibility
# =============================================================================

class QEDObjective(MolObjective):
    """
    Quantitative Estimate of Drug-likeness (QED).

    QED is a composite score based on 8 molecular properties:
    - Molecular weight
    - logP
    - Number of H-bond donors/acceptors
    - Polar surface area
    - Number of rotatable bonds
    - Number of aromatic rings
    - Structural alerts

    Range: [0, 1], higher is better (more drug-like)

    Reference:
        Bickerton et al. (2012). Quantifying the chemical beauty of drugs.
        Nature Chemistry, 4(2), 90-98.
    """

    def compute(self, mol: Chem.Mol) -> float:
        """Compute QED score."""
        # Maximize QED -> Return negative
        return -QED.qed(mol)


class SAScoreObjective(MolObjective):
    """
    Synthetic Accessibility Score.

    Estimates how easy it is to synthesize a molecule based on:
    - Complexity of molecular structure
    - Similarity to known synthesizable molecules

    Original range: [1, 10]
    - 1 = very easy to synthesize
    - 10 = very difficult to synthesize

    CSA minimizes this directly, so lower raw SA = easier = better.

    Reference:
        Ertl & Schuffenhauer (2009). Estimation of synthetic accessibility score
        of drug-like molecules based on molecular complexity and fragment contributions.
        Journal of Cheminformatics, 1(1), 8.
    """
    def __init__(self, name: str = None, validity_penalty: float = 999999.0, config=None):
        if not HAS_SASCORER:
            raise ImportError("SA scorer not available. Cannot create SAScoreObjective.")
        super().__init__(name=name, validity_penalty=validity_penalty, config=config)

    def compute(self, mol: Chem.Mol) -> float:
        """Compute raw SA score [1, 10]. CSA minimizes → finds easiest to synthesize."""
        return sascorer.calculateScore(mol)


# =============================================================================
# Physicochemical Properties
# =============================================================================

class LogPObjective(MolObjective):
    """
    Octanol-water partition coefficient (logP).

    Measures lipophilicity (how well a molecule dissolves in fats vs water).

    Typical drug range: [0, 5]
    - < 0: Very hydrophilic
    - 0-3: Moderate
    - 3-5: Lipophilic
    - > 5: Very lipophilic (poor solubility)

    Uses Wildman-Crippen method.
    """

    def compute(self, mol: Chem.Mol) -> float:
        """Compute logP using Crippen method."""
        # Maximize LogP -> Return negative
        return -Crippen.MolLogP(mol)


class MolecularWeightObjective(MolObjective):
    """
    Molecular weight (Daltons).

    Typical drug range: [150, 500]
    - Lipinski's rule of 5: MW < 500
    """

    def compute(self, mol: Chem.Mol) -> float:
        """Compute molecular weight."""
        # Minimize MW -> Return positive
        return Descriptors.MolWt(mol)


class TPSAObjective(MolObjective):
    """
    Topological Polar Surface Area (Ų).

    Sum of surfaces of polar atoms (O, N) in a molecule.
    Correlates with drug bioavailability.

    Typical ranges:
    - TPSA < 60: Good CNS penetration
    - TPSA < 140: Good oral bioavailability
    """

    def compute(self, mol: Chem.Mol) -> float:
        """Compute TPSA."""
        return Descriptors.TPSA(mol)


# =============================================================================
# Molecular Complexity
# =============================================================================

class NumRotatableBondsObjective(MolObjective):
    """
    Number of rotatable bonds.

    Measure of molecular flexibility.
    - Lipinski's rule of 5: ≤ 10 rotatable bonds
    """

    def compute(self, mol: Chem.Mol) -> float:
        """Count rotatable bonds."""
        return float(Lipinski.NumRotatableBonds(mol))


class NumAromaticRingsObjective(MolObjective):
    """Number of aromatic rings in the molecule."""

    def compute(self, mol: Chem.Mol) -> float:
        """Count aromatic rings."""
        return float(Descriptors.NumAromaticRings(mol))


class NumRingsObjective(MolObjective):
    """Total number of rings in the molecule."""

    def compute(self, mol: Chem.Mol) -> float:
        """Count all rings."""
        return float(Lipinski.RingCount(mol))


class NumHeavyAtomsObjective(MolObjective):
    """Number of heavy (non-hydrogen) atoms."""

    def compute(self, mol: Chem.Mol) -> float:
        """Count heavy atoms."""
        return float(Lipinski.HeavyAtomCount(mol))


# =============================================================================
# Hydrogen Bonding
# =============================================================================

class NumHDonorsObjective(MolObjective):
    """
    Number of hydrogen bond donors (OH, NH groups).

    Lipinski's rule of 5: ≤ 5 donors
    """

    def compute(self, mol: Chem.Mol) -> float:
        """Count H-bond donors."""
        return float(Lipinski.NumHDonors(mol))


class NumHAcceptorsObjective(MolObjective):
    """
    Number of hydrogen bond acceptors (N, O atoms).

    Lipinski's rule of 5: ≤ 10 acceptors
    """

    def compute(self, mol: Chem.Mol) -> float:
        """Count H-bond acceptors."""
        return float(Lipinski.NumHAcceptors(mol))


# =============================================================================
# Lipinski Rule of 5
# =============================================================================

class LipinskiViolationsObjective(MolObjective):
    """
    Number of Lipinski rule of 5 violations.

    Rules:
    - Molecular weight < 500
    - logP < 5
    - H-bond donors ≤ 5
    - H-bond acceptors ≤ 10

    Range: [0, 4]
    - 0 = No violations (drug-like)
    - 4 = All rules violated (not drug-like)

    We return -violations so higher = better.
    """

    def compute(self, mol: Chem.Mol) -> float:
        """Count Lipinski violations."""
        violations = 0

        # MW < 500
        if Descriptors.MolWt(mol) > 500:
            violations += 1

        # logP < 5
        if Crippen.MolLogP(mol) > 5:
            violations += 1

        # HBD ≤ 5
        if Lipinski.NumHDonors(mol) > 5:
            violations += 1

        # HBA ≤ 10
        if Lipinski.NumHAcceptors(mol) > 10:
            violations += 1

        # Return positive violations (minimize, 0 is best)
        return float(violations)


# =============================================================================
# Combined Scores
# =============================================================================

class QEDSAObjective(MolObjective):
    """
    Combined QED and SA score.

    Returns: QED * SA_normalized

    This balances drug-likeness with synthetic accessibility.
    Range: [0, 1], higher is better
    """

    def compute(self, mol: Chem.Mol) -> float:
        """Compute combined QED * SA score."""
        qed_score = QED.qed(mol)

        if not HAS_SASCORER:
            # If SA not available, just return QED
            logger.warning("SA scorer not available, using QED only")
            return qed_score

        # Compute normalized SA score
        raw_sa = sascorer.calculateScore(mol)
        sa_normalized = (10 - raw_sa) / 9.0
        sa_normalized = max(0.0, min(1.0, sa_normalized))

        # Maximize QED*SA -> Return negative
        return -(qed_score * sa_normalized)


