"""
Base classes for objective functions.

Simple interface:
- ObjectiveFunction.compute(mol) → float
- ObjectiveFunction.compute_batch(mols) → List[float]

SolutionSpec handles Candidate → mol conversion and assigns objective_value.
"""

from abc import ABC, abstractmethod
from typing import List
import logging

from rdkit import Chem

logger = logging.getLogger(__name__)


class ObjectiveFunction(ABC):
    """
    Abstract base class for objective functions.

    Subclasses implement compute(mol) → float.
    That's it. Keep it simple.

    Attributes:
        name: Identifier (e.g., 'qed', 'seh')
        validity_penalty: Score for invalid molecules (default: 999999.0)
        supports_batch: If True, compute_batch() is efficient (override it)

    Usage:
        >>> obj = QEDObjective(name='qed')
        >>> score = obj.compute(mol)  # Returns float
        >>> scores = obj.compute_batch([mol1, mol2])  # Returns List[float]
    """

    # Override in subclasses that support efficient batching
    supports_batch: bool = False

    def __init__(self, name: str = None, validity_penalty: float = 999999.0, **kwargs):
        """
        Initialize objective.

        Args:
            name: Objective name
            validity_penalty: Penalty for invalid molecules
            **kwargs: Subclass-specific parameters
        """
        self.name = name
        self.validity_penalty = validity_penalty
        self._kwargs = kwargs

    @abstractmethod
    def compute(self, mol: Chem.Mol) -> float:
        """
        Compute objective value for a molecule.

        Args:
            mol: RDKit molecule

        Returns:
            Objective value (LOWER is better for CSA)

        Note:
            If your objective is "higher is better" (e.g. QED), return -value.
        """
        pass

    def compute_batch(self, mols: List[Chem.Mol]) -> List[float]:
        """
        Compute objective values for a batch of molecules.

        Override in subclasses that support efficient batching
        (e.g., GPU models, vectorized oracles).

        Default: serial fallback.

        Args:
            mols: List of RDKit molecules

        Returns:
            List of objective values
        """
        return [self.compute(mol) for mol in mols]

    def __call__(self, mol: Chem.Mol) -> float:
        """Make objective callable: obj(mol) → score."""
        return self.compute(mol)

    def __repr__(self) -> str:
        batch_info = " [batch]" if self.supports_batch else ""
        return f"{self.__class__.__name__}(name='{self.name}'{batch_info})"


class MolObjective(ObjectiveFunction):
    """
    Base class for objectives that evaluate RDKit Mol objects.

    Use for: QED, SA score, logP, molecular weight, etc.

    Example:
        >>> class QEDObjective(MolObjective):
        ...     def compute(self, mol):
        ...         return -QED.qed(mol)  # Negate for minimization
    """
    pass


class SmilesObjective(ObjectiveFunction):
    """
    Base class for objectives that evaluate SMILES strings.

    Use for: TDC oracles, GuacaMol, external APIs.

    Subclasses implement compute_from_smiles(smiles) → float.
    The base class handles mol → SMILES conversion.

    Example:
        >>> class GSK3BObjective(SmilesObjective):
        ...     def compute_from_smiles(self, smiles):
        ...         return -self.oracle(smiles)
    """

    def compute(self, mol: Chem.Mol) -> float:
        """
        Evaluate from either an RDKit Mol or a SMILES string.

        The Candidate already carries both mol and smiles, so if a string
        is provided we skip the conversion entirely.
        """
        if isinstance(mol, str):
            return self.compute_from_smiles(mol)
        smiles = Chem.MolToSmiles(mol)
        return self.compute_from_smiles(smiles)

    @abstractmethod
    def compute_from_smiles(self, smiles: str) -> float:
        """
        Compute objective from SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            Objective value (LOWER is better)
        """
        pass

    def compute_batch(self, mols: List[Chem.Mol]) -> List[float]:
        """
        Batch compute; accepts list of mols or SMILES strings.

        If SMILES strings are provided, avoids MolToSmiles conversion.
        """
        smiles_list = [
            mol if isinstance(mol, str) else Chem.MolToSmiles(mol)
            for mol in mols
        ]
        return self.compute_batch_from_smiles(smiles_list)

    def compute_batch_from_smiles(self, smiles_list: List[str]) -> List[float]:
        """
        Batch compute from SMILES strings.

        Override for objectives that support batch evaluation.
        Default: serial fallback.
        """
        return [self.compute_from_smiles(s) for s in smiles_list]
