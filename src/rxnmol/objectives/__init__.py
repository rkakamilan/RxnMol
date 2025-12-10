"""
Molecular objective functions.

Simple interface:
    >>> from rxnmol.objectives import get_objective
    >>> obj = get_objective('qed')
    >>> score = obj.compute(mol)  # Returns float

    >>> list_objectives()  # Shows all available
    {'qed': 'QED (Drug-likeness)', 'seh': 'sEH Binding', ...}
"""

from .base import ObjectiveFunction, MolObjective, SmilesObjective
from .registry import get_objective, list_objectives

__all__ = [
    "ObjectiveFunction",
    "MolObjective",
    "SmilesObjective",
    "get_objective",
    "list_objectives",
]
