"""
Solution specification system for Fragment-CSA.

Provides abstract interface for different solution representations:
- FragmentRouteSpec: Fragment-based synthesis routes
- SmilesDirectSpec: Direct SMILES manipulation
- HybridSpec: Combination of both (future)
"""

from .base import SolutionSpec
from .fragments_route import FragmentRouteSpec
from .smiles_based import SmilesDirectSpec
from .reaction_mol import ReactionMolSpec
from .fragments_route_scaffhop import ScaffoldHopRouteSpec   

__all__ = [
    "SolutionSpec",
    "FragmentRouteSpec",
    "SmilesDirectSpec",
    "ReactionMolSpec",
    "ScaffoldHopRouteSpec"
]
