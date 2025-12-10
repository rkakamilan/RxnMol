"""
Data models for Fragment-CSA optimization.

Defines core data structures:
- Candidate: Unified solution representation (genotype + phenotype + score + metadata)
- RunContext: Shared runtime state and resources

These models replace the old 'Solution' class with clearer semantics.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
from datetime import datetime
import numpy as np


# ============================================================================
# Candidate: Unified Solution Representation
# ============================================================================

@dataclass
class Candidate:
    """
    Unified candidate representation for molecular optimization.

    Replaces the old 'Solution' class with clearer separation of concerns:
    - genotype: The representation being optimized (e.g., fragment list, SMILES)
    - phenotype_mol: The built molecule (RDKit Mol)
    - smiles: The SMILES string of the phenotype
    - objective_value: The score (lower is better for CSA)
    - metadata: Provenance and other info
    """

    # Representation (genotype)
    genotype: Any  # List[str] for fragments, str for SMILES, etc.
    genotype_type: str = "unknown" # "fragment_route", "smiles_direct", etc.

    # Built molecule (phenotype)
    phenotype_mol: Any = None  # RDKit Mol object
    smiles: Optional[str] = None
    fingerprint: Any = None  # RDKit fingerprint (Morgan fingerprint for diversity)
    
    # Synthesis metadata (for fragment routes)
    intermediates: List[str] = field(default_factory=list)
    
    # Objective
    objective_value: Optional[float] = None  # Lower is better for CSA; None = not evaluated yet

    # Validity
    is_valid: bool = False
    
    # seed usage flag
    is_seed: bool = False
    
    # Metadata (Provenance, failure reasons, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # def __post_init__(self):
    #     """
    #     Ensure required metadata keys exist with sane defaults.
    #     """
    #     # All new candidates must declare seed-usage state.
    #     if 'is_used' not in self.metadata:
    #         self.metadata['is_used'] = False
    #     elif not isinstance(self.metadata['is_used'], bool):
    #         raise ValueError("Candidate.metadata['is_used'] must be a bool")

    # ========================================================================
    # Convenience Properties
    # ========================================================================

    @property
    def obj(self) -> float:
        """Alias for objective_value."""
        return self.objective_value
    
    @property
    def mol(self) -> Any:
        """Alias for phenotype_mol."""
        return self.phenotype_mol

    # ========================================================================
    # String Representation
    # ========================================================================

    def __repr__(self) -> str:
        """Concise string representation."""
        score_repr = f"{self.objective_value:.4f}" if self.objective_value is not None else "unevaluated"
        return (
            f"Candidate("
            f"type={self.genotype_type}, "
            f"score={score_repr}, "
            f"valid={self.is_valid}"
            f")"
        )

    def __str__(self) -> str:
        """Detailed string representation."""
        score_repr = f"{self.objective_value:.6f}" if self.objective_value is not None else "unevaluated"
        lines = [
            f"Candidate ({self.genotype_type}):",
            f"  Genotype: {self._format_genotype()}",
            f"  Score: {score_repr}",
            f"  Valid: {self.is_valid}",
        ]

        if self.smiles:
            lines.append(f"  SMILES: {self.smiles}")

        if 'operator' in self.metadata:
            lines.append(f"  Operator: {self.metadata['operator']}")

        return "\n".join(lines)

    def _format_genotype(self) -> str:
        """Format genotype for display."""
        if self.genotype_type == "fragment_route":
            if isinstance(self.genotype, list):
                return " >> ".join(self.genotype[:3]) + ("..." if len(self.genotype) > 3 else "")
        return str(self.genotype)[:50]


# ============================================================================
# RunContext: Shared Runtime State
# ============================================================================

@dataclass
class RunContext:
    """
    Centralized runtime context for a CSA optimization run.

    Holds configuration, RNGs, caches, and shared resources.
    Passed to all components that need access to global state.
    """

    # Configuration
    config: Any # 'MasterConfig'

    # Random state (for reproducibility)
    rng: np.random.Generator

    # Shared resources (loaded once, shared across all components)
    building_blocks: Optional[List[str]] = None
    reaction_model: Any = None  # Loaded reaction prediction model

    # Caches (for performance)
    reaction_cache: Any = None # PersistentReactionCache
    fingerprint_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    score_cache: Dict[str, float] = field(default_factory=dict)

    # Monitoring components (injected by orchestrator)
    metrics: Any = None # 'MetricsCollector'
    artifacts: Any = None # 'ArtifactStore'

    # Execution state
    current_iteration: int = 0
    total_evaluations: int = 0

    # ========================================================================
    # Cache Management
    # ========================================================================

    def get_reaction(self, key: str) -> Optional[str]:
        """
        Thread-safe reaction cache lookup.
        Delegates to persistent cache if available.
        """
        if self.reaction_cache:
            return self.reaction_cache.get(key)
        return None

    def set_reaction(self, key: str, product: str):
        """
        Thread-safe reaction cache insert.
        Delegates to persistent cache if available.
        """
        if self.reaction_cache:
            self.reaction_cache.set(key, product)

    def cache_stats(self) -> Dict[str, int]:
        """
        Return cache statistics.
        """
        stats = {
            'fingerprint_cache_size': len(self.fingerprint_cache),
            'score_cache_size': len(self.score_cache),
        }
        if self.reaction_cache:
            stats['reaction_cache_size'] = self.reaction_cache.size()
        return stats

    def __repr__(self) -> str:
        """String representation showing key state."""
        return (
            f"RunContext(\n"
            f"  iteration={self.current_iteration},\n"
            f"  evaluations={self.total_evaluations},\n"
            f")"
        )
