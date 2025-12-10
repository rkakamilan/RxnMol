"""
Abstract base class for solution specifications.

A SolutionSpec defines:
1. How to represent solutions (genotype)
2. How to build molecules from genotypes (genotype → phenotype)
3. How to evaluate fitness (phenotype → score)
4. Which genetic operators to use
5. How to compute diversity metrics

This abstraction allows CSA to work with different solution representations
without changing the core algorithm.

Evaluation is delegated to ObjectiveFunction - SolutionSpec just calls
objective.evaluate() or objective.evaluate_batch().
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
import logging

import numpy as np
from rdkit import DataStructs

from ..objectives.base import SmilesObjective
from ..core.data_models import Candidate, RunContext

logger = logging.getLogger(__name__)


class SolutionSpec(ABC):
    """
    Abstract interface for solution representations.

    A SolutionSpec encapsulates:
    - Representation format (e.g., fragment lists, SMILES strings)
    - Building logic (genotype → phenotype)
    - Genetic operators
    - Diversity metrics

    Evaluation is delegated to self.objective (ObjectiveFunction instance).
    This allows the CSA engine to be representation-agnostic.

    Example:
        >>> spec = FragmentRouteSpec(context)
        >>> candidate = spec.random_candidate(generation=0)
        >>> spec.evaluate(candidate)
        >>> print(candidate.objective_value)
    """

    def __init__(self, context: RunContext):
        """
        Initialize solution specification.

        Args:
            context: Shared runtime context with config, RNGs, caches
        """
        self.context = context
        self.config = context.config
        self.rng = context.rng

        # Objective function - subclasses must set this
        self.objective = None

    # ========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # ========================================================================

    @abstractmethod
    def random_candidates(self, n: int, generation: int = 0) -> List[Candidate]:
        """
        Generate n random candidates using vectorized numpy sampling.

        Args:
            n: Number of candidates to generate
            generation: Generation number for provenance tracking

        Returns:
            List of n random candidates (genotypes only, not built or evaluated)
        """
        pass

    @abstractmethod
    def build(self, candidate: Candidate) -> Candidate:
        """
        Build a molecule from the candidate's genotype.

        This is the genotype → phenotype mapping.
        For fragment routes: fragment list → synthesized molecule
        For SMILES: SMILES string → validated molecule

        Args:
            candidate: Candidate with genotype to build

        Returns:
            Candidate with built phenotype (or failure reason)

        Example:
            >>> candidate = Candidate(genotype=['C', 'CC', 'CCO'], ...)
            >>> candidate = spec.build(candidate)
            >>> print(candidate.smiles)  # 'CCCCO'
        """
        pass

    @abstractmethod
    def build_batch(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Build phenotypes for a batch of candidates.

        This allows for batched operations (e.g. GPU inference) during building.

        Args:
            candidates: List of candidates with genotype to build

        Returns:
            List of candidates with built phenotypes
        """
        pass

    @abstractmethod
    def compute_fingerprint(self, candidate: Candidate):
        """
        Compute molecular fingerprint for diversity calculation.

        This method should return the RDKit fingerprint (mol_fp) that was
        already computed in build(). Used for backwards compatibility.

        Args:
            candidate: Candidate with built phenotype

        Returns:
            RDKit fingerprint (same as phenotype.mol_fp)

        Example:
            >>> fp = spec.compute_fingerprint(candidate)
            >>> # fp is the same as candidate.phenotype.mol_fp
        """
        pass

    @abstractmethod
    def distance(self, cand1: Candidate, cand2: Candidate) -> float:
        """
        Compute distance between two candidates for CSA cutoff.

        This is used by CSA to determine if candidates are similar enough
        to be considered for breeding.

        Args:
            cand1: First candidate
            cand2: Second candidate

        Returns:
            Distance in [0, 1], where 0 = identical, 1 = maximally different

        Example:
            >>> dist = spec.distance(cand1, cand2)
            >>> assert 0 <= dist <= 1
        """
        pass

    @abstractmethod
    def get_operators(self) -> Tuple[List[Callable], List[Callable]]:
        """
        Get genetic operators for this solution type.

        Returns:
            Tuple of (crossover_operators, mutation_operators)
            Each operator is a callable: (parent(s), context) → child

        Example:
            >>> crossovers, mutations = spec.get_operators()
            >>> # Use in CSA
            >>> child = crossovers[0](parent1, parent2, context)
        """
        pass

    @abstractmethod
    def local_optimize(self, candidate: Candidate, max_steps: int = 10) -> Tuple[Candidate, int]:
        """
        Perform local hill-climbing optimization on a candidate.

        This method iteratively applies small modifications to the candidate,
        keeping improvements and stopping when no improvement is found or
        max_steps is reached.

        Args:
            candidate: Candidate to optimize (must be evaluated)
            max_steps: Maximum number of optimization steps

        Returns:
            Tuple of (optimized_candidate, num_function_evaluations)
            - optimized_candidate: Best candidate found (may be original if no improvement)
            - num_function_evaluations: Number of objective evaluations performed

        Example:
            >>> candidate = spec.random_candidate(generation=0)
            >>> candidate = spec.evaluate(candidate)  # Must evaluate first
            >>> optimized, n_evals = spec.local_optimize(candidate, max_steps=10)
            >>> print(f"Improved: {candidate.score.value} -> {optimized.score.value}")
            >>> print(f"Function evals: {n_evals}")
        """
        pass

    def local_optimize_batch(
        self,
        candidates: List[Candidate],
        max_steps: int = 10,
        neighbors_per_candidate: int = 5,
    ) -> Tuple[List[Candidate], int]:
        """
        Perform local hill-climbing optimization on a batch of candidates.

        This default implementation runs them serially. Subclasses can override
        this with a more efficient batch-aware or parallel implementation.

        Args:
            candidates: List of candidates to optimize.
            max_steps: Maximum optimization steps per candidate.
            neighbors_per_candidate: Number of neighbors to generate per step (unused in serial).

        Returns:
            A tuple containing:
            - A list of the best candidates found for each input candidate.
            - The total number of function evaluations performed.
        """
        optimized_candidates = []
        total_evals = 0
        for candidate in candidates:
            opt_candidate, n_evals = self.local_optimize(candidate, max_steps=max_steps)
            optimized_candidates.append(opt_candidate)
            total_evals += n_evals

        logger.info(f"Serial local optimization complete: {total_evals} additional evaluations")
        return optimized_candidates, total_evals

    # ========================================================================
    # Evaluation Methods - Calls self.objective.compute()
    # ========================================================================

    def evaluate(self, candidate: Candidate) -> Candidate:
        """
        Evaluate a single candidate.

        Args:
            candidate: Candidate with built phenotype (mol)

        Returns:
            Same candidate with objective_value set
        """
        if self.objective is None:
            raise RuntimeError("No objective set. Subclass must set self.objective.")

        # Skip if already evaluated
        if candidate.objective_value is not None:
            return candidate

        if candidate.is_valid and candidate.mol:
            try:
                if isinstance(self.objective, SmilesObjective) and candidate.smiles:
                    candidate.objective_value = self.objective.compute_from_smiles(candidate.smiles)
                else:
                    candidate.objective_value = self.objective.compute(candidate.mol)
            except Exception as e:
                logger.warning(f"Objective failed for {candidate.smiles}: {e}")
                candidate.objective_value = self.objective.validity_penalty
        else:
            candidate.objective_value = self.objective.validity_penalty

        self.context.total_evaluations += 1
        return candidate

    def evaluate_batch(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Evaluate multiple candidates.

        Uses compute_batch() if objective supports it, otherwise serial.

        Args:
            candidates: List of candidates with built phenotypes

        Returns:
            Same list with objective_value set on each candidate
        """
        if not candidates:
            return candidates

        if self.objective is None:
            raise RuntimeError("No objective set. Subclass must set self.objective.")

        # Work only on unevaluated candidates
        pending_indices = [i for i, c in enumerate(candidates) if c.objective_value is None]
        if not pending_indices:
            return candidates

        valid_indices = []
        valid_inputs = []
        for i in pending_indices:
            cand = candidates[i]
            if cand.is_valid and cand.mol:
                valid_indices.append(i)
                # Prefer SMILES for SmilesObjective to avoid mol→SMILES conversion
                if isinstance(self.objective, SmilesObjective) and cand.smiles:
                    valid_inputs.append(cand.smiles)
                else:
                    valid_inputs.append(cand.mol)
            else:
                cand.objective_value = self.objective.validity_penalty

        # Compute scores for valid molecules
        if valid_inputs:
            if isinstance(self.objective, SmilesObjective):
                if self.objective.supports_batch and len(valid_inputs) > 1:
                    scores = self.objective.compute_batch_from_smiles(valid_inputs)
                else:
                    scores = [self.objective.compute_from_smiles(s) for s in valid_inputs]
            else:
                if self.objective.supports_batch and len(valid_inputs) > 1:
                    scores = self.objective.compute_batch(valid_inputs)
                else:
                    scores = [self.objective.compute(mol) for mol in valid_inputs]

            for idx, score in zip(valid_indices, scores):
                candidates[idx].objective_value = score

        # Count only the candidates we actually processed in this call
        self.context.total_evaluations += len(pending_indices)
        return candidates

    def validate_candidate(self, candidate: Candidate) -> bool:
        """
        Check if a candidate is valid for this solution type.

        Args:
            candidate: Candidate to validate

        Returns:
            True if valid, False otherwise
        """
        # Default: check genotype type matches
        return candidate.genotype_type == self.get_genotype_type()

    @abstractmethod
    def get_genotype_type(self) -> str:
        """
        Return the genotype type identifier for this spec.

        Returns:
            Type string (e.g., 'fragment_route', 'raw_smiles')
        """
        pass

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(type={self.get_genotype_type()})"


# ============================================================================
# Helper Functions
# ============================================================================

def tanimoto_distance(fp1, fp2) -> float:
    """
    Compute Tanimoto distance between two RDKit fingerprints.

    Uses RDKit's optimized TanimotoSimilarity for fast computation.
    Tanimoto similarity: |A ∩ B| / |A ∪ B|
    Tanimoto distance: 1 - similarity

    Args:
        fp1: First fingerprint (RDKit fingerprint or numpy array)
        fp2: Second fingerprint (RDKit fingerprint or numpy array)

    Returns:
        Distance in [0, 1]
    """
    # Try RDKit's optimized method first
    try:
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        return 1.0 - similarity
    except (TypeError, AttributeError):
        # Fallback to numpy for non-RDKit fingerprints
        if isinstance(fp1, np.ndarray) and isinstance(fp2, np.ndarray):
            intersection = np.sum(fp1 & fp2)
            union = np.sum(fp1 | fp2)

            if union == 0:
                return 1.0  # Both empty → maximally different

            similarity = intersection / union
            return 1.0 - similarity
        else:
            raise TypeError(f"Unsupported fingerprint types: {type(fp1)}, {type(fp2)}")
