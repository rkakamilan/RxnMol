"""
SMILES-based SolutionSpec Implementation

This module implements direct SMILES string manipulation for molecular optimization,
following the original MolFinder approach. Molecules are represented directly as
SMILES strings and modified using graph-based genetic operators.
"""

import logging
from typing import Any, List, Optional, Tuple, Callable, Dict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs
from rdkit import RDLogger
import copy

from .base import SolutionSpec
from ..core.data_models import Candidate, RunContext
from ..core.config import ObjectiveConfig
from ..utils.building_blocks import load_building_blocks
from .smiles_operators import crossover_two_smiles
from .smiles_operators import add_atom, delete_atom, replace_atom

# Disable RDKit warnings
RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


class SmilesDirectSpec(SolutionSpec):
    """
    SMILES-based solution specification.

    Represents molecules directly as SMILES strings and uses graph-based
    genetic operators (crossover, atom addition/deletion/replacement) for
    exploration.
    """

    def __init__(self, context: RunContext):
        """
        Initialize SMILES-based solution specification.

        Args:
            context: Runtime context with configuration and shared state
        """
        super().__init__(context)
        # Store context
        self.context = context
        self.rng = context.rng
        self.building_blocks_file = context.config.data.building_blocks_file

        # Load initial SMILES pool if provided
        self.building_blocks: List[str] = self._load_building_blocks(
            self.building_blocks_file
        )

        # Create objective instance ONCE
        self.objective = self._create_objective()

    def _create_objective(self):
        from ..objectives import get_objective
        return get_objective(
            name=self.context.config.objective.name,
            validity_penalty=self.context.config.objective.validity_penalty
        )

    def _load_building_blocks(self, building_blocks_file) -> List[str]:
        """Load initial building blocks (SMILES) from a file."""
        return load_building_blocks(building_blocks_file)


    def random_candidates(self, n: int, generation: int = 0) -> List[Candidate]:
        """Generate n random candidates efficiently."""
        indices = self.rng.integers(0, len(self.building_blocks), size=n)
        smiles_array = self.building_blocks[indices]

        return [
            Candidate(
                genotype=smiles,
                metadata={'generation': generation, 'operator': 'random_init', 'genotype_type': 'smiles_direct'}
            )
            for smiles in smiles_array
        ]

    def build(self, candidate: Candidate) -> Candidate:
        """Build molecule from SMILES string (wrapper for batch)."""
        return self.build_batch([candidate])[0]

    def build_batch(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Build phenotypes for a batch of candidates.
        
        For SMILES, this involves parsing SMILES, kekulizing, and computing fingerprints.
        This is CPU-bound, so we could use multiprocessing here if batch size is large,
        but typically evaluate_batch handles parallelism.
        
        Here we just do it serially or use the pool if available and worth it.
        Given the overhead, serial is often fine for simple RDKit ops unless batch is huge.
        """
        for candidate in candidates:
            smiles = candidate.genotype
            
            # Handle if genotype is already a Mol object
            if not isinstance(smiles, str):
                if hasattr(smiles, 'GetNumAtoms'):
                    mol = smiles
                    smiles = Chem.MolToSmiles(mol)
                else:
                    candidate.is_valid = False
                    candidate.metadata['failure_reason'] = f"Invalid genotype type: {type(smiles)}"
                    continue
            else:
                mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                candidate.is_valid = False
                candidate.metadata['failure_reason'] = "Invalid SMILES string"
                continue

            # Check size constraint
            if self.context.config.solution.max_mol_size is not None:
                if mol.GetNumAtoms() > self.context.config.solution.max_mol_size:
                    candidate.is_valid = False
                    candidate.metadata['failure_reason'] = f"Molecule too large ({mol.GetNumAtoms()} > {self.context.config.solution.max_mol_size})"
                    continue

            # Kekulize
            try:
                Chem.Kekulize(mol)
                kekulized_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
            except Exception:
                kekulized_smiles = smiles

            # Compute fingerprint
            mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=True)

            candidate.phenotype_mol = mol
            candidate.smiles = kekulized_smiles
            candidate.fingerprint = mol_fp
            candidate.is_valid = True
            
        return candidates

    def crossover(self, parent1: Candidate, parent2: Candidate, generation: int) -> Candidate:
        """Crossover two SMILES strings."""
        smiles1 = parent1.genotype
        smiles2 = parent2.genotype

        try:
            child_smiles = crossover_two_smiles(smiles1, smiles2)
            if child_smiles is None:
                child_smiles = smiles1 if self.rng.random() < 0.5 else smiles2
        except Exception:
            child_smiles = smiles1 if self.rng.random() < 0.5 else smiles2

        return Candidate(
            genotype=child_smiles,
            metadata={
                'generation': generation,
                'parents': [parent1.smiles, parent2.smiles],
                'operator': 'crossover',
                'genotype_type': 'smiles_direct'
            }
        )

    def mutate(self, parent: Candidate, generation: int) -> Candidate:
        """Mutate a SMILES string."""
        smiles = parent.genotype
        
        # Ensure string
        if not isinstance(smiles, str):
            if hasattr(smiles, 'GetNumAtoms'):
                smiles = Chem.MolToSmiles(smiles)
            else:
                return parent
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return parent

        operators = ['add', 'delete', 'replace']
        weights = [
            self.context.config.operators.mutation_add_prob,
            self.context.config.operators.mutation_delete_prob,
            self.context.config.operators.mutation_replace_prob
        ]
        
        # Normalize weights
        total = sum(weights)
        if total == 0:
            weights = [1.0, 1.0, 1.0]
            total = 3.0
        weights = [w / total for w in weights]

        op = self.rng.choice(operators, p=weights)

        if op == 'add':
            result = add_atom(smiles)
            operator_name = 'mutate_add'
        elif op == 'delete':
            result = delete_atom(smiles)
            operator_name = 'mutate_delete'
        else:
            result = replace_atom(smiles)
            operator_name = 'mutate_replace'

        if result is None:
            child_smiles = smiles
            operator_name += '_failed'
        elif isinstance(result, str):
            child_smiles = result
        elif isinstance(result, tuple):
            child_smiles = result[1] if len(result) == 2 else result[0]
        else:
            try:
                child_smiles = Chem.MolToSmiles(result)
            except:
                child_smiles = smiles
                operator_name += '_failed'

        if not isinstance(child_smiles, str):
            try:
                child_smiles = Chem.MolToSmiles(child_smiles)
            except:
                child_smiles = smiles

        return Candidate(
            genotype=child_smiles,
            metadata={
                'generation': generation,
                'parents': [parent.smiles],
                'operator': operator_name,
                'genotype_type': 'smiles_direct'
            }
        )

    def compute_fingerprint(self, candidate: Candidate) -> Any:
        """Return fingerprint."""
        return candidate.fingerprint

    def get_genotype_type(self) -> str:
        return 'smiles_direct'

    def get_operators(self) -> Tuple[List[Callable], List[Callable]]:
        def crossover_wrapper(parent1, parent2):
            gen = max(parent1.metadata.get('generation', 0), parent2.metadata.get('generation', 0)) + 1
            return self.crossover(parent1, parent2, gen)
        
        def mutation_wrapper(parent):
            gen = parent.metadata.get('generation', 0) + 1
            return self.mutate(parent, gen)
        
        return ([crossover_wrapper], [mutation_wrapper])

    def distance(self, candidate1: Candidate, candidate2: Candidate) -> float:
        if candidate1.fingerprint is None or candidate2.fingerprint is None:
            return 1.0
        return 1.0 - DataStructs.TanimotoSimilarity(candidate1.fingerprint, candidate2.fingerprint)

    def local_optimize(self, candidate: Candidate, max_steps: int = 10) -> Tuple[Candidate, int]:
        """Local optimization using batch evaluation for efficiency."""
        if candidate.objective_value == float('inf'):
            candidate = self.build(candidate)
            self.evaluate(candidate)

        best_candidate = candidate
        best_score = candidate.objective_value
        n_evals = 0

        for step in range(max_steps):
            neighbor = self.mutate(best_candidate, generation=best_candidate.metadata.get('generation', 0))
            neighbor = self.build(neighbor)
            # Use batch evaluation to reuse persistent pool
            neighbors_evaluated = self.evaluate_batch([neighbor])
            neighbor = neighbors_evaluated[0]
            n_evals += 1

            if neighbor.is_valid and neighbor.objective_value < best_score:
                best_candidate = neighbor
                best_score = neighbor.objective_value
        
        return best_candidate, n_evals

    def local_optimize_batch(
        self,
        candidates: List[Candidate],
        max_steps: int = 3,
        neighbors_per_candidate: int = 5,
    ) -> Tuple[List[Candidate], int]:
        """
        Batch local optimization (greedy hill climbing) for multiple candidates.

        For each candidate, performs greedy hill climbing:
          - Generate N neighbors (mutations of current best)
          - Build & evaluate all neighbors in batch
          - If best neighbor improves → replace current with it
          - Repeat until max_steps or convergence

        Args:
            candidates: List of candidates (already built and evaluated)
            max_steps: Maximum hill climbing steps (default 3)
            neighbors_per_candidate: Neighbors per candidate per step (default 5)

        Returns:
            Tuple of (optimized candidates, total evaluations)
        """
        if not candidates:
            return [], 0

        best_candidates = list(candidates)
        total_evals = 0
        n_candidates = len(candidates)

        # Track convergence: stop a candidate after N steps without improvement
        no_improvement_count = [0] * n_candidates
        early_stop_threshold = 3

        for step in range(max_steps):
            # Find active candidates (not yet converged)
            active_indices = [i for i in range(n_candidates)
                            if no_improvement_count[i] < early_stop_threshold]

            if not active_indices:
                logger.debug(f"Local opt step {step}: all {n_candidates} candidates converged")
                break

            # Generate neighbors for all active candidates
            all_neighbors = []
            parent_indices = []

            for i in active_indices:
                parent = best_candidates[i]
                for _ in range(neighbors_per_candidate):
                    neighbor = self.mutate(parent, generation=parent.metadata.get('generation', 0) + 1)
                    if neighbor:
                        all_neighbors.append(neighbor)
                        parent_indices.append(i)

            if not all_neighbors:
                logger.debug(f"Local opt step {step}: no neighbors generated")
                break

            # Build all neighbors in one batch
            logger.debug(
                f"Local opt step {step}: {len(all_neighbors)} neighbors "
                f"for {len(active_indices)} active candidates"
            )
            built_neighbors = self.build_batch(all_neighbors)

            # Filter valid and evaluate
            valid_neighbors = []
            valid_parent_indices = []

            for i, neighbor in enumerate(built_neighbors):
                if neighbor.is_valid:
                    valid_neighbors.append(neighbor)
                    valid_parent_indices.append(parent_indices[i])

            if not valid_neighbors:
                for i in active_indices:
                    no_improvement_count[i] += 1
                continue

            evaluated_neighbors = self.evaluate_batch(valid_neighbors)
            total_evals += len(evaluated_neighbors)

            # Greedy selection - keep best neighbor if it improves
            best_per_parent: Dict[int, Candidate] = {}

            for neighbor, parent_idx in zip(evaluated_neighbors, valid_parent_indices):
                if parent_idx not in best_per_parent:
                    best_per_parent[parent_idx] = neighbor
                elif neighbor.objective_value < best_per_parent[parent_idx].objective_value:
                    best_per_parent[parent_idx] = neighbor

            # Update if improved
            improved_count = 0
            for parent_idx, best_neighbor in best_per_parent.items():
                current = best_candidates[parent_idx]
                if best_neighbor.objective_value < current.objective_value:
                    best_candidates[parent_idx] = best_neighbor
                    no_improvement_count[parent_idx] = 0
                    improved_count += 1
                else:
                    no_improvement_count[parent_idx] += 1

            # Update no-improvement for candidates with no valid neighbors
            for i in active_indices:
                if i not in best_per_parent:
                    no_improvement_count[i] += 1

            converged = sum(1 for c in no_improvement_count if c >= early_stop_threshold)
            logger.debug(
                f"Local opt step {step}: {improved_count}/{len(active_indices)} improved, "
                f"{converged}/{n_candidates} converged"
            )

        return best_candidates, total_evals
