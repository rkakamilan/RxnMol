"""
Conformational Space Annealing (CSA) optimization engine.

This module implements the CSA algorithm in a representation-agnostic way.
The algorithm works with any SolutionSpec that defines how solutions are
generated, evaluated, and varied.

Key Features:
- Diversity-based selection using distance cutoff
- Adaptive distance decay (gamma annealing)
- Configurable genetic operators
- Support for callbacks (monitoring, persistence)

Reference:
    Lee, J., et al. (2003). Conformational space annealing.
    Physical Review E, 68(2), 021907.
"""

from typing import List, Optional, Callable, Tuple
import logging
import math
import random
import time
import numpy as np
from operator import attrgetter
from datetime import datetime#, time

from rdkit import DataStructs

from .data_models import Candidate, RunContext
from .config import MasterConfig
from ..solutions.base import SolutionSpec

logger = logging.getLogger(__name__)


# =============================================================================
# Worker Function for Parallel Local Optimization
# =============================================================================

class CSAEngine:
    """
    Conformational Space Annealing optimization engine.

    CSA is a global optimization algorithm that maintains a diverse bank
    of solutions and generates offspring through genetic operations.
    Diversity is controlled by a distance cutoff that decays over time.

    Example:
        >>> config = MasterConfig.from_yaml('input.yaml')
        >>> context = RunContext(config=config, rng=rng)
        >>> spec = FragmentRouteSpec(context)
        >>> engine = CSAEngine(spec, context)
        >>> best_candidate = engine.run()
    """

    def __init__(self, spec: SolutionSpec, context: RunContext):
        """
        Initialize CSA engine.

        Args:
            spec: Solution specification (defines representation and operators)
            context: Runtime context (config, RNG, caches)
        """
        self.spec = spec
        self.context = context
        self.config = context.config

        # CSA parameters
        self.bank_size = self.config.csa.bank_size
        self.seed_size = self.config.csa.seed_size
        self.max_iter = self.config.csa.max_iter
        self.d_init = self.config.csa.d_init
        self.d_min = self.config.csa.d_min
        self.gamma = self.config.csa.gamma

        # Operator counts
        self.n_cross_op1 = self.config.csa.n_cross_op1
        self.n_cross_op2 = self.config.csa.n_cross_op2
        self.n_mut_op = self.config.csa.n_mut_op

        # State
        self.bank: List[Candidate] = []
        self.firstbank: List[Candidate] = []  # Preserve initial bank for crossover diversity
        self.iteration = 0
        self.d_cut = 0.0
        self.d_avg = 0.0
        self.d_min_actual = 0.0
        self.d_max_actual = 0.0

        # Statistics
        self.n_evaluations = 0
        self.best_candidate: Optional[Candidate] = None
        self.best_history: List[Tuple[int, float]] = []
        
        # Tracking (like original MolFinder)
        # num_unused_seed mirrors MolFinder's "iuse" metric and drives termination.
        self.num_unused_seed = 0  # Number of bank entries not yet used as seeds (for termination)
        self.tested_candidate_count = 0  # Total candidates tested

        # Get operators from spec
        self.crossover_ops, self.mutation_ops = spec.get_operators()

        logger.info(f"CSA Engine initialized:")
        logger.info(f"  Bank size: {self.bank_size}")
        logger.info(f"  Max iterations: {self.max_iter}")
        logger.info(f"  Crossover operators: {len(self.crossover_ops)}")
        logger.info(f"  Mutation operators: {len(self.mutation_ops)}")

    # ========================================================================
    # Main Algorithm
    # ========================================================================

    def run(self) -> Candidate:
        """
        Run the CSA optimization.

        Returns:
            Best candidate found

        Example:
            >>> engine = CSAEngine(spec, context)
            >>> best = engine.run()
            >>> print(f"Best score: {best.score.value}")
        """
        logger.info("=" * 70)
        logger.info("Starting CSA Optimization")
        logger.info("=" * 70)
        logger.info(f"Start time: {datetime.now()}")

        start_time = time.time()

        # Initialize bank
        logger.info("Generating initial bank...")
        self._generate_initial_bank()
        logger.info(f"Initial bank generated: {len(self.bank)} candidates")
        logger.info(f"Best initial score: {self.bank[0].objective_value:.6f}")

        # Initialize distance cutoff
        logger.debug("Calculating initial bank distances...")
        self._update_bank_distances()
        self.d_cut = self.d_init if self.d_init is not None else self.d_avg / 2.0
        logger.info(f"Initial avg distance: {self.d_avg:.6f} and distance cutoff: {self.d_cut:.6f}")
        

        # Main optimization loop
        logger.info("" + "=" * 70)
        logger.info("Main Optimization Loop")
        logger.info("=" * 70)

        for self.iteration in range(1, self.max_iter + 1):
            iter_start = time.time()

            # Select seeds (with diversity-based selection)
            seeds = self._select_seeds()
            
            # Check termination condition 
            if self.num_unused_seed < 1:
                logger.info(f"Termination: All {self.bank_size} candidates in the bank have been used as seeds.")
                logger.info(f"  - This indicates convergence: no new better solutions are being found to replace used ones.")
                logger.info(f"  - To run longer, increase bank_size or adjust genetic operators/diversity parameters.")
                logger.info(f"Stopped at iteration {self.iteration}")
                break

            # Generate offspring
            offspring = self._generate_offspring(seeds)

            # Local optimization (if enabled)
            if self._should_apply_local_optimization():
                logger.info(f"Applying local optimization to {len(offspring)} offspring")
                offspring = self._apply_local_optimization(offspring)

            # Update bank
            n_accepted = self._update_bank(offspring)

            # Adapt distance cutoff
            self.d_cut = max(self.d_cut * self.gamma, self.d_min)

            # Track best
            if self.bank[0].objective_value < (self.best_candidate.objective_value if self.best_candidate else float('inf')):
                self.best_candidate = self.bank[0]
                self.best_history.append((self.iteration, self.best_candidate.objective_value))
            
            # Early stopping: terminate if no improvement for 30% of total iterations
            configured_patience = self.config.csa.early_stop_patience
            patience = max(1, configured_patience) if configured_patience is not None else max(1, math.ceil(self.max_iter * 0.30))
            iterations_since_best = self.iteration - self.best_history[-1][0]
            if iterations_since_best >= patience:
                logger.info(
                    f"Early stop: no improvement for {iterations_since_best} iterations "
                    f"(patience={patience}, last improvement at iter {self.best_history[-1][0]})"
                )
                break

            # Update context
            self.context.current_iteration = self.iteration

            # Log progress
            iter_time = time.time() - iter_start
            # if self.iteration % 10 == 0 or self.iteration == 1:
            self._log_iteration(n_accepted, iter_time)
            
            # Save bank snapshot and history (like original)
            self._save_bank_snapshot(self.bank, f"bank_{self.iteration}.txt")
            self._save_history()

        # Final summary
        elapsed = time.time() - start_time
        self._log_final_summary(elapsed)
        
        # Save final bank (like original)
        self._save_bank_snapshot(self.bank, "final_bank.txt")

        return self.best_candidate

    # ========================================================================
    # Initialization
    # ========================================================================

    def _generate_initial_bank(self):
        """
        Generate and evaluate initial random bank.

        The bank is sorted by objective value (best first).

        Uses efficient batch generation:
        - Step 1: Generate bank_size genotypes using vectorized numpy sampling
        - Step 2: Build all in one batch (GPU efficient for fragments)
        - Step 3: If not enough valid, generate more in batches until we have enough
        """
        logger.info(f"Generating {self.bank_size} valid random candidates...")

        # Step 1: Generate genotypes using batch numpy sampling (fast, no GPU)
        genotype_candidates = self.spec.random_candidates(int(self.bank_size * 1.25), generation=0)
        logger.info(f"Generated {len(genotype_candidates)} genotypes, now building in batch...")

        # Step 2: Build all in one batch (GPU efficient!)
        built_candidates = self.spec.build_batch(genotype_candidates)
        logger.info(f"Built {len(built_candidates)} candidates in batch...")

        # Step 3: Filter for valid ones
        valid_candidates = [c for c in built_candidates if c.is_valid]

        # Step 4: Generate more if needed (in batches)
        batch_size = max(10, self.bank_size // 10)  # Adaptive batch size
        max_attempts = self.bank_size * 5

        while len(valid_candidates) < self.bank_size and max_attempts > 0:
            needed = self.bank_size - len(valid_candidates)
            to_generate = min(batch_size, needed * 2, max_attempts)  # Generate ~2x what we need

            logger.debug(
                f"Need {needed} more valid candidates, generating {to_generate} more..."
            )

            more_genotypes = self.spec.random_candidates(to_generate, generation=0)
            more_built = self.spec.build_batch(more_genotypes)
            valid_candidates.extend([c for c in more_built if c.is_valid])
            max_attempts -= to_generate

        candidates = valid_candidates[:self.bank_size]

        if len(candidates) < self.bank_size:
            logger.warning(
                f"Could only generate {len(candidates)}/{self.bank_size} valid candidates"
            )

        # Evaluate initial random candidates (before local optimization)
        logger.info(f"Evaluating {len(candidates)} initial random candidates...")
        candidates = self.spec.evaluate_batch(candidates)
        
        # Sort by score to see initial quality
        candidates.sort(key=lambda c: c.objective_value)
        
        # Save bank_0.txt (initial random candidates before local minimization)
        self._save_bank_snapshot(candidates[:self.bank_size], "bank_0.txt")

        logger.info(f"Applying local minimization to {len(candidates)} initial candidates...")
        candidates = self._apply_local_optimization(candidates)

        # All candidates are valid (we pre-filtered), but sort by score after local opt
        logger.info(f"All {len(candidates)} candidates are valid")

        if len(candidates) < self.seed_size:
            logger.warning(f"Only {len(candidates)} valid candidates (need {self.seed_size} for seeds)")

        # Sort by objective (best first)
        candidates.sort(key=lambda c: c.objective_value)

        self.bank = candidates[:self.bank_size]
        
        # Store firstbank for crossover diversity (like original MolFinder)
        self.firstbank = self.bank.copy()
        logger.info(f"Stored firstbank ({len(self.firstbank)} candidates) for crossover diversity")
        
        # Save bank_1.txt (after local minimization) - matches original bank_0.txt format
        self._save_bank_snapshot(self.bank, "bank_1.txt")

        # Track best
        if self.bank:
            self.best_candidate = self.bank[0]
            self.best_history.append((0, self.best_candidate.objective_value))

    # ========================================================================
    # Selection
    # ========================================================================

    def _select_seeds(self) -> List[Candidate]:
        """
        Select seeds for breeding using diversity-based selection.
        
        This implements the original MolFinder seed selection algorithm:
        - First tries to select from unused candidates
        - Uses distance-based diversity to avoid clustering
        - Falls back to used candidates if needed
        
        Returns:
            List of seed candidates
        """
        # Fingerprints are already calculated during evaluation
        # No need to recalculate them
        
        # Separate unused and used candidates
        unused_as_seeds = [c for c in self.bank if not c.metadata.get('is_used', False)]
        used_as_seeds = [c for c in self.bank if c.metadata.get('is_used', False)]
        
        self.num_unused_seed = len(unused_as_seeds)
        logger.info(f"Unused seeds available: {self.num_unused_seed} and used seeds: {len(used_as_seeds)}")
        
        seed_list = []
        
        if self.num_unused_seed >= self.seed_size:
            # Case 1: Enough unused candidates
            # Select diverse seeds from unused pool
            seed_list = self._select_diverse_seeds(
                unused_as_seeds, 
                self.seed_size,
                pool_name="unused"
            )
            
        else:
            # Case 2: Not enough unused candidates
            # Use all unused + fill with diverse used candidates
            seed_list = unused_as_seeds.copy()
            
            if len(seed_list) < self.seed_size and len(used_as_seeds) > 0:
                # Need to add from used pool
                additional_needed = self.seed_size - len(seed_list)
                additional_seeds = self._select_diverse_seeds(
                    used_as_seeds,
                    additional_needed,
                    pool_name="used",
                    initial_seeds=seed_list
                )
                seed_list.extend(additional_seeds)
        
        # Mark all selected seeds as used
        for seed in seed_list:
            seed.metadata['is_used'] = True
        
        logger.info(f"Selected {len(seed_list)} seeds for this iteration")
        
        return seed_list
    
    def _select_diverse_seeds(
        self, 
        pool: List[Candidate], 
        n_select: int,
        pool_name: str = "pool",
        initial_seeds: List[Candidate] = None
    ) -> List[Candidate]:
        """
        Select diverse seeds from a pool using distance-based selection.
        
        Algorithm:
        1. Pick first seed randomly
        2. For each subsequent seed:
           - Calculate distances from current seed to all remaining candidates
           - Find average distance
           - Among candidates with above-average distance, pick the one with best score
           - This ensures diversity while maintaining quality
        
        Args:
            pool: Pool of candidates to select from.
            n_select: Number of seeds to select **from this pool**.
            pool_name: Name for logging purposes.
            initial_seeds: Already selected seeds (used for diversity calculations only).
            
        Returns:
            List of newly selected seeds (does not include initial_seeds).
        """
        if len(pool) == 0 or n_select <= 0:
            return []
        
        # If we need as many or more than are available, return all of them.
        if n_select >= len(pool):
            return pool.copy()
        
        remaining = pool.copy()
        selected = initial_seeds.copy() if initial_seeds else []
        start_count = len(selected)
        target_total = start_count + n_select

        # Select first seed randomly (only among the pool)
        seed = random.choice(remaining)
        selected.append(seed)
        remaining.remove(seed)

        # Select remaining seeds based on diversity until we reach the target total
        while len(selected) < target_total and len(remaining) > 0:
            current_seed = selected[-1]  # Use most recently added seed as reference
            
            # Calculate distances from current seed to all remaining candidates
            # Distance = 1 - Tanimoto similarity (using RDKit fingerprints)
            current_fp = current_seed.fingerprint
            remaining_fps = [cand.fingerprint for cand in remaining]
            sims = DataStructs.BulkTanimotoSimilarity(current_fp, remaining_fps)
            dist_list = [(cand, 1.0 - sim) for cand, sim in zip(remaining, sims)]
            
            if not dist_list:
                break
            
            # Find average distance
            avg_dist = np.mean([d for _, d in dist_list])
            logger.debug(f"Average distance from current seed: {avg_dist:.6f}")
            
            # Get candidates with above-average distance
            above_average = [candidate for candidate, dist in dist_list if dist >= avg_dist]
            
            if not above_average:
                # If none above average, just take the furthest one
                above_average = [max(dist_list, key=lambda x: x[1])[0]]
            
            # Among diverse candidates, select the one with best score
            next_seed = min(above_average, key=lambda c: c.objective_value)
            
            # Find the actual object in remaining (important for object identity)
            next_seed = [c for c in remaining if c == next_seed][0]
            
            selected.append(next_seed)
            remaining.remove(next_seed)
            
            logger.debug(f"Selected seed {len(selected) - start_count}/{n_select} from {pool_name} pool")
        
        # Return only the newly added seeds (exclude the initial seeds passed in)
        return selected[start_count:]

    # ========================================================================
    # Offspring Generation
    # ========================================================================

    def _generate_offspring(self, seeds: List[Candidate]) -> List[Candidate]:
        """
        Generate offspring through genetic operations.

        CRITICAL: Each seed generates multiple offspring!
        - Each seed × n_cross_op1 crossover1 operations
        - Each seed × n_cross_op2 crossover2 operations
        - Each seed × n_mut_op mutation operations

        Note: Crossover operators may return either:
        - A single Candidate (SMILES-based operators)
        - A List[Candidate] with 1-2 children (fragment-based operators)

        For 60 seeds with 5 ops each:
        - Fragment-based (2 children per crossover): ~1500 offspring
        - SMILES-based (1 child per crossover): ~900 offspring

        Args:
            seeds: Parent candidates

        Returns:
            List of offspring candidates
        """
        offspring = []
        crossover1_count = 0
        crossover2_count = 0

        # Crossover type 1: Each seed × random bank member
        # Partner selected from entire bank (excluding seed itself)
        for seed in seeds:
            for _ in range(self.n_cross_op1):
                partner_candidates = [c for c in self.bank if c != seed]
                if partner_candidates:
                    partner = self.context.rng.choice(partner_candidates)
                    result = self._apply_crossover(seed, partner)
                    # Handle both single Candidate and List[Candidate] returns
                    if result:
                        if isinstance(result, list):
                            offspring.extend(result)
                            crossover1_count += len(result)
                        else:
                            offspring.append(result)
                            crossover1_count += 1

        # Crossover type 2: Each seed × random firstbank member
        # Uses firstbank to preserve initial diversity (like original MolFinder)
        for seed in seeds:
            for _ in range(self.n_cross_op2):
                if self.firstbank:
                    partner = self.context.rng.choice(self.firstbank)
                    result = self._apply_crossover(seed, partner)
                    # Handle both single Candidate and List[Candidate] returns
                    if result:
                        if isinstance(result, list):
                            offspring.extend(result)
                            crossover2_count += len(result)
                        else:
                            offspring.append(result)
                            crossover2_count += 1

        # Mutation: Each seed × n_mut_op mutations
        mutation_count = 0
        for seed in seeds:
            for _ in range(self.n_mut_op):
                child = self._apply_mutation(seed)
                if child:
                    offspring.append(child)
                    mutation_count += 1

        # Build phenotypes for all offspring
        if offspring:
            # Use batch building for GPU acceleration
            offspring = self.spec.build_batch(offspring)

        # Filter valid offspring (must have phenotype to check validity)
        valid_offspring = [c for c in offspring if c.is_valid]

        # Evaluate only valid offspring (computes scores only)
        if valid_offspring:
            valid_offspring = self.spec.evaluate_batch(valid_offspring)
            self.n_evaluations += len(valid_offspring)

        # Log offspring generation statistics
        logger.debug(f"Generated {len(offspring)} offspring ({len(valid_offspring)} valid):")
        logger.debug(f"    Crossover1: {crossover1_count} children (seed × bank)")
        logger.debug(f"    Crossover2: {crossover2_count} children (seed × firstbank)")
        logger.debug(f"    Mutation:   {mutation_count} children")

        return valid_offspring

    def _apply_crossover(self, parent1: Candidate, parent2: Candidate):
        """
        Apply a random crossover operator.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Either:
            - A single Candidate (SMILES-based crossover)
            - A List[Candidate] with 1-2 children (fragment-based crossover)
            - None or empty list if crossover failed
        """
        if not self.crossover_ops:
            return None

        op = self.context.rng.choice(self.crossover_ops)
        return op(parent1, parent2)

    def _apply_mutation(self, parent: Candidate) -> Optional[Candidate]:
        """
        Apply a random mutation operator.

        Args:
            parent: Parent candidate

        Returns:
            Mutated child candidate or None if failed
        """
        if not self.mutation_ops:
            return None

        op = self.context.rng.choice(self.mutation_ops)
        return op(parent)

    # ========================================================================
    # Bank Update
    # ========================================================================

    def _update_bank(self, offspring: List[Candidate]) -> int:
        """
        Update bank with offspring using CSA diversity criterion.
        
        Implements the replacement strategy:
        - For each trial, find nearest neighbor in bank
        - If distance ≤ d_cut AND trial better → REPLACE nearest neighbor
        - If distance > d_cut AND trial better than worst → REPLACE worst
        - Bank size stays constant (no append+trim)
        
        This preserves unused seeds better than append+sort+trim!

        Args:
            offspring: List of offspring candidates

        Returns:
            Number of candidates accepted into bank
        """
        if not offspring:
            return 0

        n_accepted = 0

        for trial in offspring:
            # Track total number of candidates tested 
            self.tested_candidate_count += 1
            
            # Find worst bank member (bank is sorted)
            worst_idx = len(self.bank) - 1
            worst = self.bank[worst_idx]
            
            # Discard if worse than worst bank member (keep ties for diversity)
            if trial.objective_value > worst.objective_value:
                continue
            
            # Find nearest neighbor in bank and its distance
            min_dist = float('inf')
            nearest_idx = -1
            trial_fp = trial.fingerprint
            
            for i, bank_member in enumerate(self.bank):
                bank_fp = bank_member.fingerprint
                sim = DataStructs.TanimotoSimilarity(trial_fp, bank_fp)
                dist = 1.0 - sim
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
            
            nearest = self.bank[nearest_idx]
            
            # Decision: Replace based on distance and score
            if min_dist <= self.d_cut:
                # Inside d_cut: Replace if at least as good as nearest neighbor (ties allowed to keep diversity)
                if trial.objective_value <= nearest.objective_value:
                    logger.debug(f"  Replacing nearest neighbor (dist={min_dist:.4f} ≤ {self.d_cut:.4f}) and its objective={nearest.objective_value:.6f} with trial objective={trial.objective_value:.6f}")
                    # Any newly inserted candidate has not been used as a seed yet.
                    trial.metadata['is_used'] = False
                    self.bank[nearest_idx] = trial
                    n_accepted += 1
                    # Re-sort bank to maintain sorted order for next trial
                    self.bank.sort(key=lambda c: c.objective_value)
                else:
                    logger.debug(f"  Rejected: worse than nearest neighbor (dist={min_dist:.4f} ≤ {self.d_cut:.4f}) and its objective={nearest.objective_value:.6f} with trial objective={trial.objective_value:.6f}")
            else:
                # Outside d_cut: Replace worst member (already checked it's >= worst)
                logger.debug(f"  Replacing worst (dist={min_dist:.4f} > {self.d_cut:.4f}) and its objective={worst.objective_value:.6f} with trial objective={trial.objective_value:.6f}")
                trial.metadata['is_used'] = False
                self.bank[worst_idx] = trial
                n_accepted += 1
                # Re-sort bank to maintain sorted order for next trial
                self.bank.sort(key=lambda c: c.objective_value)

        # Update distances (bank composition may have changed)
        if n_accepted > 0:
            self._update_bank_distances()

        return n_accepted

    def _check_diversity(self, candidate: Candidate) -> bool:
        """
        Check if candidate is sufficiently diverse from bank.

        Uses the optimal method based on fingerprint type (RDKit, numpy, or fallback).

        Args:
            candidate: Candidate to check

        Returns:
            True if diverse (all distances > d_cut), False otherwise
        """
        if not self.bank:
            return True

        # Get candidate fingerprint (should always be present)
        cand_fp = candidate.fingerprint
        
        # Get all bank fingerprints (should always be present)
        bank_fps = [c.fingerprint for c in self.bank]

        # Determine fingerprint type and use optimal method
        # All are RDKit ExplicitBitVect
        sims = DataStructs.BulkTanimotoSimilarity(cand_fp, bank_fps)
        for sim in sims:
            dist = 1.0 - sim
            if dist <= self.d_cut:
                return False
        return True
        

    # ========================================================================
    # Distance Calculation
    # ========================================================================

    def _update_bank_distances(self):
        """
        Update pairwise distance statistics for the bank.

        Computes:
        - Average pairwise distance
        - Min/max pairwise distance
        
        Optimized: Uses RDKit's BulkTanimotoSimilarity for fast computation.
        """
        if len(self.bank) < 2:
            self.d_avg = 0.0
            self.d_min_actual = 0.0
            self.d_max_actual = 0.0
            return

        # Compute all pairwise distances using RDKit's BulkTanimotoSimilarity
        distances = self._compute_pairwise_distances()

        if distances:
            self.d_avg = np.mean(distances)
            self.d_min_actual = np.min(distances)
            self.d_max_actual = np.max(distances)
        else:
            self.d_avg = 0.0
            self.d_min_actual = 0.0
            self.d_max_actual = 0.0

    def _compute_pairwise_distances(self) -> List[float]:
        """
        Compute pairwise distances using RDKit's BulkTanimotoSimilarity.
        
        This is the fastest method, using RDKit's optimized C++ implementation.
        Works with RDKit ExplicitBitVect fingerprints.
        
        Returns:
            List of pairwise distances
        """
        n = len(self.bank)
        distances = []
        
        # Get all fingerprints
        bank_fps = [c.fingerprint for c in self.bank]
        
        # Compute pairwise distances using BulkTanimotoSimilarity
        # For each fingerprint, compute similarity to all subsequent fingerprints
        for i in range(n):
            # Compute similarities to all fingerprints after i
            remaining_fps = bank_fps[i + 1:]
            if remaining_fps:
                sims = DataStructs.BulkTanimotoSimilarity(bank_fps[i], remaining_fps)
                # Convert similarities to distances (1 - similarity)
                distances.extend([1.0 - sim for sim in sims])
                
        return distances

    # ========================================================================
    # Logging and Monitoring
    # ========================================================================

    def _log_iteration(self, n_accepted: int, iter_time: float):
        """
        Log iteration progress.

        Args:
            n_accepted: Number of offspring accepted
            iter_time: Iteration time in seconds
        """
        best_score = self.bank[0].objective_value if self.bank else float('inf')
        worst_score = self.bank[-1].objective_value if self.bank else float('inf')

        logger.info(
            f"Iter {self.iteration:3d}: "
            f"Best={best_score:.6f} | "
            f"Worst={worst_score:.6f} | "
            f"d_cut={self.d_cut:.4f} | "
            f"d_avg={self.d_avg:.4f} | "
            f"Accepted={n_accepted} | "
            f"Time={iter_time:.2f}s"
        )

    def _log_final_summary(self, elapsed_time: float):
        """
        Log final optimization summary.

        Args:
            elapsed_time: Total elapsed time in seconds
        """
        logger.info("" + "=" * 70)
        logger.info("CSA Optimization Complete")
        logger.info("=" * 70)
        logger.info(f"Total time: {elapsed_time:.2f}s")
        logger.info(f"Total evaluations: {self.n_evaluations}")
        logger.info(f"Evaluations/second: {self.n_evaluations/elapsed_time:.2f}")
        logger.info(f"Best candidate found:")
        logger.info(f"  Score: {self.best_candidate.objective_value:.6f}")
        logger.info(f"  SMILES: {self.best_candidate.smiles}")
        logger.info(f"  Found at iteration: {self.best_history[-1][0]}")
        logger.info(f"Improvement history: {len(self.best_history)} improvements")
        for iter_num, score in self.best_history[:5]:
            logger.info(f"  Iter {iter_num:3d}: {score:.6f}")
        if len(self.best_history) > 5:
            logger.info(f"  ... ({len(self.best_history) - 5} more)")

    # ========================================================================
    # Public Getters
    # ========================================================================

    def get_bank(self) -> List[Candidate]:
        """
        Get current bank of candidates.

        Returns:
            List of candidates (sorted by score, best first)
        """
        return self.bank.copy()

    def get_best(self) -> Optional[Candidate]:
        """
        Get best candidate found so far.

        Returns:
            Best candidate or None if no valid candidates
        """
        return self.best_candidate

    def get_statistics(self) -> dict:
        """
        Get optimization statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'iteration': self.iteration,
            'n_evaluations': self.n_evaluations,
            'best_score': self.best_candidate.objective_value if self.best_candidate else None,
            'd_cut': self.d_cut,
            'd_avg': self.d_avg,
            'd_min': self.d_min_actual,
            'd_max': self.d_max_actual,
            'bank_size': len(self.bank),
            'n_improvements': len(self.best_history),
        }

    # ========================================================================
    # Local Optimization
    # ========================================================================

    def _should_apply_local_optimization(self) -> bool:
        """
        Determine if local optimization should be applied in this iteration.

        Returns:
            True if local optimization should be applied
        """
        if not self.config.csa.local_optimization_enabled:
            return False

        # Apply every N iterations based on frequency
        if self.iteration % self.config.csa.local_optimization_frequency == 0:
            return True

        return False

    def _apply_local_optimization(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Apply local optimization to trial candidates (called during main loop).

        Returns:
            Optimized candidates
        """
        # logger.info(f"Applying local optimization to {len(candidates)} solutions")

        if not candidates:
            return candidates

        start_time = time.time()
        # optimized_candidates, total_evals = self._local_optimize_internal(candidates)
        max_steps = self.config.csa.local_optimization_max_steps
        neighbors_per_candidate = self.config.csa.local_optimization_neighbors

        optimized_candidates, total_evals = self.spec.local_optimize_batch(
            candidates, max_steps=max_steps, neighbors_per_candidate=neighbors_per_candidate
        )
        elapsed = time.time() - start_time

        self.n_evaluations += total_evals
        logger.info(f"Local optimization complete: {total_evals} additional evaluations in {elapsed:.2f}s")

        return optimized_candidates

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _save_bank_snapshot(self, candidates: List[Candidate], filename: str):
        """
        Save a snapshot of candidates to a text file.
        
        Updated to include Genotype and Intermediates.

        Args:
            candidates: List of candidates to save
            filename: Output filename (e.g., 'bank_0.txt')
        """
        from pathlib import Path
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            # Add header
            f.write("Score\tSMILES\tGenotype\tIntermediates\n")
            for candidate in candidates:
                score = candidate.objective_value
                smiles = candidate.smiles if candidate.smiles else "INVALID"
                
                # Format genotype
                genotype = str(candidate.genotype)
                
                # Format intermediates
                intermediates = str(getattr(candidate, 'intermediates', []))
                
                f.write(f"{score:.6f}\t{smiles}\t{genotype}\t{intermediates}\n")
        
        logger.info(f"Saved {len(candidates)} candidates to {filepath}")
    
    def _save_history(self):
        """
        Save iteration history to history file.
        
        Matches original MolFinder format with 9 columns:
        icycle, d_cut, d_avg, best, median, worst, unused_seeds, no_tested, nft, timestamp
        
        where:
        - icycle: iteration number
        - d_cut: distance cutoff
        - d_avg: average distance in bank
        - best/median/worst: score statistics
        - unused_seeds: number of unused seeds (MolFinder's "iuse")
        - no_tested: total candidates tested so far (history column kept for compatibility)
        - nft: number of function evaluations (same as no_tested in our case)
        """
        from pathlib import Path
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / "history"
        
        # Append to history file (create with header if first iteration)
        mode = 'a' if self.iteration > 1 else 'w'
        with open(filepath, mode) as f:
            if self.iteration == 1:
                # Write header (matching original format with fixed column widths)
                f.write(
                    f"{'#':>1} "
                    f"{'Iteration':>9} "
                    f"{'D_cut':>10} "
                    f"{'D_avg':>10} "
                    f"{'Best':>12} "
                    f"{'Median':>12} "
                    f"{'Worst':>12} "
                    f"{'Unused':>6} "
                    f"{'No_tested':>12} "
                    f"{'NFT':>12} "
                    f"{'Timestamp':>26}\n"
                )
            
            # Write iteration data
            best_score = self.bank[0].objective_value if self.bank else float('inf')
            median_idx = len(self.bank) // 2
            median_score = self.bank[median_idx].objective_value if self.bank else float('inf')
            worst_score = self.bank[-1].objective_value if self.bank else float('inf')
            
            # Format with fixed column widths matching header
            f.write(
                f"  "  # 2 spaces for alignment (replaces '#')
                f"{self.iteration:9d} "
                f"{self.d_cut:10.6f} "
                f"{self.d_avg:10.6f} "
                f"{best_score:12.6f} "
                f"{median_score:12.6f} "
                f"{worst_score:12.6f} "
                f"{self.num_unused_seed:6d} "
                f"{self.tested_candidate_count:12d} "
                f"{self.n_evaluations:12d} "
                f"{datetime.now()}\n"
            )
        
        logger.debug(f"Updated history file: {filepath}")


    # ========================================================================
    # Callbacks (for monitoring and persistence)
    # ========================================================================

    def register_callback(self, event: str, callback: Callable):
        """
        Register a callback for an event.

        Supported events:
        - 'iteration_end': Called at end of each iteration
        - 'bank_update': Called when bank is updated
        - 'improvement': Called when best candidate improves

        Args:
            event: Event name
            callback: Callback function

        Note:
            Callbacks will be added in Phase 4 (Monitoring)
        """
        # TODO: Implement callback system in Phase 4
        pass
