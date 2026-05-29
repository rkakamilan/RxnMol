"""
Reaction-based molecular solution specification.

This module implements the ReactionMolSpec, where solutions are represented
as SMILES strings that evolve through reaction predictions.
"""

from typing import List, Tuple, Callable, Optional, Dict, Any
import logging
from pathlib import Path
import copy

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from .base import SolutionSpec, tanimoto_distance
from ..core.data_models import Candidate, RunContext
from .cache import PersistentReactionCache, compute_reaction_model_cache_id
from .reaction_models import load_reaction_model
from ..utils.building_blocks import load_building_blocks

# Disable RDKit warnings
RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


class ReactionMolSpec(SolutionSpec):
    """
    Solution specification for reaction-based molecular evolution.

    Genotype: SMILES string (final product)
    Evolution:
        - Crossover: React(Parent1, Parent2)
        - Mutation: React(Parent, RandomFragment)
    """

    def __init__(self, context: RunContext):
        """
        Initialize reaction mol specification.

        Args:
            context: Shared runtime context
        """
        super().__init__(context)

        # Load fragment library
        self.building_blocks_file = self.config.data.building_blocks_file
        self.building_blocks = self._load_building_blocks(self.building_blocks_file)

        # Persistent cache
        model_cache_id = compute_reaction_model_cache_id(self.config.reaction_model)
        run_name = Path(self.config.output_dir).name if self.config.output_dir else None
        shared_path = self.config.data.cache_path or "./cache/reaction_cache_shared.db"
        self.reaction_cache = PersistentReactionCache(
            shared_db_path=str(shared_path),
            local_db_dir="./cache/local",
            run_name=run_name,
            model_id=model_cache_id,
            auto_merge_on_close=self.config.persistence.cache_auto_merge,
            auto_merge_delete=self.config.persistence.cache_auto_merge_delete,
            cache_max_entries=self.config.persistence.cache_max_entries,
        )

        # Load reaction predictor (shared via context)
        self.rxn_predictor = context.reaction_model
        if self.rxn_predictor is None:
            logger.info("Loading reaction prediction model...")
            self.rxn_predictor = load_reaction_model(
                self.config.reaction_model,
                device=self.config.runtime.device,
            )
            self.context.reaction_model = self.rxn_predictor
        logger.info("Reaction prediction model ready")
        
        # Genetic operators
        self.crossover_ops = [self._crossover_react]
        self.mutation_ops = [self._mutate_react]

        # Create objective instance ONCE
        self.objective = self._create_objective()

    def _create_objective(self):
        from ..objectives import get_objective
        return get_objective(
            name=self.context.config.objective.name,
            validity_penalty=self.context.config.objective.validity_penalty
        )

    # ========================================================================
    # SolutionSpec Abstract Methods
    # ========================================================================

    def random_candidate(self, generation: int = 0) -> Candidate:
        """Generate a random candidate (single fragment)."""
        smiles = self.rng.choice(self.building_blocks)
        
        candidate = Candidate(
            genotype=smiles,
            genotype_type='reaction_mol',
            intermediates=[f"Initial: {smiles}"],
            metadata={
                'generation': generation,
                'is_seed': True,
                'step': 1,
                'reactants': []
            }
        )
        
        # For random candidates, genotype IS the phenotype (no reaction needed yet)
        # But we still need to validate/neutralize it
        return candidate

    def build(self, candidate: Candidate) -> Candidate:
        """Build phenotype for a single candidate (wrapper for batch)."""
        return self.build_batch([candidate])[0]

    def build_batch(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Build phenotypes for a batch of candidates.
        
        Handles two cases:
        1. "PENDING" candidates: Need reaction prediction (from operators)
        2. Existing SMILES candidates: Just need RDKit Mol creation (from random init)
        """
        if not candidates:
            return []

        # Separate pending and ready candidates
        pending_indices = []
        pending_inputs = [] # "r1.r2"
        
        for i, cand in enumerate(candidates):
            if cand.genotype == "PENDING":
                reactants = cand.metadata.get('reactants', [])
                if len(reactants) == 2:
                    pending_indices.append(i)
                    pending_inputs.append(f"{reactants[0]}.{reactants[1]}")
                else:
                    # Invalid pending state
                    cand.is_valid = False
                    cand.metadata['failure_reason'] = "Invalid reactants for pending candidate"

        # Process pending reactions
        if pending_inputs:
            logger.debug(f"Predicting {len(pending_inputs)} reactions for batch...")
            
            # Check cache first
            products = self.reaction_cache.get_batch(pending_inputs)
            
            # Identify missing
            missing_indices = [i for i, p in enumerate(products) if p is None]
            missing_inputs_str = [pending_inputs[i] for i in missing_indices]
            
            # Predict missing
            if missing_inputs_str:
                try:
                    predicted = self.rxn_predictor.predict_batch(missing_inputs_str)
                    self.reaction_cache.set_batch(missing_inputs_str, predicted)
                    
                    for idx, pred in zip(missing_indices, predicted):
                        products[idx] = pred
                except Exception as e:
                    logger.error(f"Batch prediction failed: {e}")
                    # Mark failed
                    for idx in missing_indices:
                        products[idx] = None

            # Update pending candidates
            for idx, product in zip(pending_indices, products):
                cand = candidates[idx]
                if product:
                    cand.genotype = product # The product IS the genotype
                    
                    # Update intermediates with reaction history
                    parent_intermediates = cand.metadata.get('parent_intermediates', [])
                    reactants = cand.metadata.get('reactants', [])
                    op_type = cand.metadata.get('operator', 'unknown')
                    
                    reaction_str = f"{op_type}: {'.'.join(reactants)} >> {product}"
                    cand.intermediates = parent_intermediates + [reaction_str]
                    
                else:
                    cand.is_valid = False
                    cand.metadata['failure_reason'] = "Reaction prediction failed"

        # Finalize all candidates (create Mol objects)
        for cand in candidates:
            if not cand.is_valid and cand.metadata.get('failure_reason'):
                continue
                
            if cand.genotype == "PENDING":
                # Should have been updated above
                continue
                
            # Validate and create Mol
            final_smi = self._validate_and_clean_product(cand.genotype)
            
            if final_smi:
                mol = Chem.MolFromSmiles(final_smi)
                if mol:
                    cand.smiles = final_smi
                    cand.phenotype_mol = mol
                    cand.is_valid = True
                    cand.fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=True)
                else:
                    cand.is_valid = False
                    cand.metadata['failure_reason'] = "Invalid SMILES structure"
            else:
                cand.is_valid = False
                cand.metadata['failure_reason'] = "Validation/Cleaning failed"

        return candidates

    def compute_fingerprint(self, candidate: Candidate) -> Optional[Any]:
        """Return fingerprint."""
        return candidate.fingerprint

    def distance(self, cand1: Candidate, cand2: Candidate) -> float:
        """Compute distance."""
        if cand1.fingerprint is None or cand2.fingerprint is None:
            return 1.0
        return tanimoto_distance(cand1.fingerprint, cand2.fingerprint)

    def get_operators(self) -> Tuple[List[Callable], List[Callable]]:
        # Expanded operator set for better exploration
        return (
            [self._crossover_react, self._crossover_precursors], 
            [self._mutate_react, self._mutate_backtrack]
        )

    def get_genotype_type(self) -> str:
        return "reaction_mol"

    # ========================================================================
    # Genetic Operators
    # ========================================================================

    def _crossover_react(self, parent1: Candidate, parent2: Candidate) -> Optional[Candidate]:
        """
        Crossover: React two parents together.
        New step count = max(p1.step, p2.step) + 1
        """
        step1 = parent1.metadata.get('step', 1)
        step2 = parent2.metadata.get('step', 1)
        new_step = max(step1, step2) + 1
        
        if new_step > self.config.solution.num_step:
            return None
            
        return Candidate(
            genotype="PENDING",
            genotype_type='reaction_mol',
            metadata={
                'generation': max(parent1.metadata.get('generation', 0), parent2.metadata.get('generation', 0)) + 1,
                'parents': [parent1.smiles, parent2.smiles],
                'operator': 'crossover_react',
                'is_seed': False,
                'step': new_step,
                'reactants': [parent1.smiles, parent2.smiles],
                # Combine histories? For now, just track the merge event to avoid explosion
                'parent_intermediates': [f"Merge({len(parent1.intermediates)} steps, {len(parent2.intermediates)} steps)"] 
            }
        )

    def _crossover_precursors(self, parent1: Candidate, parent2: Candidate) -> Optional[Candidate]:
        """
        Crossover: React a precursor of Parent1 with a precursor of Parent2.
        This mixes the building blocks/intermediates of successful candidates.
        """
        # Get reactants of last step
        reactants1 = parent1.metadata.get('reactants', [])
        reactants2 = parent2.metadata.get('reactants', [])
        
        if not reactants1 or not reactants2:
            return None
            
        # Pick one reactant from each
        r1 = self.rng.choice(reactants1)
        r2 = self.rng.choice(reactants2)
        
        # Estimate new step count (approximate)
        step1 = parent1.metadata.get('step', 1)
        step2 = parent2.metadata.get('step', 1)
        # Since we go back one step for each, the new step is roughly max(step1-1, step2-1) + 1
        new_step = max(step1, step2) 
        
        if new_step > self.config.solution.num_step:
            return None

        return Candidate(
            genotype="PENDING",
            genotype_type='reaction_mol',
            metadata={
                'generation': max(parent1.metadata.get('generation', 0), parent2.metadata.get('generation', 0)) + 1,
                'parents': [parent1.smiles, parent2.smiles],
                'operator': 'crossover_precursors',
                'is_seed': False,
                'step': new_step,
                'reactants': [r1, r2],
                'parent_intermediates': [f"PrecursorMerge({len(parent1.intermediates)} steps, {len(parent2.intermediates)} steps)"]
            }
        )

    def _mutate_react(self, parent: Candidate) -> Optional[Candidate]:
        """
        Mutation: React parent with a random fragment.
        New step count = parent.step + 1
        """
        step = parent.metadata.get('step', 1)
        new_step = step + 1
        
        if new_step > self.config.solution.num_step:
            return None
            
        random_frag = self.rng.choice(self.building_blocks)
        
        return Candidate(
            genotype="PENDING",
            genotype_type='reaction_mol',
            metadata={
                'generation': parent.metadata.get('generation', 0) + 1,
                'parents': [parent.smiles],
                'operator': 'mutate_react',
                'is_seed': False,
                'step': new_step,
                'reactants': [parent.smiles, random_frag],
                'parent_intermediates': parent.intermediates
            }
        )

    def _mutate_backtrack(self, parent: Candidate) -> Optional[Candidate]:
        """
        Mutation: Replace one reactant of the last step with a random fragment.
        This allows exploring "sibling" molecules without increasing depth.
        """
        reactants = parent.metadata.get('reactants', [])
        if not reactants:
            # If no reactants (step 1), just pick a new random candidate
            return self.random_candidate(generation=parent.metadata.get('generation', 0) + 1)
            
        # Pick one reactant to keep
        keep_reactant = self.rng.choice(reactants)
        
        # Pick a new random fragment
        new_fragment = self.rng.choice(self.building_blocks)
        
        # Step count stays the same
        step = parent.metadata.get('step', 1)
        
        return Candidate(
            genotype="PENDING",
            genotype_type='reaction_mol',
            metadata={
                'generation': parent.metadata.get('generation', 0) + 1,
                'parents': [parent.smiles],
                'operator': 'mutate_backtrack',
                'is_seed': False,
                'step': step,
                'reactants': [keep_reactant, new_fragment],
                # We lose the last step of history, but keep the rest?
                # Actually, we don't know which part of history corresponds to 'keep_reactant'.
                # For simplicity, we just take the parent's history minus the last step.
                'parent_intermediates': parent.intermediates[:-1] if parent.intermediates else []
            }
        )

    def local_optimize(self, candidate: Candidate, max_steps: int = 10) -> Tuple[Candidate, int]:
        """
        Local optimization (placeholder).
        For now, we just return the candidate as is.
        """
        return candidate, 0

    # ========================================================================
    # Helpers
    # ========================================================================

    def _load_building_blocks(self, building_blocks_file: str) -> List[str]:
        return load_building_blocks(building_blocks_file)

    def _validate_and_clean_product(self, product_smiles: str) -> Optional[str]:
        if not product_smiles or product_smiles == "FAILED":
            return None

        try:
            mol = Chem.MolFromSmiles(product_smiles)
            if mol is None:
                return None

            if '.' in product_smiles:
                products = product_smiles.split('.')
                mols = [Chem.MolFromSmiles(p) for p in products]
                mols = [m for m in mols if m is not None]
                if not mols:
                    return None
                mol = max(mols, key=lambda m: m.GetNumAtoms())

            mol = self._neutralize_atoms(mol)
            
            if self.config.solution.max_mol_size is not None:
                if mol.GetNumAtoms() > self.config.solution.max_mol_size:
                    return None

            return Chem.MolToSmiles(mol)

        except Exception:
            return None

    @staticmethod
    def _neutralize_atoms(mol):
        pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]

        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
        return mol
