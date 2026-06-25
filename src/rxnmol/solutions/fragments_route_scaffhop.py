"""Constrained core-hopping spec: fix the warheads, vary only the central core.

True scaffold hopping = keep a molecule's peripheral substituents and replace ONLY its central
scaffold. RxnMol assembles molecules by sequential forward reaction, so core replacement is a
route with FIXED warhead fragments (reactive precursors) and a single VARIABLE core fragment.

Two modes:

  (A) Two-warhead, core in the middle -- `warhead_a` / `warhead_b`:
        genotype = [warhead_A, core_i, warhead_B]
        e.g. suvorexant: acid + diamine core -> amide; + 2-halo-benzoxazole -> SNAr.
        Orthogonal chemistries place each warhead on a distinct ring N (clean for diamine cores).

  (B) N-warhead, core first -- `warheads: [w1, w2, ...]`:
        genotype = [core_i, w1, w2, ...]   (core reacts with w1, then w2, ...)
        Use when the scaffold carries 3+ substituents: list the fixed reactive precursors in
        `warheads` and the core reacts with each in turn. Note: warheads attaching to similar
        groups (e.g. two amines) are NOT regio-controlled -- products may be regioisomers.

Only the core slot is mutated; warheads are never crossed-over/added/removed. A substructure gate
marks any assembled product missing a warhead as invalid (the engine discards it). Gate patterns
come from `warhead_patterns` (SMARTS) if given, else the Bemis-Murcko scaffold of each precursor.

Config (configs.yaml):
    solution:
      spec_type: scaffold_hop_route
      # mode A (suvorexant example):
      warhead_a: "OC(=O)c1cc(C)ccc1-n1nccn1"
      warhead_b: "Clc1nc2cc(Cl)ccc2o1"
      # OR mode B (3+ warheads):
      warheads: ["<precursor_1>", "<precursor_2>", "<precursor_3>"]
      warhead_patterns: ["<smarts_1>", "<smarts_2>", "<smarts_3>"]   # gate (recommended for mode B)
    data:
      building_blocks_file: cores.smi
"""
import logging
from typing import List, Optional, Tuple, Callable

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..core.data_models import Candidate
from .fragments_route import FragmentRouteSpec

logger = logging.getLogger(__name__)


class ScaffoldHopRouteSpec(FragmentRouteSpec):
    """Fixed warheads, variable core; CSA/enumeration optimizes only the core."""

    def __init__(self, context):
        super().__init__(context)
        sc = self.config.solution
        warheads = getattr(sc, "warheads", None)
        warhead_a = getattr(sc, "warhead_a", None)
        warhead_b = getattr(sc, "warhead_b", None)
        patterns = getattr(sc, "warhead_patterns", None)

        if warheads:  # mode B: core-first, N warheads
            self._warheads = list(warheads)
            self._core_index = 0
            self._template = lambda core: [core] + self._warheads
        elif warhead_a and warhead_b:  # mode A: warhead, core, warhead
            self._warheads = [warhead_a, warhead_b]
            self._core_index = 1
            self._template = lambda core: [warhead_a, core, warhead_b]
        else:
            raise ValueError(
                "scaffold_hop_route requires either solution.warheads (list) "
                "or both solution.warhead_a and solution.warhead_b."
            )
        for w in self._warheads:
            if Chem.MolFromSmiles(w) is None:
                raise ValueError(f"Invalid warhead SMILES: {w!r}")

        # Substructure gate: each warhead must survive into the assembled product.
        if patterns:
            if len(patterns) != len(self._warheads):
                raise ValueError("warhead_patterns must have one SMARTS per warhead.")
            self._gates = [Chem.MolFromSmarts(p) for p in patterns]
            if any(g is None for g in self._gates):
                raise ValueError("Invalid SMARTS in warhead_patterns.")
        else:
            self._gates = [self._murcko_pattern(w) for w in self._warheads]

        self._tlen = len(self._warheads) + 1
        self.crossover_ops = []                # single variable slot -> crossover is a no-op
        self.mutation_ops = [self._mutate_core]
        logger.info(
            "ScaffoldHopRouteSpec ready: %d cores; %d fixed warheads (core_index=%d): %s",
            len(self.building_blocks), len(self._warheads), self._core_index, self._warheads,
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _murcko_pattern(smi: str) -> Chem.Mol:
        """Ring system the warhead must retain (Murcko of the precursor); whole mol if acyclic."""
        m = Chem.MolFromSmiles(smi)
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        if scaf is not None and scaf.GetNumAtoms() > 0:
            return scaf
        return m

    def get_genotype_type(self) -> str:
        return "fragment_route"

    def get_operators(self) -> Tuple[List[Callable], List[Callable]]:
        return (self.crossover_ops, self.mutation_ops)

    # --------------------------------------------------------------- generation
    def random_candidates(self, n: int, generation: int = 0) -> List[Candidate]:
        idx = self.rng.integers(0, len(self.building_blocks), size=n)
        cores = self.building_blocks[idx]
        return [
            Candidate(
                genotype=self._template(str(c)),
                genotype_type="fragment_route",
                metadata={"generation": generation, "is_seed": True, "core": str(c)},
            )
            for c in cores
        ]

    def _mutate_core(self, parent: Candidate) -> Optional[Candidate]:
        frags = list(parent.genotype)
        if len(frags) != self._tlen:
            return None
        new_core = str(self.rng.choice(self.building_blocks))
        if new_core == frags[self._core_index]:
            return None
        return Candidate(
            genotype=self._template(new_core),
            genotype_type="fragment_route",
            metadata={
                "generation": parent.metadata.get("generation", 0) + 1,
                "parents": [parent.smiles],
                "operator": "mutate_core",
                "core": new_core,
                "is_seed": False,
            },
        )

    # ------------------------------------------------------------- warhead gate
    def build_batch(self, candidates: List[Candidate]) -> List[Candidate]:
        built = super().build_batch(candidates)
        for cand in built:
            if getattr(cand, "is_valid", False) and cand.smiles:
                mol = cand.phenotype_mol or Chem.MolFromSmiles(cand.smiles)
                if mol is None or not all(mol.HasSubstructMatch(g) for g in self._gates):
                    cand.is_valid = False
                    cand.smiles = None
                    cand.phenotype_mol = None
                    cand.metadata["failure_reason"] = "warhead_gate (missing a warhead)"
        return built
