"""Constrained core-hopping spec: fix two warheads, vary only the central core.

True scaffold hopping = keep a molecule's peripheral substituents and replace ONLY
its central scaffold. RxnMol assembles molecules by sequential forward reaction, so we
can express this as a route with a FIXED first and last fragment (the two warheads, given
as reactive precursors) and a single VARIABLE middle fragment (the core):

    genotype = [warhead_A, core_i, warhead_B]
    build:  warhead_A --react--> (warhead_A + core_i) --react--> warhead_A-core_i-warhead_B

For suvorexant (the validated case) warhead_A is the benzoic-acid form and warhead_B the
2-halo-benzoxazole; cores are bis-secondary-amine scaffolds. Step 1 is an amide coupling
(acid + one ring N), step 2 an SNAr / N-arylation (aryl halide + the remaining ring N).
The two warhead chemistries are orthogonal, so each warhead lands on a distinct ring N and
the assembly is unambiguous -- empirically clean for diamine cores.

Only the core slot is mutated; warheads are never crossed-over/added/removed. A substructure
gate marks any assembled product missing either warhead as invalid (the engine then discards
it), so the reaction model's occasional misfires never reach the objective.

Config (configs.yaml):
    solution:
      spec_type: scaffold_hop_route
      warhead_a: "OC(=O)c1cc(C)ccc1-n1nccn1"     # reactive precursor (acid)
      warhead_b: "Clc1nc2cc(Cl)ccc2o1"           # reactive precursor (2-halo-benzoxazole)
    data:
      building_blocks_file: cores.smi             # the CORE library (bis-secondary-amines)
"""
import logging
from typing import List, Optional, Tuple, Callable

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..core.data_models import Candidate
from .fragments_route import FragmentRouteSpec

logger = logging.getLogger(__name__)


class ScaffoldHopRouteSpec(FragmentRouteSpec):
    """[warhead_A, core, warhead_B] with fixed warheads; CSA optimizes only the core."""

    def __init__(self, context):
        super().__init__(context)
        sc = self.config.solution
        self.warhead_a = getattr(sc, "warhead_a", None)
        self.warhead_b = getattr(sc, "warhead_b", None)
        if not self.warhead_a or not self.warhead_b:
            raise ValueError(
                "scaffold_hop_route requires solution.warhead_a and solution.warhead_b "
                "(reactive precursor SMILES for the two fixed substituents)."
            )
        for tag, smi in (("warhead_a", self.warhead_a), ("warhead_b", self.warhead_b)):
            if Chem.MolFromSmiles(smi) is None:
                raise ValueError(f"Invalid {tag} SMILES: {smi!r}")

        # Substructure gate: require each warhead's ring system (Bemis-Murcko of the
        # precursor, which drops the leaving group) to survive into the assembled product.
        self._gate_a = self._gate_pattern(self.warhead_a)
        self._gate_b = self._gate_pattern(self.warhead_b)

        # `self.building_blocks` (loaded by the parent from data.building_blocks_file) is the
        # CORE library here. Restrict operators to the core slot only.
        self.crossover_ops = []                       # single variable slot -> crossover is a no-op
        self.mutation_ops = [self._mutate_core]
        logger.info(
            "ScaffoldHopRouteSpec ready: %d core building blocks; warheads fixed (A=%s, B=%s)",
            len(self.building_blocks), self.warhead_a, self.warhead_b,
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _gate_pattern(smi: str) -> Chem.Mol:
        """Ring system the warhead must retain in the product (Murcko of the precursor)."""
        m = Chem.MolFromSmiles(smi)
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        if scaf is not None and scaf.GetNumAtoms() > 0:
            return scaf
        return m  # acyclic warhead: fall back to the whole precursor

    def get_genotype_type(self) -> str:
        return "fragment_route"  # inherit the parent's fragment_route build/cache machinery

    def get_operators(self) -> Tuple[List[Callable], List[Callable]]:
        return (self.crossover_ops, self.mutation_ops)

    # --------------------------------------------------------------- generation
    def random_candidates(self, n: int, generation: int = 0) -> List[Candidate]:
        """Sample n routes [warhead_A, core_i, warhead_B] with random cores."""
        idx = self.rng.integers(0, len(self.building_blocks), size=n)
        cores = self.building_blocks[idx]
        return [
            Candidate(
                genotype=[self.warhead_a, str(c), self.warhead_b],
                genotype_type="fragment_route",
                metadata={"generation": generation, "is_seed": True, "core": str(c)},
            )
            for c in cores
        ]

    def _mutate_core(self, parent: Candidate) -> Optional[Candidate]:
        """Replace the central core with a different one; warheads untouched."""
        frags = list(parent.genotype)
        if len(frags) != 3:
            return None
        new_core = str(self.rng.choice(self.building_blocks))
        if new_core == frags[1]:
            return None
        frags = [self.warhead_a, new_core, self.warhead_b]
        return Candidate(
            genotype=frags,
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
                if mol is None or not (
                    mol.HasSubstructMatch(self._gate_a) and mol.HasSubstructMatch(self._gate_b)
                ):
                    cand.is_valid = False
                    cand.smiles = None
                    cand.phenotype_mol = None
                    cand.metadata["failure_reason"] = "warhead_gate (missing a warhead)"
        return built
