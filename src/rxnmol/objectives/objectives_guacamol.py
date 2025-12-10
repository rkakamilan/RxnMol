"""
GuacaMol benchmark objective functions.

This module wraps all GuacaMol goal-directed benchmarks as objectives.
Each objective is registered and can be used via get_objective("zaleplon", config).

GuacaMol benchmarks test molecular optimization algorithms on realistic drug discovery tasks.

Reference:
    Brown et al. (2019). GuacaMol: Benchmarking Models for de Novo Molecular Design.
    Journal of Chemical Information and Modeling, 59(3), 1096-1108.
"""

import logging

from .base import SmilesObjective

logger = logging.getLogger(__name__)

# Import guacamol scoring functions at module level
# import guacamol_scores
# HAS_GUACAMOL = True
try:
    from . import guacamol_scores
    HAS_GUACAMOL = True
except ImportError:
    logging.warning("GuacaMol scores module not found. GuacaMol objectives will not be available.")
    HAS_GUACAMOL = False
    guacamol_scores = None


# =============================================================================
# GuacaMol Objectives
# =============================================================================

if HAS_GUACAMOL:

    class ZaleplonObjective(SmilesObjective):
        """
        Zaleplon MPO (Multi-Parameter Optimization).

        Objective: Find molecules similar to zaleplon but with a different isomeric formula.

        Target: Zaleplon (sedative-hypnotic drug)
        SMILES: O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1
        Formula constraint: C19H17N3O2
        """

        def __init__(self, name='zaleplon', validity_penalty=999999.0, config=None):
            super().__init__(name=name, validity_penalty=validity_penalty, config=config)
            # Initialize scorer once (no lazy loading!)
            self.scorer = guacamol_scores.zaleplon_obj()

        def compute_from_smiles(self, smiles: str) -> float:
            # Maximize -> Return negative
            return -self.scorer.score(smiles)


    class OsimertinibObjective(SmilesObjective):
        """
        Osimertinib MPO.

        Objective: Find molecules similar to osimertinib with:
        - FCFP4 Tanimoto similarity ≤ 0.8
        - ECFP6 Tanimoto similarity > 0.85
        - TPSA > 100
        - logP < 1

        Target: Osimertinib (EGFR inhibitor for lung cancer)
        """

        def __init__(self, name='osimertinib', validity_penalty=999999.0, config=None):
            super().__init__(name=name, validity_penalty=validity_penalty, config=config)
            self.scorer = guacamol_scores.osimertinib_obj()

        def compute_from_smiles(self, smiles: str) -> float:
            # Maximize -> Return negative
            return -self.scorer.score(smiles)


    class FexofenadineObjective(SmilesObjective):
        """
        Fexofenadine MPO.

        Objective: Make fexofenadine less greasy:
        - Similar to fexofenadine (Tanimoto similarity)
        - TPSA > 90
        - logP < 4

        Target: Fexofenadine (antihistamine)
        """

        def __init__(self, name='fexofenadine', validity_penalty=999999.0, config=None):
            super().__init__(name=name, validity_penalty=validity_penalty, config=config)
            self.scorer = guacamol_scores.fexofenadine_obj()

        def compute_from_smiles(self, smiles: str) -> float:
            # Maximize -> Return negative
            return -self.scorer.score(smiles)


    class CobimetinibObjective(SmilesObjective):
        """
        Cobimetinib MPO.

        Objective: Find molecules similar to cobimetinib with:
        - FCFP4 similarity ≤ 0.7
        - ECFP6 similarity > 0.75
        - Fewer rotatable bonds (≈ 3)
        - Fewer aromatic rings (≈ 3)
        - CNS MPO score optimization

        Target: Cobimetinib (MEK inhibitor for melanoma)
        """

        def __init__(self, name='cobimetinib', validity_penalty=999999.0, config=None):
            super().__init__(name=name, validity_penalty=validity_penalty, config=config)
            self.scorer = guacamol_scores.cobimetinib_obj()

        def compute_from_smiles(self, smiles: str) -> float:
            # Maximize -> Return negative
            return -self.scorer.score(smiles)


    class PioglitazoneObjective(SmilesObjective):
        """
        Pioglitazone MPO.

        Objective: Find molecules dissimilar to pioglitazone but with:
        - Same molecular weight
        - Fewer rotatable bonds (≈ 2)

        Target: Pioglitazone (diabetes drug)
        """

        def __init__(self, name='pioglitazone', validity_penalty=999999.0, config=None):
            super().__init__(name=name, validity_penalty=validity_penalty, config=config)
            self.scorer = guacamol_scores.pioglitazone_mpo()

        def compute_from_smiles(self, smiles: str) -> float:
            # Maximize -> Return negative
            return -self.scorer.score(smiles)


    class DecorationHopObjective(SmilesObjective):
        """
        Decoration hop / scaffold hopping.

        Objective: Keep the scaffold but change decorations:
        - Maintain pharmacophore similarity
        - Remove specific decorations
        - Keep scaffold substructure
        """

        def __init__(self, name='decoration', validity_penalty=999999.0, config=None):
            super().__init__(name=name, validity_penalty=validity_penalty, config=config)
            self.scorer = guacamol_scores.decoration_hop()

        def compute_from_smiles(self, smiles: str) -> float:
            # Maximize -> Return negative
            return -self.scorer.score(smiles)


    class RanolazineObjective(SmilesObjective):
        """
        Ranolazine MPO.

        Objective: Make ranolazine more polar and add fluorine:
        - Similar to ranolazine
        - logP < 7
        - TPSA ≈ 95
        - Contains 1 fluorine atom
        """

        def __init__(self, name='ranolazine', validity_penalty=999999.0, config=None):
            super().__init__(name=name, validity_penalty=validity_penalty, config=config)
            self.scorer = guacamol_scores.ranolazine_mpo()

        def compute_from_smiles(self, smiles: str) -> float:
            # Maximize -> Return negative
            return -self.scorer.score(smiles)


    class ValsartanSmartsObjective(SmilesObjective):
        """
        Valsartan SMARTS.

        Objective: Match valsartan substructure with sitagliptin properties:
        - Contains valsartan SMARTS pattern
        - Similar logP, TPSA, and Bertz complexity to sitagliptin
        """

        def __init__(self, name='valsartan', validity_penalty=999999.0, config=None):
            super().__init__(name=name, validity_penalty=validity_penalty, config=config)
            self.scorer = guacamol_scores.valsartan_smarts()

        def compute_from_smiles(self, smiles: str) -> float:
            # Maximize -> Return negative
            return -self.scorer.score(smiles)


    class PerindoprilObjective(SmilesObjective):
        """
        Perindopril objective.

        Objective: Find molecules similar to perindopril with 2 aromatic rings.

        Target: Perindopril (ACE inhibitor for hypertension)
        """

        def __init__(self, name='perindopril', validity_penalty=999999.0, config=None):
            super().__init__(name=name, validity_penalty=validity_penalty, config=config)
            self.scorer = guacamol_scores.perindopril_obj()

        def compute_from_smiles(self, smiles: str) -> float:
            # Maximize -> Return negative
            return -self.scorer.score(smiles)


    class AmlodipineObjective(SmilesObjective):
        """
        Amlodipine objective.

        Objective: Find molecules similar to amlodipine with 3 rings total.

        Target: Amlodipine (calcium channel blocker for hypertension)
        """

        def __init__(self, name='amlodipine', validity_penalty=999999.0, config=None):
            super().__init__(name=name, validity_penalty=validity_penalty, config=config)
            self.scorer = guacamol_scores.amlodipine_obj()

        def compute_from_smiles(self, smiles: str) -> float:
            # Maximize -> Return negative
            return -self.scorer.score(smiles)


    class SitagliptinObjective(SmilesObjective):
        """
        Sitagliptin objective.

        Objective: Find molecules dissimilar to sitagliptin but with:
        - Similar logP
        - Similar TPSA
        - Same isomeric formula (C16H15F6N5O)

        Target: Sitagliptin (diabetes drug)
        """

        def __init__(self, name='sitagliptin', validity_penalty=999999.0, config=None):
            super().__init__(name=name, validity_penalty=validity_penalty, config=config)
            self.scorer = guacamol_scores.sitagliptin_obj()

        def compute_from_smiles(self, smiles: str) -> float:
            # Maximize -> Return negative
            return -self.scorer.score(smiles)


    class MedianCamphorMentholObjective(SmilesObjective):
        """
        Median molecules 1: Camphor and Menthol.

        Objective: Find molecules with intermediate properties between:
        - Camphor (CC1(C)C2CCC1(C)C(=O)C2)
        - Menthol (CC(C)C1CCC(C)CC1O)
        """

        def __init__(self, name='median_camphor_menthol', validity_penalty=999999.0, config=None):
            super().__init__(name=name, validity_penalty=validity_penalty, config=config)
            self.scorer = guacamol_scores.median_camphor_menthol_obj()

        def compute_from_smiles(self, smiles: str) -> float:
            # Maximize -> Return negative
            return -self.scorer.score(smiles)


    class MedianTadalafilSildenafilObjective(SmilesObjective):
        """
        Median molecules 2: Tadalafil and Sildenafil.

        Objective: Find molecules with intermediate properties between:
        - Tadalafil (erectile dysfunction drug)
        - Sildenafil (Viagra)
        """

        def __init__(self, name='median_tadalafil_sildenafil', validity_penalty=999999.0, config=None):
            super().__init__(name=name, validity_penalty=validity_penalty, config=config)
            self.scorer = guacamol_scores.median_tadalafil_sildenafil_obj()

        def compute_from_smiles(self, smiles: str) -> float:
            # Maximize -> Return negative
            return -self.scorer.score(smiles)


else:
    logger.warning("GuacaMol objectives not available (guacamol_scores module not found)")
