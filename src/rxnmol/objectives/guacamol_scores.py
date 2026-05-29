#!/usr/bin/env python
import sys

from rdkit import Chem

from guacamol.common_scoring_functions import (
    TanimotoScoringFunction,
    RdkitScoringFunction,
    CNS_MPO_ScoringFunction,
    IsomerScoringFunction,
    SMARTSScoringFunction,
)
from guacamol.distribution_learning_benchmark import (
    DistributionLearningBenchmark,
    NoveltyBenchmark,
    KLDivBenchmark,
)
from guacamol.frechet_benchmark import FrechetBenchmark
from guacamol.goal_directed_benchmark import GoalDirectedBenchmark
from guacamol.goal_directed_score_contributions import uniform_specification
from guacamol.score_modifier import (
    MinGaussianModifier,
    MaxGaussianModifier,
    ClippedScoreModifier,
    GaussianModifier,
)
from guacamol.scoring_function import (
    ArithmeticMeanScoringFunction,
    GeometricMeanScoringFunction,
    ScoringFunction,
)
from guacamol.utils.descriptors import (
    num_rotatable_bonds,
    num_aromatic_rings,
    logP,
    qed,
    tpsa,
    bertz,
    mol_weight,
    AtomCounter,
    num_rings,
)


def smarts_with_other_target(smarts: str, other_molecule: str) -> ScoringFunction:
    smarts_scoring_function = SMARTSScoringFunction(target=smarts)
    other_mol = Chem.MolFromSmiles(other_molecule)
    target_logp = logP(other_mol)
    target_tpsa = tpsa(other_mol)
    target_bertz = bertz(other_mol)

    lp = RdkitScoringFunction(
        descriptor=logP, score_modifier=GaussianModifier(mu=target_logp, sigma=0.2)
    )
    tp = RdkitScoringFunction(
        descriptor=tpsa, score_modifier=GaussianModifier(mu=target_tpsa, sigma=5)
    )
    bz = RdkitScoringFunction(
        descriptor=bertz, score_modifier=GaussianModifier(mu=target_bertz, sigma=30)
    )

    return GeometricMeanScoringFunction([smarts_scoring_function, lp, tp, bz])


def hard_cobimetinib(max_logP=5.0) -> GoalDirectedBenchmark:
    smiles = "OC1(CN(C1)C(=O)C1=C(NC2=C(F)C=C(I)C=C2)C(F)=C(F)C=C1)C1CCCCN1"

    modifier = ClippedScoreModifier(upper_x=0.7)
    os_tf = TanimotoScoringFunction(smiles, fp_type="FCFP4", score_modifier=modifier)
    os_ap = TanimotoScoringFunction(
        smiles, fp_type="ECFP6", score_modifier=MinGaussianModifier(mu=0.75, sigma=0.1)
    )

    rot_b = RdkitScoringFunction(
        descriptor=num_rotatable_bonds,
        score_modifier=MinGaussianModifier(mu=3, sigma=1),
    )

    rings = RdkitScoringFunction(
        descriptor=num_aromatic_rings, score_modifier=MaxGaussianModifier(mu=3, sigma=1)
    )

    t_cns = ArithmeticMeanScoringFunction(
        [os_tf, os_ap, rot_b, rings, CNS_MPO_ScoringFunction(max_logP=max_logP)]
    )

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(
        name="Cobimetinib MPO",
        objective=t_cns,
        contribution_specification=specification,
    )


# perindopril case.
def perindopril_obj() -> GeometricMeanScoringFunction:
    # perindopril with two aromatic rings
    perindopril = TanimotoScoringFunction(
        "O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC", fp_type="ECFP4"
    )
    arom_rings = RdkitScoringFunction(
        descriptor=num_aromatic_rings, score_modifier=GaussianModifier(mu=2, sigma=0.5)
    )

    specification = uniform_specification(1, 10, 100)
    return GeometricMeanScoringFunction([perindopril, arom_rings])


def amlodipine_obj() -> GeometricMeanScoringFunction:
    # amlodipine with 3 rings
    amlodipine = TanimotoScoringFunction(
        r"Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC", fp_type="ECFP4"
    )
    rings = RdkitScoringFunction(
        descriptor=num_rings, score_modifier=GaussianModifier(mu=3, sigma=0.5)
    )

    specification = uniform_specification(1, 10, 100)
    obj = GeometricMeanScoringFunction([amlodipine, rings])

    # return GoalDirectedBenchmark(name='Amlodipine MPO',
    #                             objective=
    #                             contribution_specification=specification)

    return obj


def sitagliptin_obj() -> GeometricMeanScoringFunction:
    # Find a molecule dissimilar to sitagliptin, but with the same properties
    smiles = "Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F"
    sitagliptin = Chem.MolFromSmiles(smiles)
    target_logp = logP(sitagliptin)
    target_tpsa = tpsa(sitagliptin)

    similarity = TanimotoScoringFunction(
        smiles, fp_type="ECFP4", score_modifier=GaussianModifier(mu=0, sigma=0.1)
    )
    lp = RdkitScoringFunction(
        descriptor=logP, score_modifier=GaussianModifier(mu=target_logp, sigma=0.2)
    )
    tp = RdkitScoringFunction(
        descriptor=tpsa, score_modifier=GaussianModifier(mu=target_tpsa, sigma=5)
    )
    isomers = IsomerScoringFunction("C16H15F6N5O")

    specification = uniform_specification(1, 10, 100)

    # return GoalDirectedBenchmark(name='Sitagliptin MPO',
    #                             objective=GeometricMeanScoringFunction([similarity, lp, tp, isomers]),
    #                             contribution_specification=specification)

    return GeometricMeanScoringFunction([similarity, lp, tp, isomers])


def median_camphor_menthol_obj(mean_cls=GeometricMeanScoringFunction):
    t_camphor = TanimotoScoringFunction("CC1(C)C2CCC1(C)C(=O)C2", fp_type="ECFP4")
    t_menthol = TanimotoScoringFunction("CC(C)C1CCC(C)CC1O", fp_type="ECFP4")
    median = mean_cls([t_menthol, t_camphor])

    specification = uniform_specification(1, 10, 100)

    # return GoalDirectedBenchmark(name='Median molecules 1',
    #                             objective=median,
    #                             contribution_specification=specification)
    return median


def median_tadalafil_sildenafil_obj():
    # median mol between tadalafil and sildenafil
    m1 = TanimotoScoringFunction(
        "O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C", fp_type="ECFP6"
    )
    m2 = TanimotoScoringFunction(
        "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
        fp_type="ECFP6",
    )
    median = GeometricMeanScoringFunction([m1, m2])

    specification = uniform_specification(1, 10, 100)

    # return GoalDirectedBenchmark(name='Median molecules 2',
    #                             objective=median,
    #                             contribution_specification=specification)
    return median


def cobimetinib_obj(max_logP=5.0):
    smiles = "OC1(CN(C1)C(=O)C1=C(NC2=C(F)C=C(I)C=C2)C(F)=C(F)C=C1)C1CCCCN1"

    modifier = ClippedScoreModifier(upper_x=0.7)
    os_tf = TanimotoScoringFunction(smiles, fp_type="FCFP4", score_modifier=modifier)
    os_ap = TanimotoScoringFunction(
        smiles, fp_type="ECFP6", score_modifier=MinGaussianModifier(mu=0.75, sigma=0.1)
    )

    rot_b = RdkitScoringFunction(
        descriptor=num_rotatable_bonds,
        score_modifier=MinGaussianModifier(mu=3, sigma=1),
    )

    rings = RdkitScoringFunction(
        descriptor=num_aromatic_rings, score_modifier=MaxGaussianModifier(mu=3, sigma=1)
    )

    t_cns = ArithmeticMeanScoringFunction(
        [os_tf, os_ap, rot_b, rings, CNS_MPO_ScoringFunction(max_logP=max_logP)]
    )
    return t_cns


def osimertinib_obj(mean_cls=GeometricMeanScoringFunction):
    smiles = "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34"

    modifier = ClippedScoreModifier(upper_x=0.8)
    similar_to_osimertinib = TanimotoScoringFunction(
        smiles, fp_type="FCFP4", score_modifier=modifier
    )

    but_not_too_similar = TanimotoScoringFunction(
        smiles, fp_type="ECFP6", score_modifier=MinGaussianModifier(mu=0.85, sigma=0.1)
    )

    tpsa_over_100 = RdkitScoringFunction(
        descriptor=tpsa, score_modifier=MaxGaussianModifier(mu=100, sigma=10)
    )

    logP_scoring = RdkitScoringFunction(
        descriptor=logP, score_modifier=MinGaussianModifier(mu=1, sigma=1)
    )

    make_osimertinib_great_again = mean_cls(
        [similar_to_osimertinib, but_not_too_similar, tpsa_over_100, logP_scoring]
    )

    return make_osimertinib_great_again


def fexofenadine_obj(mean_cls=GeometricMeanScoringFunction):
    """
    make fexofenadine less greasy
    :return:
    """
    smiles = "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4"

    modifier = ClippedScoreModifier(upper_x=0.8)
    similar_to_fexofenadine = TanimotoScoringFunction(
        smiles, fp_type="AP", score_modifier=modifier
    )

    tpsa_over_90 = RdkitScoringFunction(
        descriptor=tpsa, score_modifier=MaxGaussianModifier(mu=90, sigma=10)
    )

    logP_under_4 = RdkitScoringFunction(
        descriptor=logP, score_modifier=MinGaussianModifier(mu=4, sigma=1)
    )

    optimize_fexofenadine = mean_cls(
        [similar_to_fexofenadine, tpsa_over_90, logP_under_4]
    )

    return optimize_fexofenadine


def zaleplon_obj():
    # zaleplon_with_other_formula with other formula
    zaleplon = TanimotoScoringFunction(
        "O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1", fp_type="ECFP4"
    )
    formula = IsomerScoringFunction("C19H17N3O2")

    return GeometricMeanScoringFunction([zaleplon, formula])


def valsartan_smarts():
    # valsartan smarts with sitagliptin properties
    sitagliptin_smiles = "NC(CC(=O)N1CCn2c(nnc2C(F)(F)F)C1)Cc1cc(F)c(F)cc1F"
    valsartan_smarts = "CN(C=O)Cc1ccc(c2ccccc2)cc1"
    # specification = uniform_specification(1, 10, 100)
    return smarts_with_other_target(valsartan_smarts, sitagliptin_smiles)


def pioglitazone_mpo() -> ScoringFunction:
    # pioglitazone with same mw but less rotatable bonds
    smiles = "O=C1NC(=O)SC1Cc3ccc(OCCc2ncc(cc2)CC)cc3"
    pioglitazone = Chem.MolFromSmiles(smiles)
    target_molw = mol_weight(pioglitazone)

    similarity = TanimotoScoringFunction(
        smiles, fp_type="ECFP4", score_modifier=GaussianModifier(mu=0, sigma=0.1)
    )
    mw = RdkitScoringFunction(
        descriptor=mol_weight, score_modifier=GaussianModifier(mu=target_molw, sigma=10)
    )
    rb = RdkitScoringFunction(
        descriptor=num_rotatable_bonds, score_modifier=GaussianModifier(mu=2, sigma=0.5)
    )

    return GeometricMeanScoringFunction([similarity, mw, rb])


def decoration_hop() -> ScoringFunction:
    smiles = "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C"

    pharmacophor_sim = TanimotoScoringFunction(
        smiles, fp_type="PHCO", score_modifier=ClippedScoreModifier(upper_x=0.85)
    )
    # change deco
    deco1 = SMARTSScoringFunction("CS([#6])(=O)=O", inverse=True)
    deco2 = SMARTSScoringFunction("[#7]-c1ccc2ncsc2c1", inverse=True)

    # keep scaffold
    scaffold = SMARTSScoringFunction(
        "[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12", inverse=False
    )

    deco_hop1_fn = ArithmeticMeanScoringFunction(
        [pharmacophor_sim, deco1, deco2, scaffold]
    )

    return deco_hop1_fn


def ranolazine_mpo() -> ScoringFunction:
    """
    Make start_pop_ranolazine more polar and add a fluorine
    """
    ranolazine = "COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2"

    modifier = ClippedScoreModifier(upper_x=0.7)
    similar_to_ranolazine = TanimotoScoringFunction(
        ranolazine, fp_type="AP", score_modifier=modifier
    )

    logP_under_4 = RdkitScoringFunction(
        descriptor=logP, score_modifier=MaxGaussianModifier(mu=7, sigma=1)
    )

    tpsa_f = RdkitScoringFunction(
        descriptor=tpsa, score_modifier=MaxGaussianModifier(mu=95, sigma=20)
    )

    fluorine = RdkitScoringFunction(
        descriptor=AtomCounter("F"), score_modifier=GaussianModifier(mu=1, sigma=1.0)
    )

    optimize_ranolazine = GeometricMeanScoringFunction(
        [similar_to_ranolazine, logP_under_4, fluorine, tpsa_f]
    )

    return optimize_ranolazine


if __name__ == "__main__":

    obj = cobimetinib_obj()
    print(type(obj))

    mols = (
        "CCCCCCCCCCC",
        "OC1(CN(C1)C(=O)C1=C(NC2=C(F)C=C(I)C=C2)C(F)=C(F)C=C1)C1CCCCN1",
    )

    print(f"{obj.score_list(mols)}")

    obj = osimertinib_obj()
    print(type(obj))
    print(f"{obj.score_list(mols)}")
