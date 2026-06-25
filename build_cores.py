"""Build a core library for scaffold_hop_route: ring scaffolds bearing exactly two
secondary (N-H) ring nitrogens -- bis-secondary-amine cores that can be bis-functionalised
(amide on one N, N-aryl on the other). Combines a curated classic-diamine hand list with a
filter of the Enamine fragment library, validates, canonicalises, dedups.

Run (in the RxnMol dir):
    ~/.conda/envs/rxnmol/bin/python build_cores.py data/enamine_fragments.smi cores.smi
"""
import sys
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

BIS_SEC_AMINE = Chem.MolFromSmarts("[NX3;H1;R;!$(NC=O);!$(NS(=O)=O);!$(N=*)]")  # genuine sp3 secondary amine in a ring (excludes amide/sulfonamide/imine N-H)
MAX_HEAVY = 13

# Classic medchem diamine scaffolds to guarantee coverage (validated below; bad ones dropped).
HAND = [
    "C1CNCCN1",          # piperazine
    "CC1CNCCN1",         # 2-methylpiperazine
    "CC1CN CCNC1".replace(" ", ""),  # 2-methyl-1,4-diazepane
    "C1CNCCNC1",         # 1,4-diazepane (homopiperazine)
    "C1CNCCCNC1",        # 1,4-diazocane (8-membered)
    "C1CNCCNCC1",        # 1,5-diazocane
    "C1NCC12CNC2",       # 2,6-diazaspiro[3.3]heptane
    "C1NCC11CCNCC1",     # 2,7-diazaspiro[3.5]nonane
    "C1CC2(CNC2)CN1",    # 2,6-diazaspiro[3.4]octane-ish
    "C1NCC2CNCC12",      # octahydropyrrolo[3,4-c]pyrrole (3,7-diazabicyclo[3.3.0]octane)
    "C1NC2CNC1C2",       # 2,5-diazabicyclo[2.2.1]heptane
    "C1CC2CNCC1N2",      # 3,8-diazabicyclo[3.2.1]octane
    "C1CCC2NCCNC2C1",    # decahydroquinoxaline
    "C1NCC2NCCC12",      # octahydropyrrolo[3,4-b]pyrrole
    "C1CNCC2(C1)CCNCC2", # 3,9-diazaspiro[5.5]undecane
    "C1NCCC2(N1)CCCCC2", # diazaspiro variant
    "O=C1CNCCN1",        # piperazin-2-one (one amide N is acylated already -> likely 1 NH; validated)
]


def valid_core(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    if m.GetNumHeavyAtoms() > MAX_HEAVY:
        return None
    if len(m.GetSubstructMatches(BIS_SEC_AMINE)) != 2:
        return None
    if m.GetRingInfo().NumRings() < 1:
        return None
    return Chem.MolToSmiles(m)


def main():
    src, out = sys.argv[1], sys.argv[2]
    seen = {}
    n_hand = 0
    for smi in HAND:
        c = valid_core(smi)
        if c and c not in seen:
            seen[c] = "hand"
            n_hand += 1
    n_lib = 0
    for line in open(src):
        toks = line.split()
        if not toks:
            continue
        c = valid_core(toks[0])
        if c and c not in seen:
            seen[c] = "lib"
            n_lib += 1
    with open(out, "w") as fh:
        for c in seen:
            fh.write(c + "\n")
    print(f"cores: {len(seen)} total  (hand {n_hand}, library {n_lib}) -> {out}")
    # show a sample
    for c in list(seen)[:12]:
        print("  ", c)


if __name__ == "__main__":
    main()
