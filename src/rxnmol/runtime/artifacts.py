"""
Artifact persistence for CSA optimization runs.

Handles saving banks, traces, and checkpoints.
"""

from pathlib import Path
from typing import List
import json
import gzip


class ArtifactStore:
    """
    Manages saving and loading of optimization artifacts.

    Example:
        >>> store = ArtifactStore(output_dir='./output')
        >>> store.save_bank(bank, iteration=10)
    """

    def __init__(self, output_dir: str = './output'):
        """
        Initialize artifact store.

        Args:
            output_dir: Directory for saving artifacts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_bank(self, bank: List, iteration: int):
        """Save bank to file."""
        from rdkit import Chem

        bank_file = self.output_dir / f'bank_{iteration}.txt'
        with open(bank_file, 'w') as f:
            # Add header
            f.write("Score\tSMILES\tGenotype\tIntermediates\n")
            for i, candidate in enumerate(bank):
                # Use standardized objective_value (no backward compatibility needed)
                score = candidate.objective_value

                # Format genotype - handle both SMILES strings and RDKit Mol objects
                genotype = candidate.genotype
                if hasattr(genotype, '__class__') and 'Mol' in genotype.__class__.__name__:
                    # It's an RDKit Mol object, convert to SMILES
                    genotype = Chem.MolToSmiles(genotype)
                else:
                    genotype = str(genotype)

                # Format intermediates
                intermediates = str(getattr(candidate, 'intermediates', []))

                f.write(f"{score:.6f}\t{candidate.smiles}\t{genotype}\t{intermediates}\n")

    def save_trace(self, candidate, iteration: int, index: int):
        """Save candidate trace (genealogy)."""
        trace_dir = self.output_dir / f'cycle_{iteration}_traces'
        trace_dir.mkdir(exist_ok=True)

        # Sanitize filename to avoid filesystem issues with SMILES characters
        safe_smiles = "".join([c if c.isalnum() else "_" for c in str(candidate.smiles)])[:50]
        trace_file = trace_dir / f'{index}_{safe_smiles}.trace'
        
        with open(trace_file, 'w') as f:
            f.write(f"Score: {candidate.objective_value:.6f}\n")
            f.write(f"SMILES: {candidate.smiles}\n")
            f.write(f"Genotype: {candidate.genotype}\n")
            
            # Handle reactions from metadata (FragmentRouteSpec)
            reactions = candidate.metadata.get('reactions', [])
            if reactions:
                f.write(f"Reactions:\n")
                for rxn in reactions:
                    f.write(f"  {rxn}\n")

    def save_config(self, config, filename: str = 'config.yaml'):
        """Save configuration."""
        config_file = self.output_dir / filename
        config.save(str(config_file))

    def save_meta(self, meta: dict, filename: str = 'meta.json'):
        """Save run metadata for analysis."""
        meta_file = self.output_dir / filename
        meta_file.write_text(json.dumps(meta, indent=2))
