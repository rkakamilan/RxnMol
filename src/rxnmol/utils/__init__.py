"""
Utility functions for rxnmol.
"""

import logging
from pathlib import Path
from datetime import datetime
import shutil

from .metrics import compute_diversity_metrics, compute_qed, compute_sa
from .metrics import compute_mol_metrics, compute_task_metrics
from .building_blocks import load_building_blocks

logger = logging.getLogger(__name__)


def parse_dynamic_overrides(unknown_args):
    """
    Parse unknown arguments into a dictionary for overrides.
    Supports:
        --section.param value
        --section.param=value
        section.param=value (Hydra style)
    """
    overrides = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--"):
            # Format: --key value or --key=value
            key = arg.lstrip("-")
            if "=" in key:
                k, v = key.split("=", 1)
                overrides[k] = v
                i += 1
            else:
                # Format: --key value
                if i + 1 < len(unknown_args) and not unknown_args[i+1].startswith("-"):
                    overrides[key] = unknown_args[i+1]
                    i += 2
                else:
                    # Boolean flag? Assume True
                    overrides[key] = "true" 
                    i += 1
        elif "=" in arg:
            # Format: key=value (Hydra style)
            k, v = arg.split("=", 1)
            overrides[k] = v
            i += 1
        else:
            logger.warning(f"Ignored unknown argument: {arg}")
            i += 1
    return overrides


def resolve_output_dir(config, overwrite: bool = False) -> str:
    """
    Resolve output directory path using structured naming.

    Structure: {runs_dir}/{method}/{output_dir}

    Where:
    - runs_dir: Base directory for all runs (config.runs_dir)
    - method: Method name (rxnmol, MolFinder, etc.) - auto-set from spec_type
    - output_dir: Specific run directory name (config.output_dir or auto-generated)

    If config.output_dir is None, auto-generates:
        {objective}_{solution_params}_T{timestamp}

    Args:
        config: MasterConfig object
        overwrite: If True, cleans existing directory (not recommended)

    Returns:
        Resolved output directory path as string

    Example paths:
        ./runs/rxnmol/zaleplon_mf2-5_T20251201_143045_123456  (auto-generated)
        ./runs/rxnmol/seh_mf2-5_r1                            (explicit output_dir)
    """
    # Build the structured path
    method = config.experiment.method_name or "unknown"
    runs_dir = Path(config.runs_dir)

    # Use explicit output_dir if provided, otherwise auto-generate
    if config.output_dir:
        dir_name = config.output_dir
    else:
        # Auto-generate directory name
        objective = config.objective.name

        # Create solution params string based on spec_type
        params_parts = []

        if config.solution.spec_type == "fragment_route":
            # Fragment-based: include min/max fragments
            min_frag = config.solution.min_fragments
            max_frag = config.solution.max_fragments
            params_parts.append(f"mf{min_frag}-{max_frag}")
        elif config.solution.spec_type == "reaction_mol":
            # Reaction-based: include num_step
            num_step = config.solution.num_step
            params_parts.append(f"step{num_step}")
        # For smiles/other types, no extra params needed

        # Build directory name: {objective}_{params} or just {objective}
        if params_parts:
            dir_name = f"{objective}_{'_'.join(params_parts)}"
        else:
            dir_name = objective

        # Append timestamp with microseconds for uniqueness
        # Format: YYYYMMDD_HHMMSS_microseconds
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dir_name = f"{dir_name}_T{timestamp}"

    # Full path: runs_dir/method/dir_name
    full_path = runs_dir / method / config.objective.name / dir_name

    if overwrite and full_path.exists():
        logger.warning(f"Cleaning existing output directory: {full_path}")
        shutil.rmtree(full_path)

    return str(full_path)

__all__ = [
    "parse_dynamic_overrides",
    "resolve_output_dir",
    "compute_diversity_metrics",
    "compute_qed",
    "compute_sa",
    "compute_mol_metrics",
    "compute_task_metrics",
    "load_building_blocks",
]