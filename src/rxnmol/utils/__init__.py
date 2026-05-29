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
from .naming import (
    slugify,
    task_key,
    task_slug,
    task_display_name,
    model_slug,
    model_display_name,
    build_output_dir,
    build_run_path,
)

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
    """Resolve output directory path using centralized naming."""
    runs_dir = Path(config.runs_dir)
    method = config.experiment.method_name or "rxnmol"

    if config.output_dir:
        dir_name = config.output_dir
    else:
        dir_name = build_output_dir(
            task=config.objective.name,
            spec_type=config.solution.spec_type,
            min_frag=config.solution.min_fragments,
            max_frag=config.solution.max_fragments,
            repeat=config.experiment.repeat_id or 1,
            method_name=method,
        )

    full_path = runs_dir / dir_name

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
    "slugify",
    "task_key",
    "task_slug",
    "task_display_name",
    "model_slug",
    "model_display_name",
    "build_output_dir",
    "build_run_path",
]
