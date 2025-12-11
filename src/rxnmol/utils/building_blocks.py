"""Utility helpers for loading building block files."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def load_building_blocks(
    building_blocks_file: PathLike,
) -> Tuple[List[str], Path]:
    """Read building blocks from a text or gzipped file.

    Returns a tuple of (entries, resolved_path).
    
    is split on whitespace and the first token is treated as the SMILES string.
    """
    if not isinstance(building_blocks_file, Path):
        building_blocks_file = Path(building_blocks_file)

    is_gzipped = building_blocks_file.suffix == ".gz"
    open_fn = gzip.open if is_gzipped else open
    mode = "rt" if is_gzipped else "r"

    entries: List[str] = []
    with open_fn(building_blocks_file, mode) as handle:
        for line in handle:
            stripped = line.strip()
            # if not stripped or stripped.startswith("#"):
            #     continue

            # assume first token is the SMILES string
            entries.append(stripped.split()[0])

    return np.array(entries)