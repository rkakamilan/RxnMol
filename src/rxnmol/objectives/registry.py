"""
Lazy Objective Registry.

Provides deferred loading of objective modules - only imports when requested.
This avoids loading heavy dependencies (PyTorch, TDC, GuacaMol) at startup.

Usage:
    >>> from rxnmol.objectives.registry import get_objective, list_objectives
    >>> obj = get_objective('qed')  # Only loads objectives_standard.py now
    >>> obj.compute(mol)

    >>> list_objectives()  # Shows all available without loading
    {'qed': 'QED (Drug-likeness)', 'seh': 'sEH Binding', ...}
"""

from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


# Global registry: id → dict with metadata
_REGISTRY: Dict[str, dict] = {}

# Per-process cache of instantiated objectives
_INSTANCE_CACHE: Dict[str, Any] = {}


def register(
    id: str,
    module: str,
    class_name: str,
    display_name: Optional[str] = None,
    aliases: Optional[list] = None,
    supports_batch: bool = False,
    description: str = ""
):
    """
    Register an objective without importing it.

    Args:
        id: Internal identifier (e.g., 'seh_sa')
        module: Full module path (e.g., 'rxnmol.objectives.objectives_synflownet')
        class_name: Class name in module (e.g., 'SEHSAObjective')
        display_name: Human-readable name (e.g., 'sEH × SA')
        aliases: Alternative IDs that map to the same objective
        supports_batch: If True, objective has efficient compute_batch()
        description: Brief description for help/docs
    """
    display = display_name or id.replace('_', ' ').title()

    info = {
        'id': id,
        'display_name': display,
        'module': module,
        'class_name': class_name,
        'supports_batch': supports_batch,
        'description': description
    }

    _REGISTRY[id] = info

    # Register aliases pointing to the same info
    for alias in (aliases or []):
        _REGISTRY[alias] = info


def get_objective(
    name: str,
    validity_penalty: float = 999999.0,
    **kwargs
) -> 'ObjectiveFunction':
    """
    Get or create an objective instance (lazy loading).

    Args:
        name: Objective ID (e.g., 'qed', 'seh_sa')
        validity_penalty: Penalty for invalid molecules
        **kwargs: Additional arguments passed to objective constructor

    Returns:
        Instantiated ObjectiveFunction

    Raises:
        ValueError: If objective name is unknown
    """
    if name not in _REGISTRY:
        available = sorted(set(info['id'] for info in _REGISTRY.values()))
        raise ValueError(
            f"Unknown objective: '{name}'. "
            f"Available: {available}"
        )

    # Cache key includes name and key kwargs that affect behavior
    cache_key = f"{name}_{validity_penalty}_{hash(frozenset(kwargs.items()))}"

    if cache_key in _INSTANCE_CACHE:
        logger.debug(f"Reusing cached objective: {name}")
        return _INSTANCE_CACHE[cache_key]

    info = _REGISTRY[name]

    # Dynamic import - only happens on first use
    import importlib
    try:
        module = importlib.import_module(info['module'])
        cls = getattr(module, info['class_name'])
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Failed to load objective '{name}' from {info['module']}.{info['class_name']}: {e}"
        )

    logger.info(f"Creating objective: {info['display_name']} ({name})")

    # Instantiate with name and validity_penalty
    instance = cls(name=name, validity_penalty=validity_penalty, **kwargs)

    _INSTANCE_CACHE[cache_key] = instance
    return instance


def list_objectives() -> Dict[str, str]:
    """
    List all registered objectives with display names.

    Returns:
        Dict of {id: display_name}
    """
    # Deduplicate by id (aliases share the same info)
    seen = set()
    result = {}
    for key, info in _REGISTRY.items():
        if info['id'] not in seen:
            result[info['id']] = info['display_name']
            seen.add(info['id'])
    return result


def clear_cache():
    """Clear the instance cache (useful for testing)."""
    _INSTANCE_CACHE.clear()


# ============================================================================
# Register ALL objectives (NO imports happen here!)
# ============================================================================

# -----------------------------------------------------------------------------
# Standard RDKit objectives (fast, CPU-only)
# -----------------------------------------------------------------------------
register(
    'qed',
    'rxnmol.objectives.objectives_standard',
    'QEDObjective',
    display_name='QED (Drug-likeness)',
    description='Quantitative Estimate of Drug-likeness [0-1]'
)

register(
    'sa_score',
    'rxnmol.objectives.objectives_standard',
    'SAScoreObjective',
    display_name='SA Score',
    aliases=['sa', 'sas'],
    description='Synthetic Accessibility Score (normalized 0-1)'
)

register(
    'logp',
    'rxnmol.objectives.objectives_standard',
    'LogPObjective',
    display_name='LogP',
    description='Octanol-water partition coefficient'
)

register(
    'mw',
    'rxnmol.objectives.objectives_standard',
    'MolecularWeightObjective',
    display_name='Molecular Weight',
    aliases=['molwt', 'molecular_weight'],
    description='Molecular weight in Daltons'
)

register(
    'tpsa',
    'rxnmol.objectives.objectives_standard',
    'TPSAObjective',
    display_name='TPSA',
    description='Topological Polar Surface Area'
)

register(
    'qed_sa',
    'rxnmol.objectives.objectives_standard',
    'QEDSAObjective',
    display_name='QED × SA',
    description='Combined QED and SA score'
)

# -----------------------------------------------------------------------------
# SynFlowNet / MPNN objectives (GPU, batch-capable)
# -----------------------------------------------------------------------------
register(
    'seh',
    'rxnmol.objectives.objectives_synflownet',
    'SEHObjective',
    display_name='sEH Binding',
    supports_batch=True,
    description='Soluble Epoxide Hydrolase binding (MPNN proxy)'
)

register(
    'seh_qed',
    'rxnmol.objectives.objectives_synflownet',
    'SEHQEDObjective',
    display_name='sEH × QED',
    supports_batch=True,
    description='sEH binding weighted by QED'
)

register(
    'seh_sa',
    'rxnmol.objectives.objectives_synflownet',
    'SEHSAObjective',
    display_name='sEH × SA',
    supports_batch=True,
    description='sEH binding weighted by synthetic accessibility'
)

register(
    'synflow_qed',
    'rxnmol.objectives.objectives_synflownet',
    'SynFlowQEDObjective',
    display_name='SynFlow QED',
    description='QED with SynFlowNet normalization'
)

register(
    'synflow_sa',
    'rxnmol.objectives.objectives_synflownet',
    'SynFlowSAObjective',
    display_name='SynFlow SA',
    description='SA with SynFlowNet normalization (threshold at 3.5)'
)

# CB1 objectives
register(
    'cb1_raw',
    'rxnmol.objectives.objectives_synflownet',
    'CB1RawObjective',
    display_name='CB1 Raw Score',
    supports_batch=True,
    description='CB1 receptor binding (raw docking score)'
)

register(
    'cb1_zscore',
    'rxnmol.objectives.objectives_synflownet',
    'CB1ZscoreObjective',
    display_name='CB1 Z-Score',
    supports_batch=True,
    description='CB1 receptor binding (z-score normalized)'
)

register(
    'cb1_minmax',
    'rxnmol.objectives.objectives_synflownet',
    'CB1MinMaxObjective',
    display_name='CB1 MinMax',
    supports_batch=True,
    description='CB1 receptor binding (min-max normalized)'
)

# CB1 × SA combined objectives
register(
    'cb1_raw_sa',
    'rxnmol.objectives.objectives_synflownet',
    'CB1RawSAObjective',
    display_name='CB1 Raw × SA',
    supports_batch=True,
    description='Product of CB1 raw docking score (inverted) and SA (normalized 0-1, saturation at 3.5)'
)

register(
    'cb1_zscore_sa',
    'rxnmol.objectives.objectives_synflownet',
    'CB1ZscoreSAObjective',
    display_name='CB1 Z-Score × SA',
    supports_batch=True,
    description='Product of CB1 z-score (inverted) and SA (normalized 0-1, saturation at 3.5)'
)

register(
    'cb1_minmax_sa',
    'rxnmol.objectives.objectives_synflownet',
    'CB1MinMaxSAObjective',
    display_name='CB1 MinMax × SA',
    supports_batch=True,
    description='Product of CB1 min-max score [0-1] and SA (normalized 0-1, saturation at 3.5)'
)

# -----------------------------------------------------------------------------
# TDC objectives (batch-capable via Oracle list input)
# -----------------------------------------------------------------------------
register(
    'gsk3b',
    'rxnmol.objectives.objectives_tdc',
    'GSK3BObjective',
    display_name='GSK3β Activity',
    aliases=['gsk', 'gsk3', 'gsk3b'],
    supports_batch=True,
    description='Glycogen Synthase Kinase 3 Beta activity [0-1]'
)

register(
    'drd2',
    'rxnmol.objectives.objectives_tdc',
    'DRD2Objective',
    display_name='DRD2 Activity',
    supports_batch=True,
    description='Dopamine Receptor D2 activity [0-1]'
)

register(
    'jnk3',
    'rxnmol.objectives.objectives_tdc',
    'JNK3Objective',
    display_name='JNK3 Activity',
    supports_batch=True,
    description='c-Jun N-terminal Kinase 3 activity [0-1]'
)

# TDC MPO objectives (7 drugs)
register(
    'osimertinib_mpo',
    'rxnmol.objectives.objectives_tdc',
    'OsimertinibMPOObjective',
    display_name='Osimertinib MPO',
    aliases=['osimertinib'],
    supports_batch=True,
    description='EGFR inhibitor multi-property optimization'
)

register(
    'fexofenadine_mpo',
    'rxnmol.objectives.objectives_tdc',
    'FexofenadineMPOObjective',
    display_name='Fexofenadine MPO',
    aliases=['fexofenadine'],
    supports_batch=True,
    description='Antihistamine multi-property optimization'
)

register(
    'ranolazine_mpo',
    'rxnmol.objectives.objectives_tdc',
    'RanolazineMPOObjective',
    display_name='Ranolazine MPO',
    aliases=['ranolazine'],
    supports_batch=True,
    description='Antianginal multi-property optimization'
)

register(
    'perindopril_mpo',
    'rxnmol.objectives.objectives_tdc',
    'PerindoprilMPOObjective',
    display_name='Perindopril MPO',
    aliases=['perindopril'],
    supports_batch=True,
    description='ACE inhibitor multi-property optimization'
)

register(
    'amlodipine_mpo',
    'rxnmol.objectives.objectives_tdc',
    'AmlodipineMPOObjective',
    display_name='Amlodipine MPO',
    aliases=['amlodipine'],
    supports_batch=True,
    description='Calcium channel blocker multi-property optimization'
)

register(
    'sitagliptin_mpo',
    'rxnmol.objectives.objectives_tdc',
    'SitagliptinMPOObjective',
    display_name='Sitagliptin MPO',
    aliases=['sitagliptin'],
    supports_batch=True,
    description='DPP-4 inhibitor multi-property optimization'
)

register(
    'zaleplon_mpo',
    'rxnmol.objectives.objectives_tdc',
    'ZaleplonMPOObjective',
    display_name='Zaleplon MPO',
    supports_batch=True,
    description='Sedative-hypnotic multi-property optimization'
)

# Multi-target combinations
register(
    'gsk3b_jnk3',
    'rxnmol.objectives.objectives_tdc',
    'GSK3B_JNK3Objective',
    display_name='GSK3β × JNK3',
    supports_batch=True,
    description='Dual kinase activity (geometric mean)'
)

register(
    'gsk3b_jnk3_qed',
    'rxnmol.objectives.objectives_tdc',
    'GSK3B_JNK3_QEDObjective',
    display_name='GSK3β × JNK3 × QED',
    supports_batch=True,
    description='Dual kinase activity with drug-likeness'
)

# -----------------------------------------------------------------------------
# GuacaMol objectives
# -----------------------------------------------------------------------------
# register(
#     'zaleplon',
#     'rxnmol.objectives.objectives_guacamol',
#     'ZaleplonObjective',
#     display_name='Zaleplon (GuacaMol)',
#     description='Zaleplon MPO from GuacaMol benchmark'
# )

# register(
#     'cobimetinib',
#     'rxnmol.objectives.objectives_guacamol',
#     'CobimetinibObjective',
#     display_name='Cobimetinib MPO',
#     description='MEK inhibitor multi-property optimization'
# )

# register(
#     'pioglitazone',
#     'rxnmol.objectives.objectives_guacamol',
#     'PioglitazoneObjective',
#     display_name='Pioglitazone MPO',
#     description='Diabetes drug multi-property optimization'
# )

# register(
#     'decoration_hop',
#     'rxnmol.objectives.objectives_guacamol',
#     'DecorationHopObjective',
#     display_name='Decoration Hop',
#     aliases=['decoration'],
#     description='Scaffold hopping benchmark'
# )

# register(
#     'valsartan_smarts',
#     'rxnmol.objectives.objectives_guacamol',
#     'ValsartanSmartsObjective',
#     display_name='Valsartan SMARTS',
#     aliases=['valsartan'],
#     description='Valsartan substructure with sitagliptin properties'
# )

# register(
#     'median_camphor_menthol',
#     'rxnmol.objectives.objectives_guacamol',
#     'MedianCamphorMentholObjective',
#     display_name='Median (Camphor-Menthol)',
#     aliases=['median1'],
#     description='Intermediate properties between camphor and menthol'
# )

# register(
#     'median_tadalafil_sildenafil',
#     'rxnmol.objectives.objectives_guacamol',
#     'MedianTadalafilSildenafilObjective',
#     display_name='Median (Tadalafil-Sildenafil)',
#     aliases=['median2'],
#     description='Intermediate properties between tadalafil and sildenafil'
# )

# -----------------------------------------------------------------------------
# CSearch docking energy objectives (GNN-based, batch-capable)
# Reference: Kim et al. "CSearch: chemical space search via virtual synthesis
#            and global optimization" J Cheminform (2024)
# -----------------------------------------------------------------------------
register(
    'mpro',
    'rxnmol.objectives.objectives_csearch',
    'MProObjective',
    display_name='MPro (SARS-CoV-2)',
    aliases=['6m0k', 'mpro_docking'],
    supports_batch=True,
    description='SARS-CoV-2 Main Protease docking energy (GNN proxy)'
)

register(
    'btk',
    'rxnmol.objectives.objectives_csearch',
    'BTKObjective',
    display_name='BTK Kinase',
    aliases=['5p9h', 'btk_docking'],
    supports_batch=True,
    description='Tyrosine-protein kinase BTK docking energy (GNN proxy)'
)

register(
    'alk',
    'rxnmol.objectives.objectives_csearch',
    'ALKObjective',
    display_name='ALK Kinase',
    aliases=['4mkc', 'alk_docking'],
    supports_batch=True,
    description='Anaplastic Lymphoma Kinase docking energy (GNN proxy)'
)

register(
    'h1n1_na',
    'rxnmol.objectives.objectives_csearch',
    'H1N1NAObjective',
    display_name='H1N1 Neuraminidase',
    aliases=['3ti5', 'h1n1_docking', 'neuraminidase'],
    supports_batch=True,
    description='H1N1 Neuraminidase docking energy (GNN proxy)'
)


# -----------------------------------------------------------------------------
# CLI: python -m rxnmol.objectives.registry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Available Objective Functions:")
    print("=" * 60)
    for obj_id, display in sorted(list_objectives().items()):
        info = _REGISTRY[obj_id]
        print(f"  {obj_id:25} {display}")
        if info.get('description'):
            print(f"  {' ':25} {info['description']}")
    print("=" * 60)
    print(f"Total: {len(list_objectives())} objectives")
