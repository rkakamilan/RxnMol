"""Centralized naming helpers for runs, tasks, and models."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import re
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class RunPath:
    runs_dir: str
    output_dir: str


MODEL_NAMES: Dict[str, str] = {
    "default": "Default",
    "uspto_v1": "USPTO-MIT (v1)",
    "uspto_base": "USPTO-MIT (Base)",
    "ft_uspto_full": "USPTO (ft: Full)",
    "ft_uspto_mix90_10": "USPTO (ft: Mix 90/10)",
    "ft_uspto_freeze4": "USPTO (ft: Freeze 4)",
    "ft_uspto_l2sp": "USPTO (ft: L2SP)",
    "synthetic_base": "Synthetic (Base)",
    "ft_synthetic_full": "Synthetic (ft: Full)",
    "ft_synthetic_mix90_10": "Synthetic (ft: Mix 90/10)",
    "ft_synthetic_freeze4": "Synthetic (ft: Freeze 4)",
    "ft_synthetic_l2sp": "Synthetic (ft: L2SP)",
}

TASK_NAMES: Dict[str, str] = {
    # Standard objectives
    "qed": "QED",
    "sa_score": "SA Score",
    "sa": "SA Score",
    "sas": "SA Score",
    "logp": "LogP",
    "mw": "Mw",
    "molwt": "Mw",
    "molecular_weight": "Mw",
    "tpsa": "TPSA",
    "qed_sa": "QED x SA",

    # SynFlowNet / MPNN objectives
    "seh": "sEH",
    "seh_reaction": "sEH",
    "seh_qed": "sEH x QED",
    "seh_sa": "sEH x SA",
    "synflow_qed": "SynFlow QED",
    "synflow_sa": "SynFlow SA",

    # CB1 objectives
    "cb1_raw": "CB1 (Raw)",
    "cb1_zscore": "CB1 (ZScore)",
    "cb1_minmax": "CB1 (MinMax)",
    "cb1_raw_sa": "CB1 (Raw) x SA",
    "cb1_zscore_sa": "CB1 (ZScore) x SA",
    "cb1_minmax_sa": "CB1 (MinMax) x SA",

    # Legacy CB1 naming (keep for compatibility)
    "cb1_vip36_raw": "CB1 (Raw)",
    "cb1_vip36_minmax": "CB1 (MinMax)",
    "cb1_vip36_zscore": "CB1 (ZScore)",

    # TDC objectives
    "gsk3b": "GSK3B",
    "gsk": "GSK3B",
    "gsk3": "GSK3B",
    "drd2": "DRD2",
    "jnk3": "JNK3",

    # TDC MPO objectives
    "osimertinib_mpo": "Osimertinib MPO",
    "osimertinib": "Osimertinib MPO",
    "fexofenadine_mpo": "Fexofenadine MPO",
    "fexofenadine": "Fexofenadine MPO",
    "ranolazine_mpo": "Ranolazine MPO",
    "ranolazine": "Ranolazine MPO",
    "perindopril_mpo": "Perindopril MPO",
    "perindopril": "Perindopril MPO",
    "amlodipine_mpo": "Amlodipine MPO",
    "amlodipine": "Amlodipine MPO",
    "sitagliptin_mpo": "Sitagliptin MPO",
    "sitagliptin": "Sitagliptin MPO",
    "zaleplon_mpo": "Zaleplon MPO",
    "zaleplon": "Zaleplon MPO",

    # Multi-target combinations
    "gsk3b_jnk3": "GSK3B x JNK3",
    "gsk3b_jnk3_qed": "GSK3B x JNK3 x QED",

    # CSearch docking objectives
    "mpro": "MPro",
    "6m0k": "MPro",
    "mpro_docking": "MPro",
    "btk": "BTK",
    "5p9h": "BTK",
    "btk_docking": "BTK",
    "alk": "ALK",
    "4mkc": "ALK",
    "alk_docking": "ALK",
    "h1n1_na": "H1N1 NA",
    "3ti5": "H1N1 NA",
    "h1n1_docking": "H1N1 NA",
    "neuraminidase": "H1N1 NA",

    # Constrained docking: base × MW penalty
    "seh_mw": "sEH x MW",
    "cb1_raw_mw": "CB1 (Raw) x MW",
    "cb1_zscore_mw": "CB1 (ZScore) x MW",
    "cb1_minmax_mw": "CB1 (MinMax) x MW",
    "mpro_mw": "MPro x MW",
    "btk_mw": "BTK x MW",
    "alk_mw": "ALK x MW",
    "h1n1_na_mw": "H1N1 NA x MW",

    # Constrained docking: base × Lipinski penalty
    "seh_lipinski": "sEH x Lipinski",
    "cb1_raw_lipinski": "CB1 (Raw) x Lipinski",
    "cb1_zscore_lipinski": "CB1 (ZScore) x Lipinski",
    "cb1_minmax_lipinski": "CB1 (MinMax) x Lipinski",
    "mpro_lipinski": "MPro x Lipinski",
    "btk_lipinski": "BTK x Lipinski",
    "alk_lipinski": "ALK x Lipinski",
    "h1n1_na_lipinski": "H1N1 NA x Lipinski",

    # SynFlowNet reward string aliases (cb1_vip36_* and csearch_* prefixes)
    "cb1_vip36_raw_mw": "CB1 (Raw) x MW",
    "cb1_vip36_zscore_mw": "CB1 (ZScore) x MW",
    "cb1_vip36_minmax_mw": "CB1 (MinMax) x MW",
    "cb1_vip36_raw_lipinski": "CB1 (Raw) x Lipinski",
    "cb1_vip36_zscore_lipinski": "CB1 (ZScore) x Lipinski",
    "cb1_vip36_minmax_lipinski": "CB1 (MinMax) x Lipinski",
    "cb1_vip36_raw_sa": "CB1 (Raw) x SA",
    "cb1_vip36_zscore_sa": "CB1 (ZScore) x SA",
    "cb1_vip36_minmax_sa": "CB1 (MinMax) x SA",
    "csearch_mpro": "MPro",
    "csearch_btk": "BTK",
    "csearch_alk": "ALK",
    "csearch_h1n1_na": "H1N1 NA",
    "csearch_mpro_mw": "MPro x MW",
    "csearch_btk_mw": "BTK x MW",
    "csearch_alk_mw": "ALK x MW",
    "csearch_h1n1_na_mw": "H1N1 NA x MW",
    "csearch_mpro_lipinski": "MPro x Lipinski",
    "csearch_btk_lipinski": "BTK x Lipinski",
    "csearch_alk_lipinski": "ALK x Lipinski",
    "csearch_h1n1_na_lipinski": "H1N1 NA x Lipinski",

    # SynFlowNet sEH alias
    "seh_reaction": "sEH",

    # Extra labels used in plots/logs
    "seh_200k": "sEH 200K",
    "seh_50k": "sEH 50K",
}

# Maps constrained/multi-objective task keys to their primary (base) task key.
# Used for comparing constrained runs against unconstrained baselines.
BASE_TASK: Dict[str, str] = {
    # x SA (multi-objective)
    "cb1_raw_sa": "cb1_raw",
    "cb1_zscore_sa": "cb1_zscore",
    "cb1_minmax_sa": "cb1_minmax",
    "cb1_vip36_raw_sa": "cb1_raw",
    "cb1_vip36_zscore_sa": "cb1_zscore",
    "cb1_vip36_minmax_sa": "cb1_minmax",
    "seh_sa": "seh",
    "seh_qed": "seh",
    "qed_sa": "qed",
    "gsk3b_jnk3": "gsk3b",
    "gsk3b_jnk3_qed": "gsk3b",
    # x MW (constrained)
    "seh_mw": "seh",
    "cb1_raw_mw": "cb1_raw",
    "cb1_zscore_mw": "cb1_zscore",
    "cb1_minmax_mw": "cb1_minmax",
    "mpro_mw": "mpro",
    "btk_mw": "btk",
    "alk_mw": "alk",
    "h1n1_na_mw": "h1n1_na",
    # x Lipinski (constrained)
    "seh_lipinski": "seh",
    "cb1_raw_lipinski": "cb1_raw",
    "cb1_zscore_lipinski": "cb1_zscore",
    "cb1_minmax_lipinski": "cb1_minmax",
    "mpro_lipinski": "mpro",
    "btk_lipinski": "btk",
    "alk_lipinski": "alk",
    "h1n1_na_lipinski": "h1n1_na",
}


def slugify(text: str) -> str:
    """Convert text into a unix-friendly slug."""
    if text is None:
        return "unknown"
    value = str(text).strip().lower()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = value.strip("_")
    return value or "unknown"


def task_key(task: str) -> str:
    """Normalize a task key for lookups (lowercase, underscores)."""
    if task is None:
        return "unknown"
    value = str(task).strip().lower()
    value = re.sub(r"\s+", "_", value)
    return value


def task_slug(task: str) -> str:
    """Unix-friendly task name."""
    return slugify(task_key(task))


def task_display_name(task: str) -> str:
    """Human-readable task name."""
    key = task_key(task)
    return TASK_NAMES.get(key, task)


def _resolve_task_key(task: str) -> str:
    """Resolve a task name (slug or display) to its canonical key in TASK_NAMES.

    Handles both slug input ('cb1_raw_mw') and display input ('CB1 (Raw) x MW').
    """
    key = task_key(task)
    if key in TASK_NAMES:
        return key
    # Reverse lookup: display name -> key
    for k, v in TASK_NAMES.items():
        if task_key(v) == key or v == task:
            return k
    return key


def base_task_key(task: str) -> Optional[str]:
    """Return the base (primary) task key for a constrained/multi-objective task.

    Returns None if the task is already a base task (no parent mapping).
    Accepts both slugs ('cb1_raw_mw') and display names ('CB1 (Raw) x MW').
    """
    resolved = _resolve_task_key(task)
    return BASE_TASK.get(resolved)


def base_task_display_name(task: str) -> Optional[str]:
    """Return the display name of the base task, or None if already a base task.

    E.g. 'cb1_raw_mw' -> 'CB1 (Raw)', 'CB1 (Raw) x MW' -> 'CB1 (Raw)'.
    """
    bk = base_task_key(task)
    if bk is None:
        return None
    return task_display_name(bk)


def model_display_name(tag: str, _: Optional[str] = None) -> str:
    """Human-readable model name from hardcoded mapping."""
    if not tag:
        return "unknown"
    return MODEL_NAMES.get(tag, tag)


def model_slug(tag: str) -> str:
    """Unix-friendly model tag."""
    return slugify(tag)


def build_output_dir(
    task: str,
    spec_type: str,
    min_frag: int,
    max_frag: int,
    repeat: int,
    method_name: str = "rxnmol",
) -> str:
    task_part = task_slug(task)
    spec_part = slugify(spec_type)
    method_part = slugify(method_name)
    return f"{method_part}/{task_part}/spec={spec_part}/mf{min_frag}-{max_frag}/r{repeat}"


def build_run_path(
    runs_root: str,
    model_tag: str,
    task: str,
    spec_type: str,
    min_frag: int,
    max_frag: int,
    repeat: int,
    method_name: str = "rxnmol",
) -> RunPath:
    runs_dir = str(Path(runs_root) / model_slug(model_tag))
    output_dir = build_output_dir(task, spec_type, min_frag, max_frag, repeat, method_name)
    return RunPath(runs_dir=runs_dir, output_dir=output_dir)


def _cmd_run_path(args: argparse.Namespace) -> int:
    run_path = build_run_path(
        runs_root=args.runs_root,
        model_tag=args.model_tag,
        task=args.task,
        spec_type=args.spec_type,
        min_frag=args.min_frag,
        max_frag=args.max_frag,
        repeat=args.repeat,
        method_name=args.method_name,
    )
    print(f"runs_dir={run_path.runs_dir}")
    print(f"output_dir={run_path.output_dir}")
    return 0


def _cmd_slug(args: argparse.Namespace) -> int:
    print(slugify(args.text))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RxnMol naming helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-path", help="Compute runs/output dirs")
    run_parser.add_argument("--runs-root", default="./runs")
    run_parser.add_argument("--model-tag", required=True)
    run_parser.add_argument("--task", required=True)
    run_parser.add_argument("--spec-type", required=True)
    run_parser.add_argument("--min-frag", type=int, required=True)
    run_parser.add_argument("--max-frag", type=int, required=True)
    run_parser.add_argument("--repeat", type=int, required=True)
    run_parser.add_argument("--method-name", default="rxnmol")
    run_parser.set_defaults(func=_cmd_run_path)

    slug_parser = subparsers.add_parser("slug", help="Slugify a string")
    slug_parser.add_argument("--text", required=True)
    slug_parser.set_defaults(func=_cmd_slug)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
