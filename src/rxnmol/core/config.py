"""
Hierarchical configuration system for Fragment-CSA.

Provides type-safe, validated configuration with four-tier override system:
    defaults → YAML file → CLI args → environment variables

Example:
    >>> config = MasterConfig.from_yaml('input.yaml')
    >>> config.override_from_cli({'csa.bank_size': 200})
    >>> config.validate()
    >>> config.save('output/config.yaml')
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml
import os
import re

# Register custom YAML representer for Path objects
def path_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))

yaml.add_representer(Path, path_representer)
yaml.add_representer(type(Path()), path_representer) # Handle concrete PosixPath/WindowsPath


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class CSAConfig:
    """CSA algorithm parameters."""

    bank_size: int = 100
    seed_size: int = 60
    max_iter: int = 100
    d_init: Optional[float] = 0.6  # If None, distance cutoff will be d_avg/2
    d_min: float = 0.1
    gamma: float = 0.990
    n_cross_op1: int = 5
    n_cross_op2: int = 5
    n_mut_op: int = 5
    early_stop_patience: Optional[int] = None  # If None, defaults to ceil(0.10 * max_iter)

    # Local optimization parameters (greedy hill climbing)
    local_optimization_enabled: bool = False
    local_optimization_frequency: int = 1  # Every N iterations (1 = every iteration)
    local_optimization_max_steps: int = 3  # Max hill-climbing steps per candidate
    local_optimization_neighbors: int = 5  # Number of neighbors (mutations) per candidate per step
    # local_optimization_apply_to_bank: bool = False  # If True, optimize bank+trials (like original SMILES CSA)
    # Note: Local optimization is ALWAYS applied to initial random candidates (matching original behavior)

    def validate(self):
        """Validate CSA parameters."""
        assert 0 < self.seed_size <= self.bank_size, \
            f"seed_size ({self.seed_size}) must be > 0 and <= bank_size ({self.bank_size})"
        if self.d_init is not None:
            assert 0 < self.d_min < self.d_init <= 1.0, \
                f"Distance cutoffs invalid: d_min={self.d_min}, d_init={self.d_init}"
        assert 0 < self.gamma < 1.0, \
            f"Gamma ({self.gamma}) must be in (0, 1)"
        assert self.local_optimization_frequency > 0, \
            f"local_optimization_frequency must be positive, got {self.local_optimization_frequency}"
        assert self.local_optimization_max_steps > 0, \
            f"local_optimization_max_steps must be positive, got {self.local_optimization_max_steps}"
        assert self.local_optimization_neighbors > 0, \
            f"local_optimization_neighbors must be positive, got {self.local_optimization_neighbors}"
        if self.early_stop_patience is not None:
            assert self.early_stop_patience > 0, \
                f"early_stop_patience must be positive, got {self.early_stop_patience}"


@dataclass
class SolutionConfig:
    """Solution representation parameters."""

    spec_type: str = "fragment_route"  # "fragment_route", "raw_smiles", "hybrid"

    # Fragment-specific parameters
    max_fragments: int = 5
    min_fragments: int = 2
    max_mol_size: Optional[int] = None
    
    # ReactionMol specific parameters
    num_step: int = 5

    # Scaffold-hop (core-hopping): two FIXED reactive warhead precursors
    warhead_a: Optional[str] = None
    warhead_b: Optional[str] = None
    warheads: Optional[List[str]] = None          # mode B: core-first, N fixed warheads
    warhead_patterns: Optional[List[str]] = None  # optional gate SMARTS (one per warhead)

    def validate(self):
        """Validate solution parameters."""
        valid_types = ["fragment_route", "smiles", "hybrid", "reaction_mol", "scaffold_hop_route"]
        assert self.spec_type in valid_types, \
            f"Invalid spec_type: {self.spec_type}. Must be one of {valid_types}"
        assert 0 < self.min_fragments <= self.max_fragments, \
            f"Fragment count invalid: min={self.min_fragments}, max={self.max_fragments}"
        if self.max_mol_size is not None:
            assert self.max_mol_size > 0, \
                f"max_mol_size must be positive, got {self.max_mol_size}"
        if self.spec_type == "reaction_mol":
            assert self.num_step > 0, \
                f"num_step must be positive, got {self.num_step}"
        if self.spec_type == "scaffold_hop_route":
            assert self.warheads or (self.warhead_a and self.warhead_b), "scaffold_hop_route requires solution.warheads or (warhead_a and warhead_b)"


@dataclass
class DataConfig:
    """Data source parameters."""

    building_blocks_file: Optional[Path] = None
    data_dir: Optional[Path] = None
    cache_path: Optional[Path] = None

    def validate(self):
        """Validate data parameters."""
        if isinstance(self.building_blocks_file, str):
            self.building_blocks_file = Path(self.building_blocks_file)

        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)

        if isinstance(self.cache_path, str):
            self.cache_path = Path(self.cache_path)

        if self.building_blocks_file is None:
            raise AssertionError(
                "DataConfig requires building_blocks_file to be set or provided file name located in data_dir"
            )
        else:
            # If building_blocks_file is just a filename, prepend data_dir
            bb_path = Path(self.building_blocks_file)
            original_path_str = str(bb_path)
            is_relative_filename = not bb_path.is_absolute() and len(bb_path.parts) == 1

            if is_relative_filename and self.data_dir:
                bb_path = self.data_dir / bb_path

            resolved_path = bb_path.resolve()
            if not resolved_path.exists():
                if is_relative_filename and self.data_dir:
                    raise FileNotFoundError(
                        f"Building blocks file '{original_path_str}' not found in data_dir "
                        f"'{self.data_dir}'. Attempted path: {resolved_path}"
                    )
                else:
                    raise FileNotFoundError(
                        f"Building blocks file not found: {resolved_path}"
                    )
            self.building_blocks_file = resolved_path


@dataclass
class ReactionModelConfig:
    """Reaction prediction model configuration."""

    provider: str = "transformer_v1"  # transformer_v1 | transformer_v2

    # Common paths
    model_dir: Optional[Path] = None
    checkpoint_path: Optional[Path] = None
    tokenizer_path: Optional[Path] = None
    repo_path: Optional[Path] = None  # For transformer_v2 (external repo)
    tag: Optional[str] = None  # Optional model tag for naming

    # Inference settings
    batch_size: Optional[int] = None
    max_len: Optional[int] = 256
    beam_size: int = 1
    length_penalty: float = 1.0
    task: str = "forward"

    # Legacy-only settings
    enforce_cuda: bool = True

    # Compilation settings
    compile: bool = True
    compile_backend: str = "inductor"
    compile_mode: str = "reduce-overhead"

    def validate(self):
        """Normalize path fields and provider string."""
        if isinstance(self.model_dir, str):
            self.model_dir = Path(self.model_dir)
        if isinstance(self.checkpoint_path, str):
            self.checkpoint_path = Path(self.checkpoint_path)
        if isinstance(self.tokenizer_path, str):
            self.tokenizer_path = Path(self.tokenizer_path)
        if isinstance(self.repo_path, str):
            self.repo_path = Path(self.repo_path)

        if self.provider:
            self.provider = self.provider.strip()


@dataclass
class OperatorConfig:
    """Genetic operation parameters."""

    crossover_enabled: bool = True
    mutation_enabled: bool = True

    # SMILES-specific mutation probabilities
    crossover_prob: float = 0.5  # Probability of crossover vs mutation
    mutation_add_prob: float = 0.33  # Probability of add_atom mutation
    mutation_delete_prob: float = 0.33  # Probability of delete_atom mutation
    mutation_replace_prob: float = 0.34  # Probability of replace_atom mutation

    # Operator selection (auto-populated by SolutionSpec)
    crossover_types: List[str] = field(default_factory=lambda: ["fragment_list"])
    mutation_types: List[str] = field(default_factory=lambda: ["fragment_replace"])

    # Operator weights (for adaptive selection)
    crossover_weights: Optional[Dict[str, float]] = None
    mutation_weights: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Set default mutation weights if not provided."""
        if self.mutation_weights is None:
            self.mutation_weights = {
                'replace': 0.5,
                'add': 0.25,
                'remove': 0.25
            }


@dataclass
class ObjectiveConfig:
    """Objective function parameters."""

    name: str = "zaleplon"

    # For multi-objective optimization
    components: Optional[List[str]] = None
    weights: Optional[List[float]] = None
    aggregation: str = "weighted_sum"  # "weighted_sum", "product", "lexicographic"

    # Constraint penalties
    validity_penalty: float = 999.0

    # Additional kwargs for specific objectives
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """Monitoring and logging parameters."""

    save_interval: int = 1  # Save artifacts every N iterations
    log_level: str = "INFO"

    # Metrics tracking
    track_diversity: bool = True
    track_timing: bool = True
    track_operators: bool = True

    # Visualization
    plot_progress: bool = True
    plot_format: str = "png"  # "png", "pdf", "svg"

    # Console output
    progress_bar: bool = True
    verbose: bool = False


@dataclass
class PersistenceConfig:
    """Artifact persistence parameters."""

    save_banks: bool = True
    save_traces: bool = True
    save_cache: bool = True
    save_config: bool = True
    save_metrics: bool = True
    cache_auto_merge: bool = False
    cache_auto_merge_delete: bool = False
    cache_max_entries: Optional[int] = 5000000

    # Compression
    compress_banks: bool = True
    compression_format: str = "gzip"  # "gzip", "bzip2", "lzma"

    # Checkpoint/Resume
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 50  # iterations


@dataclass
class RuntimeConfig:
    """Runtime execution parameters."""

    random_seed: Optional[int] = None  # None = use system time for random seed
    num_workers: int = 1  # For parallel evaluation
    device: str = "cuda"  # "cuda", "cpu"
    require_cuda: bool = False

    # Memory management
    cache_max_size: int = 100000
    reaction_cache_enabled: bool = True


@dataclass
class ExperimentConfig:
    """Experiment management parameters."""

    method_name: Optional[str] = None  # Auto-set based on spec_type: "rxnmol" or "MolFinder"
    repeat_id: Optional[int] = None  # Repeat number for multiple runs

@dataclass
class MasterConfig:
    """
    Top-level unified configuration for Fragment-CSA.

    Supports four-tier override system:
        1. Defaults (defined in dataclasses above)
        2. YAML file
        3. CLI arguments (dot notation: --csa.bank_size 200)
        4. Environment variables (FRAGCSA__CSA__BANK_SIZE=200)

    Example:
        >>> config = MasterConfig.from_yaml('input.yaml')
        >>> config.override_from_cli({'csa.bank_size': 200, 'output_dir': './results'})
        >>> config.override_from_env()
        >>> config.validate()
    """

    # Core configuration sections
    csa: CSAConfig = field(default_factory=CSAConfig)
    solution: SolutionConfig = field(default_factory=SolutionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reaction_model: ReactionModelConfig = field(default_factory=ReactionModelConfig)
    operators: OperatorConfig = field(default_factory=OperatorConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # Top-level settings
    runs_dir: str = "./runs"  # Base directory for all experiment runs
    output_dir: Optional[str] = None  # Specific run directory name (auto-generated if None)

    # ========================================================================
    # Configuration Loading
    # ========================================================================

    @staticmethod
    def _resolve_interpolations(data: dict) -> dict:
        """
        Resolve variable interpolations in the format ${section.key}.
        Supports nested resolution and type preservation.
        """
        # Helper to flatten dictionary for lookup
        def flatten(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        # Helper to resolve a single value
        def resolve_value(val, flat_data):
            if not isinstance(val, str):
                return val, False
            
            pattern = re.compile(r'\$\{(.*?)\}')
            matches = pattern.findall(val)
            
            if not matches:
                return val, False
                
            # Handle single full replacement to preserve type (e.g. "${csa.bank_size}" -> 100)
            if len(matches) == 1 and val.strip() == f"${{{matches[0]}}}":
                key = matches[0]
                if key in flat_data:
                    return flat_data[key], True
                return val, False

            # Handle string interpolation (e.g. "prefix_${run_name}")
            new_val = val
            changed = False
            for key in matches:
                if key in flat_data:
                    repl = str(flat_data[key])
                    if f"${{{key}}}" in new_val:
                        new_val = new_val.replace(f"${{{key}}}", repl)
                        changed = True
            
            return new_val, changed

        # Multi-pass resolution to handle dependencies
        max_passes = 5
        current_data = data
        
        for _ in range(max_passes):
            flat_data = flatten(current_data)
            any_changes = False
            
            def walk_and_replace(obj):
                nonlocal any_changes
                if isinstance(obj, dict):
                    return {k: walk_and_replace(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [walk_and_replace(v) for v in obj]
                elif isinstance(obj, str):
                    new_val, changed = resolve_value(obj, flat_data)
                    if changed:
                        any_changes = True
                    return new_val
                return obj

            current_data = walk_and_replace(current_data)
            if not any_changes:
                break
                
        return current_data

    @classmethod
    def from_yaml(cls, path: str) -> 'MasterConfig':
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            MasterConfig instance with values from YAML

        Example:
            >>> config = MasterConfig.from_yaml('configs/zaleplon.yaml')
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        
        # Resolve interpolations
        data = cls._resolve_interpolations(data)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> 'MasterConfig':
        """
        Create configuration from dictionary with nested sections.

        Args:
            data: Dictionary with configuration values

        Returns:
            MasterConfig instance
        """
        data_section = dict(data.get('data', {}))

        return cls(
            csa=CSAConfig(**data.get('csa', {})),
            solution=SolutionConfig(**data.get('solution', {})),
            data=DataConfig(**data_section),
            reaction_model=ReactionModelConfig(**data.get('reaction_model', {})),
            operators=OperatorConfig(**data.get('operators', {})),
            objective=ObjectiveConfig(**data.get('objective', {})),
            monitoring=MonitoringConfig(**data.get('monitoring', {})),
            persistence=PersistenceConfig(**data.get('persistence', {})),
            runtime=RuntimeConfig(**data.get('runtime', {})),
            experiment=ExperimentConfig(**data.get('experiment', {})),
            runs_dir=data.get('runs_dir', './runs'),
            output_dir=data.get('output_dir'),
        )

    # ========================================================================
    # Configuration Override
    # ========================================================================

    def override_from_cli(self, args: Dict[str, Any]):
        """
        Override configuration from command-line arguments.

        Supports dot notation for nested parameters:
            --csa.bank_size 200
            --objective.name qed
            --output_dir ./results

        Args:
            args: Dictionary of argument name/value pairs
        """
        for key, value in args.items():
            if value is None:
                continue

            parts = key.split('.')
            obj = self
            
            # Navigate to the parent of the target attribute
            valid_path = True
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    valid_path = False
                    break
            
            if not valid_path:
                continue

            # Set the value on the leaf attribute
            param = parts[-1]
            if hasattr(obj, param):
                # Handle 'null' string -> None conversion
                if str(value).lower() in ('null', 'none', '~'):
                    setattr(obj, param, None)
                    continue

                # Cast to correct type
                current = getattr(obj, param)
                if current is not None:
                    if isinstance(current, bool):
                        # Handle boolean strings specifically
                        if str(value).lower() in ('true', '1', 'yes', 'on'):
                            value = True
                        elif str(value).lower() in ('false', '0', 'no', 'off'):
                            value = False
                    else:
                        try:
                            value = type(current)(value)
                        except ValueError:
                            pass
                else:
                    # For None values, try to infer type from type hints
                    try:
                        # Get type hint from the dataclass
                        import typing
                        type_hints = typing.get_type_hints(type(obj))
                        if param in type_hints:
                            hint = type_hints[param]
                            # Handle Optional[T] which is Union[T, None]
                            if hasattr(hint, '__origin__') and hint.__origin__ is typing.Union:
                                # Get the non-None type from Union
                                args = hint.__args__
                                for arg in args:
                                    if arg is not type(None):
                                        value = arg(value)
                                        break
                            else:
                                value = hint(value)
                    except (ValueError, KeyError, AttributeError):
                        # If type inference fails, keep as string
                        pass
                setattr(obj, param, value)

    def override_from_env(self, prefix: str = "FRAGCSA"):
        """
        Override from environment variables.

        Format: FRAGCSA__SECTION__PARAMETER=value
        Example: FRAGCSA__CSA__BANK_SIZE=200

        Args:
            prefix: Environment variable prefix (default: FRAGCSA)
        """
        for key, value in os.environ.items():
            if not key.startswith(f"{prefix}__"):
                continue

            # Parse: FRAGCSA__CSA__BANK_SIZE -> ['csa', 'bank_size']
            parts = key.replace(f"{prefix}__", "").lower().split("__")

            if len(parts) == 2:
                section, param = parts

                if hasattr(self, section):
                    section_obj = getattr(self, section)
                    if hasattr(section_obj, param):
                        # Type coercion based on current value
                        current = getattr(section_obj, param)
                        if current is not None:
                            casted = type(current)(value)
                            setattr(section_obj, param, casted)

    # ========================================================================
    # Validation and Serialization
    # ========================================================================

    def validate(self):
        """
        Validate all configuration sections.

        Raises:
            AssertionError: If any validation fails
        """
        self.csa.validate()
        self.solution.validate()
        self.data.validate()
        self.reaction_model.validate()
        
        # Set method_name based on spec_type
        if self.experiment.method_name is None:
            if self.solution.spec_type == "fragment_route":
                self.experiment.method_name = "rxnmol"
            elif self.solution.spec_type in ["smiles", "raw_smiles", "smiles_direct"]:
                self.experiment.method_name = "MolFinder"
            else:
                self.experiment.method_name = self.solution.spec_type

    def save(self, path: str):
        """
        Save configuration to YAML file.

        Args:
            path: Output file path
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return asdict(self)

    def __repr__(self) -> str:
        """String representation showing key parameters."""
        return (
            f"MasterConfig(\n"
            f"  solution_type={self.solution.spec_type},\n"
            f"  objective={self.objective.name},\n"
            f"  bank_size={self.csa.bank_size},\n"
            f"  max_iter={self.csa.max_iter},\n"
            f"  runs_dir={self.runs_dir},\n"
            f"  output_dir={self.output_dir}\n"
            f")"
        )
