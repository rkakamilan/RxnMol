### !/usr/bin/env python3
"""
Main orchestrator for Fragment-CSA optimization.

Usage:
    python run.py --config input.yaml
    python run.py --config input.yaml --objective zaleplon --bank-size 100
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import os
import random
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from rxnmol.core import MasterConfig, RunContext, CSAEngine
from rxnmol.solutions import FragmentRouteSpec, SmilesDirectSpec, ReactionMolSpec, ScaffoldHopRouteSpec
from rxnmol.solutions.reaction_models import load_reaction_model
from rxnmol.runtime import MetricsCollector, ArtifactStore
from rxnmol.utils import parse_dynamic_overrides, resolve_output_dir

# Initial logging setup (will be reconfigured after loading config)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run Fragment-CSA optimization')

    # Configuration
    parser.add_argument('--config', type=str, default='input.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--objective', type=str,
                        help='Objective function name (overrides config)')

    parser.add_argument('--output-dir', type=str,
                        help='Output directory (overrides config)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite output directory if it exists')

    return parser.parse_known_args()




def main():
    """Main orchestrator."""
    args, unknown_args = parse_args()

    logger.info("=" * 70)
    logger.info("Fragment-CSA Molecular Optimization")
    logger.info("=" * 70)

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    try:
        config = MasterConfig.from_yaml(args.config)
    except FileNotFoundError:
        logger.info(f"Config file not found, using defaults")
        config = MasterConfig()

    # Apply CLI overrides (Explicit)
    overrides = {}
    if args.objective:
        overrides['objective.name'] = args.objective
    if args.output_dir:
        overrides['output_dir'] = args.output_dir


    # Apply Dynamic Overrides
    if unknown_args:
        dynamic_overrides = parse_dynamic_overrides(unknown_args)
        if dynamic_overrides:
            logger.info(f"Found dynamic overrides: {dynamic_overrides}")
            overrides.update(dynamic_overrides)

    if overrides:
        logger.info(f"Applying CLI overrides: {overrides}")
        config.override_from_cli(overrides)

    # Validate configuration (this sets method_name based on spec_type)
    config.validate()

    # Resolve output directory (needs method_name to be set)
    config.output_dir = resolve_output_dir(config, args.overwrite)
    logger.info(f"Output directory set to: {config.output_dir}")

    # Reconfigure logging with level from config
    log_level_str = config.monitoring.log_level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.getLogger().setLevel(log_level)  # Update root logger

    # Silence noisy library loggers when DEBUG is enabled
    if log_level == logging.DEBUG:
        # Keep these at INFO to avoid spam
        logging.getLogger('torch_tensorrt').setLevel(logging.INFO)
        logging.getLogger('torch._dynamo').setLevel(logging.INFO)
        logging.getLogger('torch.fx').setLevel(logging.INFO)
        logging.getLogger('torch.jit').setLevel(logging.INFO)
        logging.getLogger('mol_opt.solutions.smiles_operators').setLevel(logging.INFO)

    logger.info(f"Log level set to: {log_level_str}")
    logger.info(f"Configuration validated:")
    logger.info(f"  Objective: {config.objective.name}")
    logger.info(f"  Bank size: {config.csa.bank_size}")
    logger.info(f"  Max iterations: {config.csa.max_iter}")
    logger.info(f"  Output dir: {config.output_dir}")

    # Create runtime context
    logger.info(f"Initializing runtime context...")
    # Use system time as seed if random_seed is None
    if config.runtime.random_seed is None:
        import time
        current_time = time.time()
        random_seed = int(current_time * 1000) % (2**32)
        logger.info(f"  Using random seed from system time: {random_seed} at time: {current_time}")
    else:
        random_seed = config.runtime.random_seed
        logger.info(f"  Using fixed random seed: {random_seed}")

    if config.runtime.device == "cuda" and config.runtime.require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available. "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                f"SLURM_JOB_GPUS={os.environ.get('SLURM_JOB_GPUS')} "
                "Check GPU allocation/environment or set runtime.require_cuda=false."
            )

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    rng = np.random.default_rng(random_seed)
    context = RunContext(config=config, rng=rng)

    # Setup monitoring
    logger.info(f"Setting up monitoring and persistence...")
    metrics = MetricsCollector()
    artifacts = ArtifactStore(output_dir=config.output_dir)
    context.metrics = metrics
    context.artifacts = artifacts

    # Save configuration and metadata
    artifacts.save_config(config, filename="config.yaml")
    logger.info(f"  Saved config to: {config.output_dir}/config.yaml")
    try:
        meta = {
            "model_tag": getattr(config.reaction_model, "tag", None),
            "model_provider": config.reaction_model.provider,
            "objective": config.objective.name,
            "spec_type": config.solution.spec_type,
            "min_fragments": config.solution.min_fragments,
            "max_fragments": config.solution.max_fragments,
            "repeat": config.experiment.repeat_id,
            "bank_size": config.csa.bank_size,
            "seed_size": config.csa.seed_size,
            "max_iter": config.csa.max_iter,
        }
        artifacts.save_meta(meta)
    except Exception as exc:
        logger.warning("Failed to save meta.json: %s", exc)

    # Create solution specification based on spec_type
    logger.info(f"Creating solution specification...")
    spec_type = config.solution.spec_type

    # Load reaction model once (shared across specs that need it)
    if spec_type in ["fragment_route", "reaction_mol", "scaffold_hop_route"]:
        logger.info("Loading reaction model adapter...")
        context.reaction_model = load_reaction_model(
            config.reaction_model,
            device=config.runtime.device,
        )
        logger.info("Reaction model loaded and attached to context.")

    if spec_type == "fragment_route":
        logger.info(f"  Type: Fragment-based synthesis routes")
        spec = FragmentRouteSpec(context)
        
    elif spec_type in ["raw_smiles", "smiles_direct", "smiles"]:
        spec = SmilesDirectSpec(context)
        logger.info(f"  Type: Direct SMILES optimization")
    elif spec_type == "reaction_mol":
        spec = ReactionMolSpec(context)
        logger.info(f"  Type: ReactionMol optimization")
    elif spec_type == "scaffold_hop_route":
        spec = ScaffoldHopRouteSpec(context)
        logger.info(f"  Type: Scaffold-hop (fixed warheads, variable core)")
    else:
        raise ValueError(f"Unsupported spec_type: {spec_type}")

    # Create CSA engine
    logger.info(f"Creating CSA engine...")
    engine = CSAEngine(spec, context)

    # Run optimization
    logger.info(f"" + "=" * 70)
    logger.info("Starting CSA Optimization")
    logger.info("=" * 70)

    try:
        best_candidate = engine.run()

        # Save final results
        logger.info(f"Saving final results...")
        artifacts.save_bank(engine.get_bank(), iteration=engine.iteration)
        artifacts.save_trace(best_candidate, iteration=engine.iteration, index=0)

        # Print summary
        logger.info(f"" + "=" * 70)
        logger.info("Optimization Complete!")
        logger.info("=" * 70)
        logger.info(f"Best Candidate Found:")
        logger.info(f"  Score: {best_candidate.objective_value:.6f}")
        logger.info(f"  SMILES: {best_candidate.smiles}")
        logger.info(f"  Genotype: {best_candidate.genotype}")

        logger.info(f"Statistics:")
        stats = engine.get_statistics()
        logger.info(f"  Total iterations: {stats['iteration']}")
        logger.info(f"  Total evaluations: {stats['n_evaluations']}")
        logger.info(f"  Improvements: {stats['n_improvements']}")

        logger.info(f"Results saved to: {config.output_dir}/")
        logger.info(f"  Banks: bank_{engine.iteration}.txt")
        logger.info(f"  Traces: cycle_{engine.iteration}_traces/")
        logger.info(f"  Config: config.yaml")

        return 0

    except KeyboardInterrupt:
        logger.info(f"Optimization interrupted by user")
        logger.info(f"Saving partial results...")
        if engine.best_candidate:
            artifacts.save_bank(engine.get_bank(), iteration=engine.iteration)
        logger.info(f"Partial results saved to: {config.output_dir}/")
        return 1

    except Exception as e:
        logger.error(f"Optimization failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
