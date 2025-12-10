"""
Core components for Fragment-CSA optimization.

This module contains the fundamental building blocks:
- Configuration management
- Data models (Candidate, RunContext)
- CSA optimization engine
"""

from .config import (
    CSAConfig,
    SolutionConfig,
    DataConfig,
    OperatorConfig,
    ObjectiveConfig,
    MonitoringConfig,
    PersistenceConfig,
    RuntimeConfig,
    MasterConfig,
)

from .data_models import (
    Candidate,
    RunContext,
)

from .engine import CSAEngine

__all__ = [
    # Config classes
    "CSAConfig",
    "SolutionConfig",
    "DataConfig",
    "OperatorConfig",
    "ObjectiveConfig",
    "MonitoringConfig",
    "PersistenceConfig",
    "RuntimeConfig",
    "MasterConfig",
    # Data models
    "Candidate",
    "RunContext",
    # Engine
    "CSAEngine",
]
