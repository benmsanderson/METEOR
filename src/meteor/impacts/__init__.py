"""
METEOR-impacts: Post-processing layer for climate impact calculations
====================================================================

This module provides tools for calculating climate impacts from METEOR outputs,
including degree days, heat stress indices, and other derived metrics.

The impacts layer is designed to work with both:
- Annual climatologies from METEOR core
- Monthly realizations from METEOR-noise

Example usage:
    >>> from meteor.impacts import DegreeDaysCalculator, ImpactEnsemble
    >>> calculator = DegreeDaysCalculator(base_temperature=18.0)
    >>> results = calculator.calculate(climate_data)
"""

from .calculators.degree_days import DegreeDaysCalculator
from .ensemble import (
    apply_impact_calculator,
    create_impact_ensemble,
    ensemble_statistics,
)
from .ggcm import GgcmDownloader, load_agmerra_baseline
from .impacts_core import ImpactCalculator, ImpactEnsemble, ImpactResult

__all__ = [
    "ImpactCalculator",
    "ImpactResult",
    "ImpactEnsemble",
    "DegreeDaysCalculator",
    "GgcmDownloader",
    "load_agmerra_baseline",
    "apply_impact_calculator",
    "create_impact_ensemble",
    "ensemble_statistics",
]
