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

from .impacts_core import ImpactCalculator, ImpactEnsemble, ImpactResult
from .calculators.degree_days import DegreeDaysCalculator
from .ensemble import (
    apply_impact_calculator,
    create_impact_ensemble,
    ensemble_statistics,
)

__all__ = [
    "ImpactCalculator",
    "ImpactResult",
    "ImpactEnsemble",
    "DegreeDaysCalculator",
    "apply_impact_calculator",
    "create_impact_ensemble",
    "ensemble_statistics",
]
