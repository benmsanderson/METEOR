"""
Ensemble processing utilities for METEOR-impacts
===============================================

This module provides utilities for applying impact calculators to
climate data ensembles and managing the resulting impact ensembles.
"""

from typing import List, Optional, Union

import xarray as xr

from .impacts_core import ImpactCalculator, ImpactEnsemble, ImpactResult


def apply_impact_calculator(
    calculator: ImpactCalculator,
    climate_data: Union[xr.DataArray, List[xr.DataArray]],
    ensemble_dim: Optional[str] = None,
) -> Union[ImpactResult, List[ImpactResult]]:
    """
    Apply an impact calculator to climate data.

    This function handles both single realizations and ensembles,
    providing a unified interface for impact calculations.

    Parameters
    ----------
    calculator : ImpactCalculator
        The impact calculator to apply
    climate_data : Union[xr.DataArray, List[xr.DataArray]]
        Either a single DataArray or list of DataArrays (ensemble)
    ensemble_dim : str, optional
        If climate_data has an ensemble dimension, specify its name
        to process each member separately

    Returns
    -------
    Union[ImpactResult, List[ImpactResult]]
        ImpactResult (single realization) or List[ImpactResult] (ensemble)

    Raises
    ------
    ValueError
        If ensemble dimension not found in climate_data
    TypeError
        If climate_data is not the correct type
    """
    # Handle single DataArray with ensemble dimension
    if isinstance(climate_data, xr.DataArray) and ensemble_dim is not None:
        if ensemble_dim not in climate_data.dims:
            raise ValueError(
                f"Ensemble dimension '{ensemble_dim}' not found in climate_data"
            )

        ensemble_members = [
            climate_data.isel({ensemble_dim: i})
            for i in range(climate_data.sizes[ensemble_dim])
        ]
        return _apply_to_ensemble(calculator, ensemble_members)

    # Handle list of DataArrays (ensemble)
    if isinstance(climate_data, list):
        return _apply_to_ensemble(calculator, climate_data)

    # Handle single DataArray
    if isinstance(climate_data, xr.DataArray):
        return calculator.calculate(climate_data)

    raise TypeError("climate_data must be an xarray.DataArray or list of DataArrays")


def _apply_to_ensemble(
    calculator: ImpactCalculator, ensemble_members: List[xr.DataArray]
) -> List[ImpactResult]:
    """
    Apply calculator to each member of an ensemble.

    Parameters
    ----------
    calculator : ImpactCalculator
        The impact calculator to apply
    ensemble_members : List[xr.DataArray]
        List of climate data arrays

    Returns
    -------
    List[ImpactResult]
        List of ImpactResult objects

    Raises
    ------
    RuntimeError
        If calculation fails for any ensemble member
    """
    results = []

    for i, member in enumerate(ensemble_members):
        try:
            result = calculator.calculate(member)
            # Add ensemble member info to metadata
            result.metadata.update(
                {"ensemble_member": i, "ensemble_size": len(ensemble_members)}
            )
            results.append(result)
        except Exception as e:
            raise RuntimeError(
                f"Failed to calculate impacts for ensemble member {i}: {e}"
            ) from e

    return results


def create_impact_ensemble(
    calculator: ImpactCalculator, climate_ensemble: List[xr.DataArray]
) -> ImpactEnsemble:
    """
    Create an ImpactEnsemble from a climate ensemble.

    This is a convenience function that creates an ImpactEnsemble object
    and immediately calculates impacts for all ensemble members.

    Parameters
    ----------
    calculator : ImpactCalculator
        The impact calculator to apply
    climate_ensemble : List[xr.DataArray]
        List of climate data arrays (ensemble members)

    Returns
    -------
    ImpactEnsemble
        ImpactEnsemble with calculated results
    """
    ensemble = ImpactEnsemble(calculator)
    ensemble.calculate_ensemble(climate_ensemble)
    return ensemble


def ensemble_statistics(
    impact_results: List[ImpactResult],
    variable: str,
    statistics: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Calculate ensemble statistics for a specific impact variable.

    Parameters
    ----------
    impact_results : List[ImpactResult]
        List of ImpactResult objects from ensemble calculation
    variable : str
        Name of the impact variable to analyze
    statistics : List[str], optional
        List of statistics to calculate ('mean', 'std', 'min', 'max', 'percentile_XX')

    Returns
    -------
    xr.Dataset
        xarray Dataset containing the requested statistics

    Raises
    ------
    ValueError
        If no impact results provided or variable not found
    """
    if statistics is None:
        statistics = ["mean", "std", "min", "max"]

    if not impact_results:
        raise ValueError("No impact results provided")

    if variable not in impact_results[0]:
        raise ValueError(f"Variable '{variable}' not found in impact results")

    # Collect data from all ensemble members
    ensemble_data = [result[variable] for result in impact_results]
    ensemble_array = xr.concat(ensemble_data, dim="ensemble_member")

    # Calculate requested statistics
    stats_data = {}

    for stat in statistics:
        if stat == "mean":
            stats_data["ensemble_mean"] = ensemble_array.mean(dim="ensemble_member")
        elif stat == "std":
            stats_data["ensemble_std"] = ensemble_array.std(dim="ensemble_member")
        elif stat == "min":
            stats_data["ensemble_min"] = ensemble_array.min(dim="ensemble_member")
        elif stat == "max":
            stats_data["ensemble_max"] = ensemble_array.max(dim="ensemble_member")
        elif stat.startswith("percentile_"):
            # Extract percentile from string like 'percentile_95'
            try:
                percentile = float(stat.split("_")[1])
                quantile = percentile / 100.0
                stats_data[f"ensemble_p{int(percentile)}"] = ensemble_array.quantile(
                    quantile, dim="ensemble_member"
                ).drop_vars(
                    "quantile", errors="ignore"
                )  # Remove quantile coordinate
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid percentile specification: {stat}") from exc
        else:
            raise ValueError(f"Unknown statistic: {stat}")

    # Create dataset with metadata
    dataset = xr.Dataset(stats_data)
    dataset.attrs.update(
        {
            "variable": variable,
            "ensemble_size": len(impact_results),
            "statistics": statistics,
            "calculator": impact_results[0].calculator_name,
        }
    )

    return dataset
