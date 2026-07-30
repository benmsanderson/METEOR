"""
Ensemble Output Container

This module provides containers for METEOR ensemble outputs, including
climate variables, spatial aggregations, and impact metrics.
"""

from pathlib import Path

import numpy as np
import xarray as xr


class VariableOutput:
    """
    Container for a single variable's ensemble outputs.

    Attributes
    ----------
    variable : str
        Variable name (e.g., 'tas', 'pr')
    timeseries : dict
        Dictionary of time series outputs {aggregation_name: xr.DataArray}
    gridded : dict
        Dictionary of gridded outputs {time_slice: xr.DataArray}
    impacts : dict
        Dictionary of impact metrics {impact_name: {aggregation: xr.DataArray}}
    metadata : dict
        Metadata about the outputs
    """

    def __init__(self, variable_name, metadata=None):
        self.variable = variable_name
        self.timeseries = {}
        self.gridded = {}
        self.impacts = {}
        self.metadata = metadata or {}

    def list_timeseries(self):
        """List available time series aggregations."""
        return list(self.timeseries.keys())

    def list_gridded(self):
        """List available gridded time slices."""
        return list(self.gridded.keys())

    def list_impacts(self):
        """List available impact metrics."""
        return list(self.impacts.keys())

    def __repr__(self):
        """Return detailed representation of VariableOutput."""
        lines = [f"VariableOutput('{self.variable}')"]
        if self.timeseries:
            lines.append(f"  Timeseries: {len(self.timeseries)} aggregations")
        if self.gridded:
            lines.append(f"  Gridded: {len(self.gridded)} time slices")
        if self.impacts:
            lines.append(f"  Impacts: {list(self.impacts.keys())}")
        return "\n".join(lines)


class EnsembleOutput:
    """
    Container for complete METEOR ensemble outputs.

    Provides convenient access to generated climate variables, spatial
    aggregations, and derived impact metrics.

    Parameters
    ----------
    results : dict
        Dictionary mapping variable names to VariableOutput objects
    metadata : dict, optional
        Metadata about the ensemble (scenario, years, n_realizations, etc.)

    Examples
    --------
    >>> ensemble = emulator.generate(...)
    >>>
    >>> # Access time series
    >>> tas_global = ensemble["tas"].timeseries["global"]
    >>> pr_regional = ensemble["pr"].timeseries["regional:EAS"]
    >>>
    >>> # Access gridded outputs
    >>> tas_2050 = ensemble["tas"].gridded["annual"][2050]
    >>>
    >>> # Access impact metrics
    >>> hdd = ensemble["tas"].impacts["hdd"]["point:59.9,10.8"]
    """

    def __init__(self, results=None, metadata=None):
        self.variables = results or {}
        self.metadata = metadata or {}
        # Top-level dict for crop yield impacts, which depend on both tas and pr.
        # Structure: {crop_name: {agg_key: np.ndarray (n_years,)}}
        self.crop_impacts = {}

    def __getitem__(self, key):
        """Access variable outputs."""
        return self.variables[key]

    def __contains__(self, key):
        """Check if variable exists."""
        return key in self.variables

    def list_variables(self):
        """List available variables."""
        return list(self.variables.keys())

    def to_netcdf(self, path, include_impacts=True):
        """
        Save ensemble outputs to netCDF file.

        Parameters
        ----------
        path : str
            Output file path
        include_impacts : bool, optional
            Whether to include impact metrics (default True)
        """
        datasets = {}

        for var_name, var_data in self.variables.items():
            # Create dataset for this variable
            ds = xr.Dataset()

            # Do gridded variables first, because the timeseries variables just return arrays without start/end year information
            for grid_name, grid_array in var_data.gridded.items():
                safe_name = str(grid_name).replace("-", "_").replace(":", "_")
                if isinstance(grid_array, dict) and grid_name == "annual":
                    years = sorted(grid_array.keys())
                    stacked = xr.concat(
                        [grid_array[y] for y in years], dim="year"
                    ).assign_coords(year=years)
                    ds[f"{var_name}_grid_{safe_name}"] = stacked

                elif isinstance(grid_array, dict) and grid_name == "monthly":
                    years = sorted(grid_array.keys())
                    slices = []
                    for y in years:
                        da = grid_array[y]
                        yyyymm = [y * 100 + m for m in range(1, 13)]
                        slices.append(da.assign_coords(month=yyyymm))
                    stacked = xr.concat(slices, dim="month")
                    stacked = stacked.transpose("month", "realization", "lat", "lon")
                    ds[f"{var_name}_grid_{safe_name}"] = stacked
                elif isinstance(grid_array, dict) and grid_name == "climatology":
                    years_keys = sorted(grid_array.keys())
                    years = [
                        np.mean(
                            (int(year_str.split("-")[0]), int(year_str.split("-")[1]))
                        )
                        for year_str in years_keys
                    ]
                    stacked = xr.concat(
                        [grid_array[y] for y in years_keys], dim="year"
                    ).assign_coords(year=years)
                    ds[f"{var_name}_grid_{safe_name}"] = stacked
                else:
                    ds[f"{var_name}_grid_{safe_name}"] = grid_array

            # Add time series
            for ts_name, ts_array in var_data.timeseries.items():
                safe_name = ts_name.replace(":", "_").replace(".", "p")
                if ts_array.ndim == 1:
                    dims = ("month",)
                elif ts_array.ndim == 2:
                    dims = ("realization", "month")
                else:
                    raise ValueError(
                        f"Unexpected dimensions for {ts_name}: {ts_array.shape}"
                    )
                ds[f"{var_name}_{safe_name}"] = (dims, ts_array.data)

            # Add impacts if requested
            if include_impacts:
                for impact_name, impact_dict in var_data.impacts.items():
                    for agg_name, impact_array in impact_dict.items():
                        if isinstance(impact_array, xr.DataArray):
                            dimensions = impact_array.dims
                            data = impact_array.data
                        elif isinstance(impact_array, np.ndarray):
                            if impact_array.shape[0] == 1:
                                dimensions = ("year",)
                                data = impact_array[0]
                            else:
                                dimensions = ("month", "year")
                                data = impact_array
                        else:
                            raise ValueError(
                                f"Unexpected type for impact array: {type(impact_array)}"
                            )
                        safe_agg = agg_name.replace(":", "_").replace(".", "p")
                        ds[f"{var_name}_{impact_name}_{safe_agg}"] = (
                            dimensions,
                            data,
                        )

            datasets[var_name] = ds

        # Combine all variables into one dataset
        combined = xr.merge(list(datasets.values()))

        # Add metadata as attributes
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float)):
                combined.attrs[key] = value

        # Save to netCDF
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        combined.to_netcdf(path)
        print(f"✅ Saved ensemble to {path}")

    def __repr__(self):
        """Return detailed string representation of EnsembleOutput."""
        lines = ["EnsembleOutput"]
        lines.append(f"  Variables: {list(self.variables.keys())}")
        if "scenario" in self.metadata:
            lines.append(f"  Scenario: {self.metadata['scenario']}")
        if "n_realizations" in self.metadata:
            lines.append(f"  Realizations: {self.metadata['n_realizations']}")
        if "year_range" in self.metadata:
            lines.append(f"  Years: {self.metadata['year_range']}")
        return "\n".join(lines)
