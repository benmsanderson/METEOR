"""
Base classes for METEOR-impacts
===============================

This module defines the abstract base classes and core data structures
for the METEOR-impacts system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import xarray as xr


class ImpactResult:
    """
    Container for impact calculation results.

    This class standardizes the output from impact calculators and provides
    methods for accessing and manipulating the results.

    Attributes
    ----------
        data (Dict[str, xr.DataArray]): Dictionary of impact variables
        metadata (Dict[str, Any]): Metadata about the calculation
        calculator_name (str): Name of the calculator that produced this result
    """

    def __init__(
        self,
        data: Dict[str, xr.DataArray],
        metadata: Optional[Dict[str, Any]] = None,
        calculator_name: str = "unknown",
    ):
        """
        Initialize an ImpactResult.

        Parameters
        ----------
        data : Dict[str, xr.DataArray]
            Dictionary mapping variable names to xarray DataArrays
        metadata : Dict[str, Any], optional
            Optional metadata about the calculation
        calculator_name : str, default "unknown"
            Name of the calculator that produced this result
        """
        self.data = data
        self.metadata = metadata or {}
        self.calculator_name = calculator_name

    def __getitem__(self, key: str) -> xr.DataArray:
        """
        Access impact variables by name.

        Parameters
        ----------
        key : str
            Variable name to access

        Returns
        -------
        xr.DataArray
            The requested impact variable
        """
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        """
        Check if a variable exists in the result.

        Parameters
        ----------
        key : str
            Variable name to check

        Returns
        -------
        bool
            True if variable exists, False otherwise
        """
        return key in self.data

    def keys(self):
        """
        Return the names of available impact variables.

        Returns
        -------
        dict_keys
            Names of all available impact variables
        """
        return self.data.keys()

    def to_dataset(self) -> xr.Dataset:
        """
        Convert the result to an xarray Dataset.

        Returns
        -------
        xr.Dataset
            Dataset containing all impact variables with metadata as attributes
        """
        return xr.Dataset(self.data, attrs=self.metadata)


class ImpactCalculator(ABC):
    """
    Abstract base class for all impact calculators.

    This class defines the interface that all impact calculators must implement.
    Subclasses should override the calculate() method to perform specific
    impact calculations.
    """

    def __init__(self, name: str):
        """
        Initialize the calculator.

        Parameters
        ----------
        name : str
            Human-readable name for this calculator
        """
        self.name = name

    @abstractmethod
    def calculate(self, climate_data: xr.DataArray) -> ImpactResult:
        """
        Calculate impacts from climate data.

        Parameters
        ----------
        climate_data : xr.DataArray
            xarray DataArray containing climate variables

        Returns
        -------
        ImpactResult
            Object containing calculated impacts

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement calculate()")

    @abstractmethod
    def validate_input(self, climate_data: xr.DataArray) -> None:
        """
        Validate that input data is suitable for this calculator.

        Parameters
        ----------
        climate_data : xr.DataArray
            Input climate data to validate

        Raises
        ------
        ValueError
            If input data is not suitable
        """
        raise NotImplementedError("Subclasses must implement validate_input()")

    def __str__(self) -> str:
        """
        Return string representation of the calculator.

        Returns
        -------
        str
            String representation showing class name and calculator name
        """
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        """
        Return detailed string representation of the calculator.

        Returns
        -------
        str
            Detailed string representation
        """
        return self.__str__()


class ImpactEnsemble:
    """
    Container and processor for ensemble impact calculations.

    This class handles the application of impact calculators to ensembles
    of climate data, providing methods for ensemble statistics and analysis.
    """

    def __init__(self, calculator: ImpactCalculator):
        """
        Initialize the ensemble processor.

        Parameters
        ----------
        calculator : ImpactCalculator
            The impact calculator to apply to ensemble members
        """
        self.calculator = calculator
        self.results: List[ImpactResult] = []

    def calculate_ensemble(
        self, climate_ensemble: List[xr.DataArray]
    ) -> List[ImpactResult]:
        """
        Apply the impact calculator to each member of a climate ensemble.

        Parameters
        ----------
        climate_ensemble : List[xr.DataArray]
            List of climate data arrays (ensemble members)

        Returns
        -------
        List[ImpactResult]
            List of ImpactResult objects, one per ensemble member

        Raises
        ------
        RuntimeError
            If calculation fails for any ensemble member
        """
        self.results = []

        for i, member in enumerate(climate_ensemble):
            try:
                result = self.calculator.calculate(member)
                # Add ensemble member info to metadata
                result.metadata.update(
                    {"ensemble_member": i, "ensemble_size": len(climate_ensemble)}
                )
                self.results.append(result)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to calculate impacts for ensemble member {i}: {e}"
                ) from e

        return self.results

    def ensemble_mean(self, variable: str) -> xr.DataArray:
        """
        Calculate ensemble mean for a specific impact variable.

        Parameters
        ----------
        variable : str
            Name of the impact variable

        Returns
        -------
        xr.DataArray
            xarray DataArray containing ensemble mean

        Raises
        ------
        ValueError
            If no ensemble results are available or variable not found
        """
        if not self.results:
            raise ValueError(
                "No ensemble results available. Run calculate_ensemble() first."
            )

        if variable not in self.results[0]:
            raise ValueError(f"Variable '{variable}' not found in results")

        ensemble_data = [result[variable] for result in self.results]
        ensemble_array = xr.concat(ensemble_data, dim="ensemble_member")
        return ensemble_array.mean(dim="ensemble_member")

    def ensemble_std(self, variable: str) -> xr.DataArray:
        """
        Calculate ensemble standard deviation for a specific impact variable.

        Parameters
        ----------
        variable : str
            Name of the impact variable

        Returns
        -------
        xr.DataArray
            xarray DataArray containing ensemble standard deviation

        Raises
        ------
        ValueError
            If no ensemble results are available or variable not found
        """
        if not self.results:
            raise ValueError(
                "No ensemble results available. Run calculate_ensemble() first."
            )

        if variable not in self.results[0]:
            raise ValueError(f"Variable '{variable}' not found in results")

        ensemble_data = [result[variable] for result in self.results]
        ensemble_array = xr.concat(ensemble_data, dim="ensemble_member")
        return ensemble_array.std(dim="ensemble_member")

    def ensemble_percentiles(
        self, variable: str, percentiles: Union[float, List[float]]
    ) -> Union[xr.DataArray, Dict[float, xr.DataArray]]:
        """
        Calculate ensemble percentiles for a specific impact variable.

        Parameters
        ----------
        variable : str
            Name of the impact variable
        percentiles : Union[float, List[float]]
            Percentile(s) to calculate (0-100)

        Returns
        -------
        Union[xr.DataArray, Dict[float, xr.DataArray]]
            xarray DataArray (single percentile) or dict of DataArrays (multiple)

        Raises
        ------
        ValueError
            If no ensemble results are available or variable not found
        """
        if not self.results:
            raise ValueError(
                "No ensemble results available. Run calculate_ensemble() first."
            )

        if variable not in self.results[0]:
            raise ValueError(f"Variable '{variable}' not found in results")

        ensemble_data = [result[variable] for result in self.results]
        ensemble_array = xr.concat(ensemble_data, dim="ensemble_member")

        if isinstance(percentiles, (int, float)):
            return ensemble_array.quantile(percentiles / 100, dim="ensemble_member")

        return {
            p: ensemble_array.quantile(p / 100, dim="ensemble_member")
            for p in percentiles
        }

    def to_dataset(self, variables: Optional[List[str]] = None) -> xr.Dataset:
        """
        Convert ensemble results to a single Dataset with ensemble dimension.

        Parameters
        ----------
        variables : List[str], optional
            List of variables to include (None for all)

        Returns
        -------
        xr.Dataset
            xarray Dataset with ensemble dimension

        Raises
        ------
        ValueError
            If no ensemble results are available or specified variable not found
        """
        if not self.results:
            raise ValueError(
                "No ensemble results available. Run calculate_ensemble() first."
            )

        if variables is None:
            variables = list(self.results[0].keys())

        data_vars = {}

        for var in variables:
            if var not in self.results[0]:
                raise ValueError(f"Variable '{var}' not found in results")

            ensemble_data = [result[var] for result in self.results]
            data_vars[var] = xr.concat(ensemble_data, dim="ensemble_member")

        return xr.Dataset(data_vars)
