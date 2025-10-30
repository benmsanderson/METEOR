"""
Degree Days Calculator for METEOR-impacts
=========================================

This module implements calculation of Heating Degree Days (HDD) and Cooling Degree Days (CDD)
from monthly temperature data using the methodology from Isaac and van Vuuren (2009).

References:
    Isaac, M., & van Vuuren, D. P. (2009). Modeling global residential sector energy demand
    for heating and air conditioning in the context of climate change. Energy Policy, 37(2), 507-521.

    Erbs, D. G., Klein, S. A., & Duffie, J. A. (1983). Estimation of the diffuse radiation
    fraction for hourly, daily and monthly-average global radiation. Solar Energy, 28(4), 293-302.
"""

import warnings
from typing import Tuple, Optional
import numpy as np
import xarray as xr

from ..base import ImpactCalculator, ImpactResult


class DegreeDaysCalculator(ImpactCalculator):
    """
    Calculator for Heating Degree Days (HDD) and Cooling Degree Days (CDD).

    This calculator implements the methodology described in Isaac and van Vuuren (2009),
    which uses a correction formula from Erbs et al. (1983) to account for
    daily temperature variations within each month.

    The calculation assumes that daily temperatures follow a normal distribution
    around the monthly mean, and uses an analytical approach to estimate the
    degree days without requiring daily temperature data.
    """

    def __init__(
        self,
        base_temperature: float = 18.0,
        sigma_m_c1: float = 1.45,
        sigma_m_c2: float = 0.29,
        sigma_m_c3: float = 0.664,
        a_val_c1: float = 1.698,
        name: str = "DegreeDays",
    ):
        """
        Initialize the Degree Days calculator.

        Args:
            base_temperature: Base temperature in Celsius for degree day calculation.
                            Defaults to 18.0°C (common for residential heating/cooling).
            sigma_m_c1: Coefficient c1 for σ_m calculation (default: 1.45)
            sigma_m_c2: Coefficient c2 for σ_m calculation (default: 0.29)
            sigma_m_c3: Coefficient c3 for σ_m calculation (default: 0.664)
            a_val_c1: Coefficient for 'a' value calculation (default: 1.698)
            name: Human-readable name for this calculator
        """
        super().__init__(name)
        self.base_temperature = base_temperature
        self.sigma_m_c1 = sigma_m_c1
        self.sigma_m_c2 = sigma_m_c2
        self.sigma_m_c3 = sigma_m_c3
        self.a_val_c1 = a_val_c1

    def validate_input(self, climate_data: xr.DataArray) -> None:
        """
        Validate that input data is suitable for degree days calculation.

        Args:
            climate_data: Input temperature data to validate

        Raises:
            ValueError: If input data is not suitable
        """
        if not isinstance(climate_data, xr.DataArray):
            raise ValueError("Input must be an xarray.DataArray")

        if "month" not in climate_data.dims:
            raise ValueError("Input DataArray must have a dimension named 'month'")

        # Check for reasonable temperature values (assuming Kelvin or Celsius)
        temp_min = float(climate_data.min())
        temp_max = float(climate_data.max())

        # Heuristic check: if all values are > 200, assume Kelvin; if < 100, assume Celsius
        if temp_min > 200:
            # Likely Kelvin - check for reasonable range
            if temp_min < 150 or temp_max > 400:
                warnings.warn(
                    f"Temperature values seem unusual (range: {temp_min:.1f} to {temp_max:.1f}K). "
                    "Please verify the data is correct."
                )
        elif temp_max < 100:
            # Likely Celsius - check for reasonable range
            if temp_min < -100 or temp_max > 80:
                warnings.warn(
                    f"Temperature values seem unusual (range: {temp_min:.1f} to {temp_max:.1f}°C). "
                    "Please verify the data is correct."
                )
        else:
            # Mixed range - could be Celsius with hot values or unusual units
            if temp_max > 100:
                warnings.warn(
                    f"Temperature values seem unusual (range: {temp_min:.1f} to {temp_max:.1f}). "
                    "Very high values detected - please verify units are correct."
                )

    def calculate(self, climate_data: xr.DataArray) -> ImpactResult:
        """
        Calculate Heating and Cooling Degree Days from monthly temperature data.

        Args:
            climate_data: xarray DataArray with monthly mean temperatures.
                         Must have a 'month' dimension. Temperature should be in
                         the same units as base_temperature (typically Celsius).

        Returns:
            ImpactResult containing:
                - 'monthly_hdd': Monthly heating degree days
                - 'monthly_cdd': Monthly cooling degree days
                - 'annual_hdd': Annual total heating degree days
                - 'annual_cdd': Annual total cooling degree days
        """
        # Validate input
        self.validate_input(climate_data)

        # Calculate degree days
        monthly_hdd, monthly_cdd, annual_hdd, annual_cdd = self._calculate_degree_days(
            climate_data
        )

        # Create result
        result_data = {
            "monthly_hdd": monthly_hdd,
            "monthly_cdd": monthly_cdd,
            "annual_hdd": annual_hdd,
            "annual_cdd": annual_cdd,
        }

        metadata = {
            "base_temperature": self.base_temperature,
            "method": "Isaac and van Vuuren (2009)",
            "reference": "Isaac, M., & van Vuuren, D. P. (2009). Energy Policy, 37(2), 507-521.",
            "parameters": {
                "sigma_m_c1": self.sigma_m_c1,
                "sigma_m_c2": self.sigma_m_c2,
                "sigma_m_c3": self.sigma_m_c3,
                "a_val_c1": self.a_val_c1,
            },
        }

        return ImpactResult(result_data, metadata, self.name)

    def _calculate_degree_days(
        self, monthly_mean_temps: xr.DataArray
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Core calculation of degree days using the Isaac and van Vuuren (2009) method.

        This method implements Equation 5 from the paper, which accounts for the
        distribution of daily temperatures within each month.

        Args:
            monthly_mean_temps: Monthly mean temperatures

        Returns:
            Tuple of (monthly_hdd, monthly_cdd, annual_hdd, annual_cdd)
        """
        # Group by calendar month using the modulo operator to get climatology
        month_grouper = monthly_mean_temps["month"] % 12
        monthly_clim = monthly_mean_temps.groupby(month_grouper).mean(dim="month")

        # Warn if the climatology is built from less than a full year's data
        if len(monthly_clim.month) < 12:
            warnings.warn(
                "Input data covers less than a full 12-month cycle. "
                "The standard deviation of monthly temperatures (sigma_y) is calculated "
                "from an incomplete year, which may affect the accuracy of the "
                "degree-day estimation.",
                UserWarning,
            )

        # Calculate interannual standard deviation
        sigma_y = monthly_clim.std(dim="month")

        # Get the number of days in each month (assuming standard non-leap year)
        days_in_month_map = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        calendar_month_indices = monthly_mean_temps["month"] % 12
        days_in_month = xr.DataArray(
            days_in_month_map[calendar_month_indices.values],
            coords={"month": monthly_mean_temps["month"]},
            dims=["month"],
        )

        # --- Vectorized implementation of Equation 5 ---

        # σ_m = c1 - c2*T_a + c3*σ_y
        sigma_m = (
            self.sigma_m_c1
            - (self.sigma_m_c2 * monthly_mean_temps)
            + (self.sigma_m_c3 * sigma_y)
        )

        # a = c1_a * sqrt(D_m)
        a_val = self.a_val_c1 * np.sqrt(days_in_month)

        # h = |T_base - T_a| / (σ_m * sqrt(D_m))
        h_val = np.abs(self.base_temperature - monthly_mean_temps) / (
            sigma_m * np.sqrt(days_in_month)
        )

        # Core of the equation, handling potential overflow
        # For large x, log(exp(-x) + exp(x)) -> log(exp(x)) -> x
        # The term then becomes h/2 + (a*h)/(2a) = h
        x = a_val * h_val
        term_in_brackets = xr.where(
            x > 100, h_val, (h_val / 2) + (np.log(np.exp(-x) + np.exp(x))) / (2 * a_val)
        )

        # DD_m = σ_m * (D_m)^1.5 * [term]
        degree_days_m = sigma_m * (days_in_month**1.5) * term_in_brackets

        # Ensure non-negative and finite values
        degree_days_m = xr.where(
            (degree_days_m < 0) | (~np.isfinite(degree_days_m)), 0, degree_days_m
        )

        # Assign to HDD or CDD based on temperature relative to the base
        monthly_hdd = xr.where(
            monthly_mean_temps < self.base_temperature, degree_days_m, 0
        )
        monthly_cdd = xr.where(
            monthly_mean_temps > self.base_temperature, degree_days_m, 0
        )

        # Set names and attributes
        monthly_hdd.name = "monthly_hdd"
        monthly_cdd.name = "monthly_cdd"
        monthly_hdd.attrs.update(
            {
                "long_name": "Monthly Heating Degree Days",
                "units": "degree-days",
                "base_temperature": f"{self.base_temperature}°C",
            }
        )
        monthly_cdd.attrs.update(
            {
                "long_name": "Monthly Cooling Degree Days",
                "units": "degree-days",
                "base_temperature": f"{self.base_temperature}°C",
            }
        )

        # Calculate annual totals by grouping by year
        annual_grouper = monthly_mean_temps["month"] // 12
        annual_hdd = monthly_hdd.groupby(annual_grouper).sum(dim="month")
        annual_cdd = monthly_cdd.groupby(annual_grouper).sum(dim="month")

        # The groupby operation creates dimension with grouper name
        # Rename the grouped dimension to 'year' if it exists
        if "group" in annual_hdd.dims:
            annual_hdd = annual_hdd.rename({"group": "year"})
            annual_cdd = annual_cdd.rename({"group": "year"})
        elif annual_grouper.name in annual_hdd.dims and annual_grouper.name != "year":
            annual_hdd = annual_hdd.rename({annual_grouper.name: "year"})
            annual_cdd = annual_cdd.rename({annual_grouper.name: "year"})

        annual_hdd.name = "annual_hdd"
        annual_cdd.name = "annual_cdd"
        annual_hdd.attrs.update(
            {
                "long_name": "Annual Heating Degree Days",
                "units": "degree-days",
                "base_temperature": f"{self.base_temperature}°C",
            }
        )
        annual_cdd.attrs.update(
            {
                "long_name": "Annual Cooling Degree Days",
                "units": "degree-days",
                "base_temperature": f"{self.base_temperature}°C",
            }
        )

        return monthly_hdd, monthly_cdd, annual_hdd, annual_cdd
