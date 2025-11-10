"""
Module for generating climate noise realizations using PCA and VARX modeling.

This module implements the methodology for:
1. Temperature-dependent seasonal cycle extraction using modulated harmonic regression
        xr.DataArray or list of xr.DataArray
            Generated climate realizations. If noise_only=True, returns the
            stochastic component plus temperature-modulated harmonics (but without
            direct temperature trends) that can be added to other predictions.PCA-based spatial decomposition of anomalies
3. VARX modeling of principal components
4. Stochastic simulation of new climate realizations
5. Noise-only generation for combining with annual climate projections
"""

import os
import pickle  # nosec - Used for trusted model serialization only
import warnings

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import VAR

from .prpatt import global_mean


class MeteorNoiseGenerator:
    """
    Climate noise generator using PCA and VARX modeling.

    This class implements a method to generate stochastic climate realizations
    by separating deterministic (temperature-dependent seasonal cycle) and
    stochastic (internal variability) components from monthly climate data.

    Can generate either full climate realizations or noise-only components
    that can be added to annual climate projections from METEOR.

    Attributes
    ----------
    n_modes : int
        Number of PCA modes to retain
    lag_order : int
        Lag order for VARX model
    seasonal_model : sklearn.LinearRegression
        Fitted seasonal cycle regression model
    pca : sklearn.decomposition.PCA
        Fitted PCA model for anomalies
    varx_results : statsmodels VAR results
        Fitted VARX model results
    coords : dict
        Coordinate information for spatial reconstruction
    fitted : bool
        Whether the model has been fitted
    """

    def __init__(self, n_modes=10, lag_order=2):
        """
        Initialize the noise generator.

        Parameters
        ----------
        n_modes : int, default 10
            Number of PCA modes to retain
        lag_order : int, default 2
            Lag order for VARX model
        """
        self.n_modes = n_modes
        self.lag_order = lag_order
        self.seasonal_model = None
        self.pca = None
        self.varx_results = None
        self.coords = None
        self.fitted = False

        # In-memory cache for regional EOF projections (model-invariant)
        self._regional_eof_projections = {}

    def _create_harmonic_features(self, time, t_glob):
        """
        Create harmonic features for seasonal cycle modeling.

        Parameters
        ----------
        time : array-like
            Time indices (in months)
        t_glob : array-like
            Global mean temperature time series

        Returns
        -------
        np.ndarray
            Design matrix X with harmonic features and interactions
        """
        months_per_year = 12
        annual_cos = np.cos(2 * np.pi * time / months_per_year)
        annual_sin = np.sin(2 * np.pi * time / months_per_year)
        semiannual_cos = np.cos(4 * np.pi * time / months_per_year)
        semiannual_sin = np.sin(4 * np.pi * time / months_per_year)

        # Stack all features into the design matrix X
        X = np.vstack(
            [
                t_glob,
                annual_cos,
                annual_sin,
                semiannual_cos,
                semiannual_sin,
                t_glob * annual_cos,
                t_glob * annual_sin,
                t_glob * semiannual_cos,
                t_glob * semiannual_sin,
            ]
        ).T

        return X

    # pylint: disable=missing-type-doc,too-many-locals
    def fit(
        self,
        monthly_data,
        variable_name,
        custom_global_temp=None,
        picontrol_baseline=None,
    ):
        """
        Fit the noise generator to monthly climate data.

        # pylint: disable=missing-type-doc

        Parameters
        ----------
        monthly_data : xr.Dataset
            Monthly climate data with dimensions (month, lat, lon, ens)
        variable_name : str
            Name of the variable to model (e.g., 'tas', 'pr')
        custom_global_temp : array-like, optional
            Custom smoothed global mean temperature timeseries to use instead
            of computing from the data. Must have same length as monthly_data
            time dimension. If None, will compute from the variable data.
        picontrol_baseline : float or xr.DataArray, optional
            Pre-industrial control baseline to use for temperature anomalies.
            If provided, global temperature will be computed relative to this
            baseline, ensuring consistency with pattern scaling. If None,
            falls back to using first 42 years of training data as baseline.
        """
        # Extract the variable data
        if variable_name not in monthly_data:
            raise ValueError(f"Variable '{variable_name}' not found in dataset")

        ds = monthly_data.copy()

        # Get time coordinate
        time = ds["month"].values

        # Calculate or use provided global temperature
        if custom_global_temp is not None:
            # Validate custom temperature array
            if len(custom_global_temp) != len(time):
                raise ValueError(
                    f"custom_global_temp length ({len(custom_global_temp)}) "
                    f"must match data time dimension ({len(time)})"
                )
            t_glob = np.array(custom_global_temp)
        else:
            # Calculate latitude-weighted global mean temperature
            t_globm = global_mean(ds[variable_name].mean(dim=["ens"]))

            # Apply baseline correction
            if picontrol_baseline is not None:
                # Use piControl baseline for consistency with pattern scaling
                if isinstance(picontrol_baseline, (int, float)):
                    baseline = picontrol_baseline
                else:
                    # Assume it's an array-like, take mean
                    baseline = float(np.mean(picontrol_baseline))
                t_globm = t_globm - baseline
                print(f"   Using piControl baseline: {baseline:.3f}")
            else:
                # Fall back to original method (first 42 years)
                t_globm = t_globm - t_globm[:500].mean()  # Remove baseline
                print("   Using first 42 years as baseline")

            # Apply smoothing
            t_glob = (
                t_globm.rolling(month=60, center=True)
                .mean()
                .interpolate_na("month", method="nearest", fill_value="extrapolate")
                .values
            )

        # Create harmonic features
        X = self._create_harmonic_features(time, t_glob)

        # Prepare data for seasonal cycle fitting
        Y_xr = (  # pylint: disable=invalid-name
            ds[variable_name].mean(dim=["ens"]).stack(space=("lat", "lon"))
        )
        Y = Y_xr.data  # pylint: disable=invalid-name

        # Fit seasonal cycle model
        self.seasonal_model = LinearRegression(fit_intercept=True)
        self.seasonal_model.fit(X, Y)

        # Reconstruct seasonal cycle
        seasonal_cycle_fit = self.seasonal_model.predict(X)
        seasonal_cycle_fit_xr = xr.DataArray(
            seasonal_cycle_fit, coords=Y_xr.coords, dims=Y_xr.dims
        ).unstack("space")

        # Calculate anomalies
        anomalies = ds[variable_name].mean(dim=["ens"]) - seasonal_cycle_fit_xr

        # Fit PCA to anomalies
        anomalies_flat = anomalies.stack(space=("lat", "lon")).data

        self.pca = PCA(n_components=self.n_modes)
        pcs = self.pca.fit_transform(anomalies_flat)

        # Fit VARX model to PCs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_var = VAR(
                endog=pcs, exog=X[:, :3]
            )  # Use first 3 features as exogenous
            self.varx_results = model_var.fit(self.lag_order)

        # Store coordinate information
        self.coords = {
            "lat": ds.coords["lat"],
            "lon": ds.coords["lon"],
            "month": ds.coords["month"],
        }

        self.fitted = True

        print("Noise generator fitted successfully.")
        print(f"   - PCA modes: {self.n_modes}")
        print(
            f"   - Variance explained: {self.pca.explained_variance_ratio_.sum():.2%}"
        )
        print(f"   - VARX lag order: {self.lag_order}")

    # pylint: disable=missing-type-doc,too-many-locals
    def generate_stochastic_pcs(
        self,
        global_temp_trajectory,
        n_realizations=1,
        random_seed=None,
    ):
        """
        Generate stochastic principal component time series.

        This method generates only the stochastic PC loadings, which can be
        used to reconstruct either gridded fields or regional/global means.
        This enables self-consistent ensemble generation across different
        spatial aggregations.

        Parameters
        ----------
        global_temp_trajectory : array-like
            Global temperature trajectory (used for exogenous variables in VARX model)
        n_realizations : int, default 1
            Number of realizations to generate
        random_seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        np.ndarray
            Stochastic PC time series with shape:
            - (n_time, n_modes) if n_realizations == 1
            - (n_realizations, n_time, n_modes) if n_realizations > 1

        Examples
        --------
        >>> # Generate PCs once, use for multiple outputs
        >>> pcs = model.generate_stochastic_pcs(monthly_warming, n_realizations=100)
        >>> global_means = model.generate_regional_mean_realizations(
        ...     monthly_warming, region='global', stochastic_pcs=pcs)
        >>> neu_means = model.generate_regional_mean_realizations(
        ...     monthly_warming, region='NEU', stochastic_pcs=pcs)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before generating realizations")

        if random_seed is not None:
            np.random.seed(random_seed)

        n_time = len(global_temp_trajectory)
        time = np.arange(n_time)

        # Create exogenous variables for VARX
        X = self._create_harmonic_features(time, global_temp_trajectory)
        X_exog = X[:, :3]  # First 3 columns: t_glob, annual_cos, annual_sin

        # Generate stochastic PCs for each realization
        if n_realizations == 1:
            return self._generate_stochastic_pcs(X_exog, n_time)
        else:
            all_pcs = []
            for _ in range(n_realizations):
                pcs = self._generate_stochastic_pcs(X_exog, n_time)
                all_pcs.append(pcs)
            return np.array(all_pcs)  # Shape: (n_realizations, n_time, n_modes)

    # pylint: disable=missing-type-doc,too-many-locals
    def generate_realization(
        self,
        global_temp_trajectory,
        n_realizations=1,
        random_seed=None,
        noise_only=False,
        add_base=None,
    ):
        """
        Generate stochastic climate realizations.

        Parameters
        ----------
        global_temp_trajectory : array-like
            Global temperature trajectory to drive the seasonal cycle.
            If noise_only=True, this can be any length array (values ignored for temperature effects).
        n_realizations : int, default 1
            Number of realizations to generate
        random_seed : int, optional
            Random seed for reproducibility
        noise_only : bool, default False
            If True, generate only the stochastic noise component without direct temperature
            effects or constant terms, but preserve temperature-modulated seasonal harmonics.
            This is useful for adding to METEOR annual predictions.
        add_base : xr.DataArray, optional
            Base climatology to add to each realization. If provided, the addition is done
            efficiently in NumPy before XArray conversion, avoiding expensive XArray operations.
            Must have compatible shape with the output.

        Returns
        -------
        xr.DataArray or list of xr.DataArray
            Generated climate realizations. If noise_only=True, returns just the
            stochastic component that can be added to other predictions.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before generating realizations")

        if random_seed is not None:
            np.random.seed(random_seed)

        # Create time coordinate (shared across all realizations)
        n_time = len(global_temp_trajectory)
        time = np.arange(n_time)

        # Precompute harmonic features (shared across all realizations)
        X = self._create_harmonic_features(time, global_temp_trajectory)
        X_exog = X[:, :3]  # Exogenous variables for VARX

        # Precompute seasonal cycle as NumPy array (shared across all realizations)
        if noise_only:
            # For noise-only: keep seasonal harmonics AND temperature-modulated harmonics
            # but remove the direct temperature effect (intercept + t_glob term)
            seasonal_cycle = self.seasonal_model.predict(X)

            # Calculate what to subtract (intercept + direct temperature effect)
            intercept_effect = self.seasonal_model.intercept_
            temp_effect = (
                self.seasonal_model.coef_[:, 0] * global_temp_trajectory[:, np.newaxis]
            )

            # Remove intercept and direct temperature effect from seasonal cycle
            seasonal_cycle_np = (
                seasonal_cycle - intercept_effect[np.newaxis, :] - temp_effect
            )
        else:
            # Standard operation: full seasonal cycle with temperature dependence
            seasonal_cycle_np = self.seasonal_model.predict(X)

        # Reshape seasonal cycle once
        n_lat = len(self.coords["lat"])
        n_lon = len(self.coords["lon"])
        seasonal_cycle_reshaped = seasonal_cycle_np.reshape(n_time, n_lat, n_lon)

        # Convert base climatology to NumPy if provided (once for all realizations)
        base_clim_np = None
        if add_base is not None:
            # Handle potential ensemble dimension and squeeze it
            base_values = add_base.values
            if base_values.ndim == 4:
                # Shape is (ens, time, lat, lon) - squeeze out ensemble dimension
                base_values = base_values.squeeze()

            # Now reshape to (n_time, n_lat, n_lon)
            # If the time dimension doesn't match, select the first n_time steps
            if base_values.shape[0] != n_time:
                base_values = base_values[:n_time, :, :]

            base_clim_np = base_values.reshape(n_time, n_lat, n_lon)

        # Generate realizations (only stochastic component varies)
        realizations = []
        for _ in range(n_realizations):
            # Generate stochastic component (this is the only unique part per realization)
            synthetic_pcs = self._generate_stochastic_pcs(X_exog, n_time)

            # Reconstruct anomalies (NumPy)
            reconstructed_anomalies = synthetic_pcs @ self.pca.components_
            reconstructed_anomalies_reshaped = reconstructed_anomalies.reshape(
                n_time, n_lat, n_lon
            )

            # Combine seasonal cycle and anomalies in NumPy (FAST!)
            realization_np = seasonal_cycle_reshaped + reconstructed_anomalies_reshaped

            # Add base climatology in NumPy if provided (FAST!)
            if base_clim_np is not None:
                realization_np = realization_np + base_clim_np

            # Convert to xarray only once at the end
            realization_xr = xr.DataArray(
                realization_np,
                coords={
                    "month": time,
                    "lat": self.coords["lat"],
                    "lon": self.coords["lon"],
                },
                dims=("month", "lat", "lon"),
            )
            realizations.append(realization_xr)

        return realizations if n_realizations > 1 else realizations[0]

    # pylint: disable=invalid-name
    def _generate_stochastic_pcs(self, X_exog, n_time):
        """
        Generate stochastic principal components using fitted VARX model.

        Parameters
        ----------
        X_exog : np.ndarray
            Exogenous variables for VARX model
        n_time : int
            Number of time steps to generate

        Returns
        -------
        np.ndarray
            Generated principal components
        """
        synthetic_pcs = np.zeros((n_time, self.n_modes))

        # Use zero initial conditions (could be improved)
        synthetic_pcs[: self.lag_order] = 0

        # Get residual covariance
        residual_cov = self.varx_results.sigma_u
        mean_shock = np.zeros(self.n_modes)

        # Generate time series
        for t in range(self.lag_order, n_time):
            current_initial_conditions = synthetic_pcs[t - self.lag_order : t]
            current_exog = X_exog[t : t + 1]

            # Get mean forecast
            mean_forecast = self.varx_results.forecast(
                y=current_initial_conditions, steps=1, exog_future=current_exog
            )

            # Add random shock
            random_shock = np.random.multivariate_normal(mean_shock, residual_cov)
            synthetic_pcs[t] = mean_forecast.flatten() + random_shock

        return synthetic_pcs

    def _compute_spatial_weights(self):
        """Compute area-weighted spatial averaging weights (cosine of latitude)."""
        lats = self.coords["lat"]
        if hasattr(lats, "values"):
            lats = lats.values
        return np.cos(np.deg2rad(lats))

    def _get_regional_eof_projection(self, region, region_mask=None):
        """
        Get or compute the spatial mean projection of each EOF for a region.

        Cached in memory since EOFs are model-invariant.

        Parameters
        ----------
        region : str
            Region identifier ('global' or AR6 region code like 'NEU')
        region_mask : np.ndarray, optional
            Custom 2D boolean mask (n_lat, n_lon) for the region

        Returns
        -------
        np.ndarray
            Mean projection of each EOF mode for the region, shape (n_modes,)
        """
        # Check cache first
        region_id = region if region_mask is None else f"custom_{id(region_mask)}"
        if region_id in self._regional_eof_projections:
            return self._regional_eof_projections[region_id]

        # Compute EOF projections
        n_lat = len(self.coords["lat"])
        n_lon = len(self.coords["lon"])

        # Get EOFs reshaped to spatial grid (n_modes, n_lat, n_lon)
        eof_components = self.pca.components_.reshape(self.n_modes, n_lat, n_lon)

        # Compute area weights
        weights = self._compute_spatial_weights()
        weight_grid = weights[:, np.newaxis]  # (n_lat, 1) for broadcasting

        # Apply regional mask if needed
        if region == "global" and region_mask is None:
            # Global mean: average over all gridpoints
            # Total weight is sum(lat_weights) * n_lon since weights broadcast across longitude
            total_weight = np.sum(weights) * n_lon
            eof_projections = np.array(
                [
                    np.sum(eof_components[i] * weight_grid) / total_weight
                    for i in range(self.n_modes)
                ]
            )
        else:
            # Regional mean: use mask
            if region_mask is None:
                # Use AR6 regions (requires regionmask)
                try:
                    import regionmask

                    ar6_regions = regionmask.defined_regions.ar6.all

                    # Find region number
                    region_number = None
                    for r in ar6_regions:
                        if r.abbrev == region:
                            region_number = r.number
                            break

                    if region_number is None:
                        raise ValueError(f"AR6 region '{region}' not found")

                    # Create mask on this grid
                    lons = (
                        self.coords["lon"].values
                        if hasattr(self.coords["lon"], "values")
                        else self.coords["lon"]
                    )
                    lats = (
                        self.coords["lat"].values
                        if hasattr(self.coords["lat"], "values")
                        else self.coords["lat"]
                    )
                    lon_2d, lat_2d = np.meshgrid(lons, lats)
                    mask_3d = ar6_regions.mask(lon_2d, lat_2d)
                    region_mask = mask_3d == region_number

                except ImportError as exc:
                    raise ImportError(
                        "regionmask is required for AR6 regions. "
                        "Install with: pip install regionmask"
                    ) from exc

            # Compute weighted mean over region
            eof_projections = np.zeros(self.n_modes)
            for i in range(self.n_modes):
                masked_eof = np.where(region_mask, eof_components[i], np.nan)
                masked_weights = np.where(region_mask, weight_grid, 0)
                eof_projections[i] = np.nansum(masked_eof * masked_weights) / np.sum(
                    masked_weights
                )

        # Cache and return
        self._regional_eof_projections[region_id] = eof_projections
        return eof_projections

    def generate_regional_mean_realizations(
        self,
        global_temp_trajectory,
        region="global",
        region_mask=None,
        n_realizations=1,
        random_seed=None,
        noise_only=False,
        add_base=None,
        stochastic_pcs=None,
        return_numpy=False,
    ):
        """
        Generate regional or global mean realizations efficiently.

        This method avoids creating full 3D gridded fields by computing the
        regional mean directly from the PC projections. This is orders of
        magnitude faster when only scalar time series are needed.

        Parameters
        ----------
        global_temp_trajectory : array-like
            Global temperature trajectory to drive the seasonal cycle
        region : str, default 'global'
            Region identifier: 'global' or AR6 region code (e.g., 'NEU', 'WNA')
        region_mask : np.ndarray, optional
            Custom 2D boolean mask (n_lat, n_lon) for region. If provided, overrides `region`.
        n_realizations : int, default 1
            Number of realizations to generate
        random_seed : int, optional
            Random seed for reproducibility
        noise_only : bool, default False
            If True, generate only stochastic component (for adding to predictions)
        add_base : xr.DataArray or np.ndarray, optional
            Base climatology to add. Can be:
            - Scalar time series (n_time,) - will be added directly
            - Gridded field (n_time, n_lat, n_lon) - will be spatially averaged
        stochastic_pcs : np.ndarray, optional
            Pre-generated stochastic PCs from generate_stochastic_pcs().
            If provided, these PCs are used (enabling self-consistent multi-region generation).
            Shape: (n_time, n_modes) or (n_realizations, n_time, n_modes)
        return_numpy : bool, default False
            If True, return numpy arrays. If False, return xarray DataArrays.

        Returns
        -------
        xr.DataArray or np.ndarray
            Regional/global mean time series.
            
            - If n_realizations == 1:
              Shape (n_time,) with dims ('month',)
            - If n_realizations > 1:
              Shape (n_realizations, n_time) with dims ('realization', 'month')
            
            When return_numpy=False (default), returns xarray DataArray with proper
            coordinates and dims. When return_numpy=True, returns numpy array.

        Examples
        --------
        >>> # Fast generation of 100 global mean realizations
        >>> global_means = model.generate_regional_mean_realizations(
        ...     monthly_warming, region='global', n_realizations=100)
        >>> # Returns shape (100, n_time) with dims ('realization', 'month')
        >>>
        >>> # Self-consistent multi-region generation
        >>> pcs = model.generate_stochastic_pcs(monthly_warming, n_realizations=50)
        >>> global_m = model.generate_regional_mean_realizations(
        ...     monthly_warming, region='global', stochastic_pcs=pcs)
        >>> neu_m = model.generate_regional_mean_realizations(
        ...     monthly_warming, region='NEU', stochastic_pcs=pcs)
        >>> # Both share the same stochastic variability from pcs
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before generating realizations")

        if random_seed is not None and stochastic_pcs is None:
            np.random.seed(random_seed)

        n_time = len(global_temp_trajectory)
        time = np.arange(n_time)

        # Get EOF projections for this region (cached)
        eof_projections = self._get_regional_eof_projection(region, region_mask)

        # Compute seasonal cycle regional mean
        X = self._create_harmonic_features(time, global_temp_trajectory)

        if noise_only:
            # For noise-only: seasonal harmonics without direct temperature effect
            seasonal_cycle = self.seasonal_model.predict(X)
            intercept_effect = self.seasonal_model.intercept_
            temp_effect = (
                self.seasonal_model.coef_[:, 0] * global_temp_trajectory[:, np.newaxis]
            )
            seasonal_cycle_full = (
                seasonal_cycle - intercept_effect[np.newaxis, :] - temp_effect
            )
        else:
            seasonal_cycle_full = self.seasonal_model.predict(X)

        # Compute regional mean of seasonal cycle
        n_lat = len(self.coords["lat"])
        n_lon = len(self.coords["lon"])
        seasonal_cycle_grid = seasonal_cycle_full.reshape(n_time, n_lat, n_lon)

        weights = self._compute_spatial_weights()
        weight_grid = weights[:, np.newaxis]

        if region == "global" and region_mask is None:
            # Total weight is sum(lat_weights) * n_lon since weights broadcast across longitude
            total_weight = np.sum(weights) * n_lon
            seasonal_mean = np.array(
                [
                    np.sum(seasonal_cycle_grid[t] * weight_grid) / total_weight
                    for t in range(n_time)
                ]
            )
        else:
            if region_mask is None:
                # Get mask from AR6 regions
                import regionmask

                ar6_regions = regionmask.defined_regions.ar6.all
                region_number = None
                for r in ar6_regions:
                    if r.abbrev == region:
                        region_number = r.number
                        break
                lons = (
                    self.coords["lon"].values
                    if hasattr(self.coords["lon"], "values")
                    else self.coords["lon"]
                )
                lats = (
                    self.coords["lat"].values
                    if hasattr(self.coords["lat"], "values")
                    else self.coords["lat"]
                )
                lon_2d, lat_2d = np.meshgrid(lons, lats)
                mask_3d = ar6_regions.mask(lon_2d, lat_2d)
                region_mask = mask_3d == region_number

            seasonal_mean = np.zeros(n_time)
            for t in range(n_time):
                masked_data = np.where(region_mask, seasonal_cycle_grid[t], np.nan)
                masked_weights = np.where(region_mask, weight_grid, 0)
                seasonal_mean[t] = np.nansum(masked_data * masked_weights) / np.sum(
                    masked_weights
                )

        # Compute base climatology regional mean (if provided)
        base_mean = None
        if add_base is not None:
            if isinstance(add_base, xr.DataArray):
                add_base = add_base.values

            if add_base.ndim == 1:
                # Already a time series
                base_mean = add_base
            elif add_base.ndim == 3:
                # Gridded field - compute regional mean
                if region == "global" and region_mask is None:
                    # Total weight is sum(lat_weights) * n_lon since weights broadcast across longitude
                    total_weight = np.sum(weights) * n_lon
                    base_mean = np.array(
                        [
                            np.sum(add_base[t] * weight_grid) / total_weight
                            for t in range(n_time)
                        ]
                    )
                else:
                    base_mean = np.zeros(n_time)
                    for t in range(n_time):
                        masked_data = np.where(region_mask, add_base[t], np.nan)
                        masked_weights = np.where(region_mask, weight_grid, 0)
                        base_mean[t] = np.nansum(masked_data * masked_weights) / np.sum(
                            masked_weights
                        )

        # Generate or use provided stochastic PCs
        if stochastic_pcs is None:
            X_exog = X[:, :3]
            if n_realizations == 1:
                pcs_to_use = [self._generate_stochastic_pcs(X_exog, n_time)]
            else:
                pcs_to_use = [
                    self._generate_stochastic_pcs(X_exog, n_time)
                    for _ in range(n_realizations)
                ]
        else:
            # Use provided PCs
            if stochastic_pcs.ndim == 2:
                # Single realization
                pcs_to_use = [stochastic_pcs]
            else:
                # Multiple realizations
                pcs_to_use = list(stochastic_pcs)

        # Reconstruct regional means from PCs
        realizations = []
        for pcs in pcs_to_use:
            # Anomaly contribution: PCs @ EOF_projections
            anomaly_mean = pcs @ eof_projections  # (n_time,)

            # Combine components
            realization = seasonal_mean + anomaly_mean
            if base_mean is not None:
                realization = realization + base_mean

            realizations.append(realization)

        # Return format
        if return_numpy:
            # Return as numpy array with shape (n_realizations, n_time) or (n_time,) if single
            if len(realizations) == 1:
                return realizations[0]
            else:
                return np.array(realizations)
        else:
            # Return as xarray DataArray
            if len(realizations) == 1:
                # Single realization - return 1D DataArray
                return xr.DataArray(
                    realizations[0],
                    coords={"month": time},
                    dims=("month",),
                    attrs={"region": region},
                )
            else:
                # Multiple realizations - concatenate with 'realization' dimension
                return xr.DataArray(
                    np.array(realizations),
                    coords={
                        "realization": np.arange(len(realizations)),
                        "month": time,
                    },
                    dims=("realization", "month"),
                    attrs={"region": region},
                )

    def save_model(self, filepath):
        """
        Save the fitted noise model to disk.

        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            "n_modes": self.n_modes,
            "lag_order": self.lag_order,
            "seasonal_model": self.seasonal_model,
            "pca": self.pca,
            "varx_results": self.varx_results,
            "coords": self.coords,
            "fitted": self.fitted,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a fitted noise model from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved model
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)  # nosec - Loading trusted model files only

        self.n_modes = model_data["n_modes"]
        self.lag_order = model_data["lag_order"]
        self.seasonal_model = model_data["seasonal_model"]
        self.pca = model_data["pca"]
        self.varx_results = model_data["varx_results"]
        self.coords = model_data["coords"]
        self.fitted = model_data["fitted"]

        print(f"Model loaded from {filepath}")


# pylint: disable=too-many-arguments,too-many-positional-arguments,missing-type-doc
def train_noise_model_from_cmip6(
    data_getter,
    experiments,
    model_name,
    variable_name,
    n_modes=10,
    lag_order=2,
    cache_dir=None,
    custom_global_temp=None,
    use_picontrol_baseline=True,
):
    """
    Train a noise generator from CMIP6 data.

    This is the primary training interface that provides intuitive access
    to noise model training from CMIP6 composite experimental data.

    Parameters
    ----------
    data_getter : Cmip6MeteorDataGetter
        Data getter instance with access to CMIP6 data
    experiments : list
        List of experiments to use for training (e.g., ["historical", "ssp245"])
    model_name : str
        Name of the climate model (must be available in data_getter)
    variable_name : str
        Variable to model (e.g., 'tas', 'pr')
    n_modes : int, default 10
        Number of PCA modes to retain
    lag_order : int, default 2
        Lag order for VARX model
    cache_dir : str, optional
        Directory to cache the trained model. If None, model is not cached.
    custom_global_temp : array-like, optional
        Custom smoothed global mean temperature timeseries to use for training
        instead of computing from the variable data. Must have same length as
        the monthly data time dimension. Useful when you want to use a specific
        temperature trajectory (e.g., from a different variable or processing).
    use_picontrol_baseline : bool, default True
        Whether to use piControl data as baseline for temperature anomalies.
        This ensures consistency with pattern scaling.
        If False, falls back to using first 42 years of training data.

    Returns
    -------
    MeteorNoiseGenerator
        Fitted noise generator ready for realization generation

    Examples
    --------
    >>> # Train a temperature noise model with piControl baseline
    >>> noise_model = train_noise_model_from_cmip6(
    ...     data_getter, ["historical", "ssp245"], "CanESM5", "tas",
    ...     n_modes=8, cache_dir="./models"
    ... )
    >>>
    >>> # Generate realizations
    >>> realizations = noise_model.generate_realization(temp_trajectory)
    """
    # Get monthly training data
    monthly_data = data_getter.make_meteor_training_data_composite(
        experiments, model_name, monthly=True
    )

    # Get piControl baseline if requested
    picontrol_baseline = None
    if use_picontrol_baseline:
        try:
            # Try to fetch piControl data for baseline calculation
            picontrol_data = data_getter.get_single_var_mod_data_yearmean(
                "piControl", variable_name, model_name
            )
            picontrol_baseline = float(picontrol_data.mean().values)
            print(
                f"   Fetched piControl baseline for {model_name} {variable_name}: {picontrol_baseline:.3f}"
            )
        except (KeyError, AttributeError) as e:
            print(
                f"   Warning: Could not fetch piControl data ({e}), falling back to legacy baseline"
            )
            picontrol_baseline = None

    # Create and fit noise generator
    noise_gen = MeteorNoiseGenerator(n_modes=n_modes, lag_order=lag_order)
    noise_gen.fit(
        monthly_data,
        variable_name,
        custom_global_temp=custom_global_temp,
        picontrol_baseline=picontrol_baseline,
    )

    # Cache if requested
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f"{model_name}_{variable_name}_noise_model.pkl"
        )
        noise_gen.save_model(cache_path)

    return noise_gen


# pylint: disable=too-many-arguments,too-many-positional-arguments,missing-type-doc
def train_multiple_noise_models_from_cmip6(
    data_getter,
    experiments,
    models=None,
    variables=None,
    n_modes=10,
    lag_order=2,
    cache_dir=None,
    custom_global_temp=None,
    use_picontrol_baseline=True,
):
    """
    Train noise generators for multiple model/variable combinations.

    This method provides batch training functionality for multiple
    models and variables, useful for comprehensive noise model creation.

    Parameters
    ----------
    data_getter : Cmip6MeteorDataGetter
        Data getter instance with access to CMIP6 data
    experiments : list
        List of experiments to use for training
    models : list, optional
        List of models to train. If None, uses all available models.
    variables : list, optional
        List of variables to train. If None, uses all fields in data getter.
    n_modes : int, default 10
        Number of PCA modes to retain
    lag_order : int, default 2
        Lag order for VARX model
    cache_dir : str, optional
        Directory to cache trained models
    custom_global_temp : array-like, optional
        Custom smoothed global mean temperature timeseries to use for all
        model/variable combinations. Must have same length as monthly data.
    use_picontrol_baseline : bool, default True
        Whether to use piControl data as baseline for temperature anomalies.
        This ensures consistency with pattern scaling.
        If False, falls back to using first 42 years of training data.

    Returns
    -------
    dict
        Dictionary mapping (model, variable) tuples to fitted noise generators.
        Nested dictionary with structure: {model: {variable: MeteorNoiseGenerator}}

    Examples
    --------
    >>> # Train noise models for all available combinations with piControl baseline
    >>> noise_models = train_multiple_noise_models_from_cmip6(
    ...     data_getter, ["historical", "ssp245"],
    ...     models=["CanESM5", "CESM2"], variables=["tas", "pr"],
    ...     cache_dir="./models"
    ... )
    >>>
    >>> # Access specific model
    >>> tas_model = noise_models["CanESM5"]["tas"]
    """
    if models is None:
        models = data_getter.models
    if variables is None:
        variables = data_getter.flds

    noise_models = {}

    for model in models:
        if not data_getter.check_if_model_has_data(model):
            print(f"Skipping {model} - no complete data available")
            continue

        noise_models[model] = {}

        for variable in variables:
            print(f"Training noise model for {model} - {variable}")
            try:
                noise_gen = train_noise_model_from_cmip6(
                    data_getter,
                    experiments,
                    model,
                    variable,
                    n_modes=n_modes,
                    lag_order=lag_order,
                    cache_dir=cache_dir,
                    custom_global_temp=custom_global_temp,
                    use_picontrol_baseline=use_picontrol_baseline,
                )
                noise_models[model][variable] = noise_gen
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Failed to train noise model for {model} - {variable}: {e}")
                continue

    return noise_models


def load_noise_model_from_cache(cache_dir, model_name, variable_name):
    """
    Load a previously cached noise model.

    Parameters
    ----------
    cache_dir : str
        Directory containing cached models
    model_name : str
        Name of the climate model
    variable_name : str
        Name of the variable

    Returns
    -------
    MeteorNoiseGenerator
        Loaded noise generator

    Examples
    --------
    >>> # Load a previously trained model
    >>> noise_model = load_noise_model_from_cache(
    ...     "./models", "CanESM5", "tas"
    ... )
    """
    cache_path = os.path.join(
        cache_dir, f"{model_name}_{variable_name}_noise_model.pkl"
    )

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"No cached model found at {cache_path}")

    noise_gen = MeteorNoiseGenerator()
    noise_gen.load_model(cache_path)
    return noise_gen


# pylint: disable=too-many-arguments,too-many-positional-arguments,missing-type-doc
def train_noise_model_from_composite(
    data_getter,
    experiments,
    model_name,
    variable_name,
    n_modes=10,
    lag_order=2,
    cache_dir=None,
):
    """
    Train a noise model from composite experimental data.

    Parameters
    ----------
    data_getter : Cmip6MeteorDataGetter
        Data getter instance
    experiments : list
        List of experiments to use for training (e.g., ["historical", "ssp245"])
    model_name : str
        Name of the climate model
    variable_name : str
        Variable to model (e.g., 'tas', 'pr')
    n_modes : int, default 10
        Number of PCA modes
    lag_order : int, default 2
        VARX lag order
    cache_dir : str, optional
        Directory to cache the trained model

    Returns
    -------
    MeteorNoiseGenerator
        Fitted noise generator
    """
    # Use the new function for consistency
    return train_noise_model_from_cmip6(
        data_getter,
        experiments,
        model_name,
        variable_name,
        n_modes=n_modes,
        lag_order=lag_order,
        cache_dir=cache_dir,
    )
