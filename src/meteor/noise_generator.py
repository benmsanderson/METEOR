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
    def generate_realization(
        self,
        global_temp_trajectory,
        n_realizations=1,
        random_seed=None,
        noise_only=False,
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
