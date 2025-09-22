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

import numpy as np
import xarray as xr
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR
import warnings


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
        n_modes : int, optional
            Number of PCA modes to retain. Default is 10.
        lag_order : int, optional
            Lag order for VARX model. Default is 2.
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

    def fit(self, monthly_data, variable_name):
        """
        Fit the noise generator to monthly climate data.

        Parameters
        ----------
        monthly_data : xr.Dataset
            Monthly climate data with dimensions (month, lat, lon, ens)
        variable_name : str
            Name of the variable to model (e.g., 'tas', 'pr')
        """
        # Extract the variable data
        if variable_name not in monthly_data:
            raise ValueError(f"Variable '{variable_name}' not found in dataset")

        ds = monthly_data.copy()

        # Calculate global mean temperature and remove ensemble mean for reference
        t_globm = ds[variable_name].mean(dim=["lat", "lon", "ens"])
        t_globm = t_globm - t_globm[:500].mean()  # Remove baseline

        # Apply smoothing
        t_glob = (
            t_globm.rolling(month=60, center=True)
            .mean()
            .interpolate_na("month", method="nearest", fill_value="extrapolate")
            .values
        )

        # Get time coordinate
        time = ds["month"].values

        # Create harmonic features
        X = self._create_harmonic_features(time, t_glob)

        # Prepare data for seasonal cycle fitting
        Y_xr = ds[variable_name].mean(dim=["ens"]).stack(space=("lat", "lon"))
        Y = Y_xr.data

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

        print(f"✅ Noise generator fitted successfully.")
        print(f"   - PCA modes: {self.n_modes}")
        print(
            f"   - Variance explained: {self.pca.explained_variance_ratio_.sum():.2%}"
        )
        print(f"   - VARX lag order: {self.lag_order}")

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
        n_realizations : int, optional
            Number of realizations to generate. Default is 1.
        random_seed : int, optional
            Random seed for reproducibility
        noise_only : bool, optional
            If True, generate only the stochastic noise component without direct temperature
            effects or constant terms, but preserve temperature-modulated seasonal harmonics.
            This is useful for adding to METEOR annual predictions. Default is False.        Returns
        -------
        xr.DataArray or list of xr.DataArray
            Generated climate realizations. If noise_only=True, returns just the
            stochastic component that can be added to other predictions.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before generating realizations")

        if random_seed is not None:
            np.random.seed(random_seed)

        realizations = []

        for _ in range(n_realizations):
            # Create time coordinate
            n_time = len(global_temp_trajectory)
            time = np.arange(n_time)

            if noise_only:
                # For noise-only: keep seasonal harmonics AND temperature-modulated harmonics
                # but remove the direct temperature effect (intercept + t_glob term)
                X = self._create_harmonic_features(time, global_temp_trajectory)

                # Generate full seasonal cycle prediction
                seasonal_cycle = self.seasonal_model.predict(X)

                # Now remove only the direct temperature effects (intercept + t_glob)
                # Keep: harmonics (indices 1-4) + temperature-modulated harmonics (indices 5-8)
                # Remove: intercept + t_glob (index 0)

                # Create a modified prediction by zeroing out unwanted terms
                # The seasonal model coefficients are organized as:
                # coef_[:, 0] = t_glob coefficient (direct temperature effect)
                # coef_[:, 1-4] = pure harmonic coefficients
                # coef_[:, 5-8] = temperature-modulated harmonic coefficients

                # Calculate what to subtract (intercept + direct temperature effect)
                intercept_effect = self.seasonal_model.intercept_
                temp_effect = (
                    self.seasonal_model.coef_[:, 0]
                    * global_temp_trajectory[:, np.newaxis]
                )

                # Remove intercept and direct temperature effect from seasonal cycle
                seasonal_cycle_adjusted = (
                    seasonal_cycle - intercept_effect[np.newaxis, :] - temp_effect
                )

                seasonal_cycle_xr = xr.DataArray(
                    seasonal_cycle_adjusted.reshape(
                        n_time, len(self.coords["lat"]), len(self.coords["lon"])
                    ),
                    coords={
                        "month": time,
                        "lat": self.coords["lat"],
                        "lon": self.coords["lon"],
                    },
                    dims=("month", "lat", "lon"),
                )
            else:
                # Standard operation: full seasonal cycle with temperature dependence
                X = self._create_harmonic_features(time, global_temp_trajectory)

                # Generate seasonal cycle
                seasonal_cycle = self.seasonal_model.predict(X)
                seasonal_cycle_xr = xr.DataArray(
                    seasonal_cycle.reshape(
                        n_time, len(self.coords["lat"]), len(self.coords["lon"])
                    ),
                    coords={
                        "month": time,
                        "lat": self.coords["lat"],
                        "lon": self.coords["lon"],
                    },
                    dims=("month", "lat", "lon"),
                )

            # Generate stochastic component
            if noise_only:
                # For noise-only, keep temperature in exogenous variables but use actual temperature
                # This preserves the temperature-dependent variability patterns
                synthetic_pcs = self._generate_stochastic_pcs(X[:, :3], n_time)
            else:
                synthetic_pcs = self._generate_stochastic_pcs(X[:, :3], n_time)

            # Reconstruct anomalies
            reconstructed_anomalies = synthetic_pcs @ self.pca.components_
            reconstructed_anomalies_xr = xr.DataArray(
                reconstructed_anomalies.reshape(
                    n_time, len(self.coords["lat"]), len(self.coords["lon"])
                ),
                coords={
                    "month": time,
                    "lat": self.coords["lat"],
                    "lon": self.coords["lon"],
                },
                dims=("month", "lat", "lon"),
            )

            # Combine seasonal cycle and anomalies
            realization = seasonal_cycle_xr + reconstructed_anomalies_xr
            realizations.append(realization)

        return realizations if n_realizations > 1 else realizations[0]

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
            model_data = pickle.load(f)

        self.n_modes = model_data["n_modes"]
        self.lag_order = model_data["lag_order"]
        self.seasonal_model = model_data["seasonal_model"]
        self.pca = model_data["pca"]
        self.varx_results = model_data["varx_results"]
        self.coords = model_data["coords"]
        self.fitted = model_data["fitted"]

        print(f"Model loaded from {filepath}")

    @classmethod
    def train_from_cmip6(
        cls,
        data_getter,
        experiments,
        model_name,
        variable_name,
        n_modes=10,
        lag_order=2,
        cache_dir=None,
    ):
        """
        Train a noise generator directly from CMIP6 data getter.

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
        n_modes : int, optional
            Number of PCA modes to retain. Default is 10.
        lag_order : int, optional
            Lag order for VARX model. Default is 2.
        cache_dir : str, optional
            Directory to cache the trained model. If None, model is not cached.

        Returns
        -------
        MeteorNoiseGenerator
            Fitted noise generator ready for realization generation

        Examples
        --------
        >>> # Train a temperature noise model
        >>> noise_model = MeteorNoiseGenerator.train_from_cmip6(
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

        # Create and fit noise generator
        noise_gen = cls(n_modes=n_modes, lag_order=lag_order)
        noise_gen.fit(monthly_data, variable_name)

        # Cache if requested
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(
                cache_dir, f"{model_name}_{variable_name}_noise_model.pkl"
            )
            noise_gen.save_model(cache_path)

        return noise_gen

    @classmethod
    def train_multiple_from_cmip6(
        cls,
        data_getter,
        experiments,
        models=None,
        variables=None,
        n_modes=10,
        lag_order=2,
        cache_dir=None,
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
        n_modes : int, optional
            Number of PCA modes to retain. Default is 10.
        lag_order : int, optional
            Lag order for VARX model. Default is 2.
        cache_dir : str, optional
            Directory to cache trained models

        Returns
        -------
        dict
            Nested dictionary with structure: {model: {variable: MeteorNoiseGenerator}}

        Examples
        --------
        >>> # Train noise models for all available combinations
        >>> noise_models = MeteorNoiseGenerator.train_multiple_from_cmip6(
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
                    noise_gen = cls.train_from_cmip6(
                        data_getter,
                        experiments,
                        model,
                        variable,
                        n_modes=n_modes,
                        lag_order=lag_order,
                        cache_dir=cache_dir,
                    )
                    noise_models[model][variable] = noise_gen
                except Exception as e:
                    print(f"Failed to train noise model for {model} - {variable}: {e}")
                    continue

        return noise_models

    @classmethod
    def load_from_cache(cls, cache_dir, model_name, variable_name):
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
        >>> noise_model = MeteorNoiseGenerator.load_from_cache(
        ...     "./models", "CanESM5", "tas"
        ... )
        """
        cache_path = os.path.join(
            cache_dir, f"{model_name}_{variable_name}_noise_model.pkl"
        )

        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"No cached model found at {cache_path}")

        noise_gen = cls()
        noise_gen.load_model(cache_path)
        return noise_gen


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
    Convenience function to train a noise model from composite experimental data.


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
    n_modes : int, optional
        Number of PCA modes. Default is 10.
    lag_order : int, optional
        VARX lag order. Default is 2.
    cache_dir : str, optional
        Directory to cache the trained model

    Returns
    -------
    MeteorNoiseGenerator
        Fitted noise generator
    """
    # Use the new class method for consistency
    return MeteorNoiseGenerator.train_from_cmip6(
        data_getter,
        experiments,
        model_name,
        variable_name,
        n_modes=n_modes,
        lag_order=lag_order,
        cache_dir=cache_dir,
    )
