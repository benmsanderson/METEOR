#!/usr/bin/env python3
"""
Build cache of trained noise models for METEOR Streamlit app.

This script pre-trains and caches noise models for specified models, scenarios,
and variables to speed up the Streamlit app. Noise models are slow to train
(~30-60 seconds each) but fast to load from cache (~1 second).

Usage:
    python build_noise_cache.py                          # Cache default models/scenarios
    python build_noise_cache.py --models CanESM5 CESM2   # Cache specific models
    python build_noise_cache.py --scenarios ssp245       # Cache specific scenario
    python build_noise_cache.py --variables tas          # Cache only temperature
    python build_noise_cache.py --n-modes 10 --lag-order 3  # Custom parameters
    python build_noise_cache.py --force                  # Re-train existing caches
    python build_noise_cache.py --list-models            # Show available models
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from meteor import Cmip6MeteorDataGetter

# Default models to cache (high-quality, commonly used models)
DEFAULT_MODELS = ["CanESM5", "CESM2", "UKESM1-0-LL", "ACCESS-ESM1-5", "MIROC-ES2L"]

DEFAULT_SCENARIOS = ["ssp126", "ssp245", "ssp585"]
DEFAULT_VARIABLES = ["tas", "pr"]


def initialize_data_getter(scenarios):
    """Initialize CMIP6 data getter with required experiments."""
    # Need piControl, historical, and all scenarios
    experiments = ["piControl", "abrupt-4xCO2", "historical"] + scenarios

    # Map to database locations
    dbe = ["CMIP", "CMIP", "CMIP"] + ["ScenarioMIP"] * len(scenarios)

    print(f"Initializing data getter with experiments: {experiments}")
    return Cmip6MeteorDataGetter(exps=experiments, flds=DEFAULT_VARIABLES, dbe=dbe)


def get_available_models(data_getter, requested_models=None):
    """Get list of available models with complete data."""
    print("Checking available models...")

    available = []
    for model in tqdm(data_getter.models, desc="Checking models"):
        if data_getter.check_if_model_has_data(model):
            available.append(model)

    print(f"\nFound {len(available)} models with complete data")

    # Filter to requested models if specified
    if requested_models:
        available = [m for m in available if m in requested_models]
        print(f"Filtered to {len(available)} requested models")

    return sorted(available)


def noise_cache_exists(model_name, scenario, variable, n_modes, lag_order, cache_dir):
    """Check if noise model cache already exists."""
    cache_file = os.path.join(
        cache_dir,
        f"{model_name}_{scenario}_{variable}_n{n_modes}_lag{lag_order}_noise.pkl",
    )
    return os.path.exists(cache_file)


def train_and_cache_noise_model(
    data_getter, model_name, scenario, variable, n_modes, lag_order, cache_dir
):
    """Train noise model and save to cache."""
    cache_file = os.path.join(
        cache_dir,
        f"{model_name}_{scenario}_{variable}_n{n_modes}_lag{lag_order}_noise.pkl",
    )

    print(f"  Training noise model for {variable}...")
    print(f"    Parameters: n_modes={n_modes}, lag_order={lag_order}")

    try:
        noise_model = data_getter.train_noise_model(
            experiments=["historical", scenario],
            model=model_name,
            variable_name=variable,
            n_modes=n_modes,
            lag_order=lag_order,
            use_picontrol_baseline=True,
            cache_dir=cache_dir,
        )

        # Save to pickle
        with open(cache_file, "wb") as f:
            pickle.dump(noise_model, f)

        print(f"  ✅ Cached: {os.path.basename(cache_file)}")
        return True

    except Exception as e:
        print(f"  ❌ Failed to train {model_name} - {scenario} - {variable}: {e}")
        return False


def build_noise_cache(
    models=None,
    scenarios=None,
    variables=None,
    cache_dir=None,
    n_modes=8,
    lag_order=2,
    force=False,
):
    """Build noise model cache for specified models and scenarios."""

    # Set defaults
    if models is None:
        models = DEFAULT_MODELS
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    if variables is None:
        variables = DEFAULT_VARIABLES
    if cache_dir is None:
        cache_dir = os.path.join("streamlit", "cache", "noise_models")

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 80)
    print("METEOR Noise Model Cache Builder")
    print("=" * 80)
    print(f"\nCache directory: {cache_dir}")
    print(f"Models to cache: {len(models)}")
    print(f"Scenarios: {scenarios}")
    print(f"Variables: {variables}")
    print(f"Noise parameters: n_modes={n_modes}, lag_order={lag_order}")
    print(f"Force re-train: {force}")
    print()

    # Initialize data getter
    data_getter = initialize_data_getter(scenarios)

    # Get available models
    available_models = get_available_models(data_getter, models)

    if not available_models:
        print("❌ No models with complete data found!")
        return 1

    print(f"\nCaching noise models for {len(available_models)} models...")
    print()

    # Track statistics
    total = len(available_models) * len(scenarios) * len(variables)
    cached = 0
    trained = 0
    failed = 0

    # Cache each combination
    for model in available_models:
        print(f"\n{'='*80}")
        print(f"Model: {model}")
        print(f"{'='*80}")

        for scenario in scenarios:
            print(f"\nScenario: {scenario}")

            for variable in variables:
                # Check if already cached
                if not force and noise_cache_exists(
                    model, scenario, variable, n_modes, lag_order, cache_dir
                ):
                    print(f"  {variable}: Already cached (skip)")
                    cached += 1
                    continue

                # Train and cache
                success = train_and_cache_noise_model(
                    data_getter,
                    model,
                    scenario,
                    variable,
                    n_modes,
                    lag_order,
                    cache_dir,
                )

                if success:
                    trained += 1
                else:
                    failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total combinations: {total}")
    print(f"Already cached: {cached}")
    print(f"Newly trained: {trained}")
    print(f"Failed: {failed}")

    # Show cache size
    cache_size = get_cache_size(cache_dir)
    print(f"\nTotal cache size: {cache_size:.1f} MB")
    print(f"Cache location: {os.path.abspath(cache_dir)}")

    return 0 if failed == 0 else 1


def get_cache_size(cache_dir):
    """Get total size of cache directory in MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB


def list_available_models():
    """List all available models with complete data."""
    print("Checking available models...")
    data_getter = initialize_data_getter(DEFAULT_SCENARIOS)

    available = []
    for model in tqdm(data_getter.models, desc="Checking"):
        if data_getter.check_if_model_has_data(model):
            available.append(model)

    print(f"\n{'='*80}")
    print(f"Available Models ({len(available)} total)")
    print(f"{'='*80}")
    for model in sorted(available):
        print(f"  - {model}")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pre-build cache of trained noise models for METEOR Streamlit app",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Cache default models
  %(prog)s --models CanESM5 CESM2             # Cache specific models
  %(prog)s --scenarios ssp245                 # Cache only SSP2-4.5
  %(prog)s --variables tas                    # Cache only temperature
  %(prog)s --n-modes 10 --lag-order 3         # Custom parameters
  %(prog)s --force                            # Re-train all
  %(prog)s --list-models                      # Show available models
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help=f'Models to cache (default: {" ".join(DEFAULT_MODELS[:3])} ...)',
    )

    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["ssp126", "ssp245", "ssp370", "ssp585"],
        help="Scenarios to cache (default: ssp126 ssp245 ssp585)",
    )

    parser.add_argument(
        "--variables",
        nargs="+",
        choices=["tas", "pr"],
        help="Variables to cache (default: tas pr)",
    )

    parser.add_argument(
        "--cache-dir",
        default="streamlit/cache/noise_models",
        help="Directory to store cache files (default: streamlit/cache/noise_models)",
    )

    parser.add_argument(
        "--n-modes",
        type=int,
        default=8,
        help="Number of PCA modes for noise model (default: 8)",
    )

    parser.add_argument(
        "--lag-order",
        type=int,
        default=2,
        help="Lag order for noise model (default: 2)",
    )

    parser.add_argument(
        "--force", action="store_true", help="Force re-training even if cache exists"
    )

    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )

    args = parser.parse_args()

    # Handle list-models
    if args.list_models:
        list_available_models()
        return 0

    # Build cache
    return build_noise_cache(
        models=args.models,
        scenarios=args.scenarios,
        variables=args.variables,
        cache_dir=args.cache_dir,
        n_modes=args.n_modes,
        lag_order=args.lag_order,
        force=args.force,
    )


if __name__ == "__main__":
    sys.exit(main())
