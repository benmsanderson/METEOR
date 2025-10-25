#!/usr/bin/env python
"""
METEOR Streamlit Cache Builder
===============================

Pre-download and cache training data for common models and scenarios.
This speeds up the Streamlit app by avoiding slow downloads during interactive use.

Usage:
    python streamlit/build_cache.py

Options:
    --models    Specific models to cache (default: all available)
    --scenarios Specific scenarios to cache (default: ssp245, ssp126, ssp585)
    --cache-dir Directory to store cache (default: ./streamlit/cache/training_data)
"""

import os
import sys
import argparse
import xarray as xr
from tqdm import tqdm
from meteor import Cmip6MeteorDataGetter

# Default configuration
DEFAULT_SCENARIOS = ["ssp126", "ssp245", "ssp585"]
DEFAULT_CACHE_DIR = "./streamlit/cache/training_data"

# Commonly used models (well-tested, reliable)
RECOMMENDED_MODELS = ["CanESM5", "CESM2", "UKESM1-0-LL", "ACCESS-ESM1-5", "MIROC-ES2L"]


def initialize_data_getter(scenarios):
    """Initialize CMIP6 data getter with required experiments."""
    experiments = ["piControl", "abrupt-4xCO2", "historical"] + scenarios

    dbe_map = {
        "piControl": "CMIP",
        "abrupt-4xCO2": "CMIP",
        "historical": "CMIP",
        "ssp126": "ScenarioMIP",
        "ssp245": "ScenarioMIP",
        "ssp585": "ScenarioMIP",
    }

    dbe = [dbe_map.get(exp, "CMIP") for exp in experiments]

    print(f"Initializing data getter for experiments: {experiments}")

    return Cmip6MeteorDataGetter(exps=experiments, flds=["tas", "pr"], dbe=dbe)


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


def cache_exists(model_name, scenario, cache_dir):
    """Check if training data already exists in cache."""
    required_files = [
        f"{model_name}_base_training_data.nc",
        f"{model_name}_co2x4_training_data.nc",
        f"{model_name}_{scenario}_training_data.nc",
    ]

    return all(os.path.exists(os.path.join(cache_dir, f)) for f in required_files)


def download_and_cache_training_data(data_getter, model_name, scenario, cache_dir):
    """Download training data and save to cache."""
    print(f"\n  Downloading training data for {model_name} - {scenario}...")

    try:
        # Download base and co2x4 data (shared across scenarios)
        base_path = os.path.join(cache_dir, f"{model_name}_base_training_data.nc")
        if not os.path.exists(base_path):
            print("    - piControl (base)")
            base_data = data_getter.make_meteor_training_data("base", model_name)
            base_data.to_netcdf(base_path)
        else:
            print("    - piControl (cached)")

        co2x4_path = os.path.join(cache_dir, f"{model_name}_co2x4_training_data.nc")
        if not os.path.exists(co2x4_path):
            print("    - abrupt-4xCO2")
            co2x4_data = data_getter.make_meteor_training_data("co2x4", model_name)
            co2x4_data.to_netcdf(co2x4_path)
        else:
            print("    - abrupt-4xCO2 (cached)")

        # Download scenario-specific data
        scenario_path = os.path.join(
            cache_dir, f"{model_name}_{scenario}_training_data.nc"
        )
        print(f"    - historical + {scenario}")
        sulxanom_data = data_getter.make_meteor_training_data_composite(
            ["historical", scenario], model_name
        )
        sulxanom_data.to_netcdf(scenario_path)

        print(f"  ✅ Cached {model_name} - {scenario}")
        return True

    except Exception as e:
        print(f"  ❌ Failed to cache {model_name} - {scenario}: {e}")
        return False


def build_cache(models=None, scenarios=None, cache_dir=None, force=False):
    """Build cache for specified models and scenarios."""

    # Set defaults
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 70)
    print("METEOR Streamlit Cache Builder")
    print("=" * 70)
    print(f"\nCache directory: {os.path.abspath(cache_dir)}")
    print(f"Scenarios: {', '.join(scenarios)}")

    # Initialize data getter
    data_getter = initialize_data_getter(scenarios)

    # Get available models
    if models is None:
        # Use recommended models if none specified
        available_models = get_available_models(data_getter, RECOMMENDED_MODELS)
        if not available_models:
            # Fall back to all available
            available_models = get_available_models(data_getter)
    else:
        available_models = get_available_models(data_getter, models)

    if not available_models:
        print("\n❌ No models available! Check your internet connection.")
        return 1

    print(f"\nModels to cache: {', '.join(available_models)}")

    # Calculate total work
    total_tasks = 0
    for model in available_models:
        for scenario in scenarios:
            if force or not cache_exists(model, scenario, cache_dir):
                total_tasks += 1

    if total_tasks == 0:
        print("\n✅ All requested data already cached!")
        print("\nUse --force to re-download and overwrite existing cache.")
        return 0

    print(f"\nTotal downloads needed: {total_tasks}")
    confirm = input("\nProceed with download? This may take 10-30 minutes. [y/N]: ")

    if confirm.lower() != "y":
        print("Cancelled.")
        return 1

    # Download and cache data
    print("\n" + "=" * 70)
    print("Downloading and caching data...")
    print("=" * 70)

    success_count = 0
    fail_count = 0

    for model in available_models:
        print(f"\n📦 {model}")
        for scenario in scenarios:
            if force or not cache_exists(model, scenario, cache_dir):
                if download_and_cache_training_data(
                    data_getter, model, scenario, cache_dir
                ):
                    success_count += 1
                else:
                    fail_count += 1
            else:
                print(f"  ⏭️  Skipped {scenario} (already cached)")

    # Summary
    print("\n" + "=" * 70)
    print("Cache Build Complete!")
    print("=" * 70)
    print(f"\n✅ Successfully cached: {success_count} model-scenario combinations")

    if fail_count > 0:
        print(f"❌ Failed: {fail_count}")

    print(f"\nCache location: {os.path.abspath(cache_dir)}")
    print(f"Cache size: {get_cache_size(cache_dir):.1f} MB")

    print("\n📋 Cached models:")
    for model in available_models:
        scenarios_cached = [s for s in scenarios if cache_exists(model, s, cache_dir)]
        print(f"  • {model}: {', '.join(scenarios_cached)}")

    print("\n🚀 Ready to run Streamlit app with fast loading!")
    print("   Run: streamlit run streamlit_app.py")

    return 0 if fail_count == 0 else 1


def get_cache_size(cache_dir):
    """Get total size of cache directory in MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pre-cache METEOR training data for Streamlit app",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cache recommended models for default scenarios
  python build_streamlit_cache.py
  
  # Cache specific models
  python build_streamlit_cache.py --models CanESM5 CESM2
  
  # Cache specific scenarios
  python build_streamlit_cache.py --scenarios ssp245 ssp585
  
  # Force re-download (overwrite existing)
  python build_streamlit_cache.py --force
  
  # Custom cache directory
  python streamlit/build_cache.py --cache-dir /path/to/cache
""",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to cache (default: recommended subset)",
    )

    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=DEFAULT_SCENARIOS,
        choices=["ssp126", "ssp245", "ssp370", "ssp585"],
        help=f'Scenarios to cache (default: {", ".join(DEFAULT_SCENARIOS)})',
    )

    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})",
    )

    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if cache exists"
    )

    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )

    args = parser.parse_args()

    # List models mode
    if args.list_models:
        print("Checking available models...")
        data_getter = initialize_data_getter(DEFAULT_SCENARIOS)
        models = get_available_models(data_getter)
        print(f"\n{len(models)} models available:")
        for model in models:
            print(f"  • {model}")
        return 0

    # Build cache
    return build_cache(
        models=args.models,
        scenarios=args.scenarios,
        cache_dir=args.cache_dir,
        force=args.force,
    )


if __name__ == "__main__":
    sys.exit(main())
