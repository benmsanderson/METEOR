#!/usr/bin/env python3
"""
Interactive script for bulk caching of METEOR data.

This script allows users to pre-cache CMIP6 data for multiple models, scenarios,
and variables using METEOR's caching system. This is useful for ensuring data
is available offline or for batch processing workflows.

Usage:
    python scripts/bulk_cache_data.py

The script will prompt for user input with sensible defaults for comprehensive
METEOR analysis workflows.
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

try:
    from meteor.cmip6_meteor_data_getter import Cmip6MeteorDataGetter
except ImportError:
    print("Error: Could not import METEOR. Please ensure METEOR is installed.")
    print("Try: pip install meteor")
    sys.exit(1)


# Default configurations for comprehensive METEOR analysis
DEFAULT_SCENARIOS = [
    "piControl",
    "abrupt-4xCO2",
    "historical",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp585",
]

DEFAULT_MODELS = [
    "ACCESS-ESM1-5",
    "AWI-CM-1-1-MR",
    "BCC-CSM2-MR",
    "CAMS-CSM1-0",
    "CAS-ESM2-0",
    "CESM2",
    "CMCC-ESM2",
    "CNRM-ESM2-1",
    "CanESM5",
    "EC-Earth3-Veg",
    "FGOALS-f3-L",
    "GFDL-ESM4",
    "GISS-E2-1-H",
    "INM-CM5-0",
    "KACE-1-0-G",
    "MCM-UA-1-0",
    "MIROC-ES2L",
    "MPI-ESM1-2-HR",
    "NorESM2-MM",
    "UKESM1-0-LL",
]

DEFAULT_VARIABLES = ["pr", "tas"]


def save_checkpoint(
    checkpoint_file: Path,
    scenarios: List[str],
    models: List[str],
    variables: List[str],
    completed: List[tuple],
    failed_combinations: List,
):
    """Save progress checkpoint to file."""
    checkpoint_data = {
        "scenarios": scenarios,
        "models": models,
        "variables": variables,
        "completed": completed,
        "failed_combinations": failed_combinations,
        "timestamp": time.time(),
    }

    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=2)


def load_checkpoint(checkpoint_file: Path):
    """Load progress checkpoint from file."""
    if not checkpoint_file.exists():
        return None

    try:
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def get_pending_combinations(
    scenarios: List[str],
    models: List[str],
    variables: List[str],
    completed: List[tuple],
) -> List[tuple]:
    """Get list of combinations that haven't been completed yet."""
    all_combinations = [(s, m, v) for s in scenarios for m in models for v in variables]
    completed_set = set(completed)
    return [combo for combo in all_combinations if combo not in completed_set]


def parse_flexible_list(input_list: List[str]) -> List[str]:
    """
    Parse a list that may contain comma-separated values into individual items.

    Handles both space-separated args: --models A B C
    And comma-separated args: --models A,B,C
    And mixed: --models A,B C,D

    Args:
        input_list: List of strings from argparse

    Returns:
        Flattened list of individual items
    """
    if not input_list:
        return []

    result = []
    for item in input_list:
        # Split on commas and strip whitespace
        sub_items = [s.strip() for s in item.split(",")]
        # Filter out empty strings
        result.extend([s for s in sub_items if s])

    return result


def get_default_cache_location() -> str:
    """Get the default METEOR cache location."""
    # Create a temporary data getter to get the default cache location
    print("✅ Checking available models for validation...")
    try:
        temp_getter = Cmip6MeteorDataGetter(
            exps=["piControl"],
            flds=["tas"],
            enable_cache=True,  # Enable cache to get cache_dir
            enable_compression=False,  # No compression needed for validation
        )
        return temp_getter.cache_dir
    except Exception:
        # Fallback if we can't determine the default
        return "<repository>/.cache/cmip6 (default)"


def get_user_input_list(
    prompt: str, default_list: List[str], item_type: str = "items"
) -> List[str]:
    """
    Get a list input from user with defaults.

    Args:
        prompt: Prompt message to display
        default_list: Default list to use if user presses enter
        item_type: Description of what the items are (for help text)

    Returns:
        List of user-selected items
    """
    print(f"\n{prompt}")
    print(f"Default {item_type}: {', '.join(default_list)}")
    print(f"Enter custom {item_type} (comma-separated) or press Enter for defaults:")
    print("You can also enter 'all' to use all available options.")

    user_input = input("> ").strip()

    if not user_input:
        return default_list
    elif user_input.lower() == "all":
        if item_type == "scenarios":
            # For scenarios, 'all' means the default comprehensive list
            return default_list
        else:
            # For models/variables, we'd need to query available options
            print(f"Using default comprehensive {item_type} list...")
            return default_list
    else:
        # Parse comma-separated input
        items = [item.strip() for item in user_input.split(",")]
        return [item for item in items if item]  # Remove empty strings


def get_user_boolean_input(prompt: str, default: bool = False) -> bool:
    """
    Get a yes/no input from user with default.

    Args:
        prompt: Prompt message to display
        default: Default value if user presses enter

    Returns:
        Boolean value based on user input
    """
    default_text = "Y/n" if default else "y/N"
    user_input = input(f"{prompt} ({default_text}): ").strip().lower()

    if not user_input:
        return default
    elif user_input in ["y", "yes"]:
        return True
    elif user_input in ["n", "no"]:
        return False
    else:
        print("Please enter 'y' for yes or 'n' for no.")
        return get_user_boolean_input(prompt, default)


def get_cache_directory() -> Optional[Path]:
    """Get cache directory from user or use default."""
    default_location = get_default_cache_location()

    print("\nCache Directory:")
    print(f"Default METEOR cache location: {default_location}")
    print("Enter custom cache directory path or press Enter for default:")

    user_input = input("> ").strip()

    if not user_input:
        print(f"Using default cache location: {default_location}")
        return None  # Use METEOR's default
    else:
        cache_path = Path(user_input)
        if not cache_path.exists():
            create = input(f"Directory {cache_path} doesn't exist. Create it? (y/n): ")
            if create.lower() in ["y", "yes"]:
                cache_path.mkdir(parents=True, exist_ok=True)
                print(f"Created cache directory: {cache_path}")
                return cache_path
            else:
                print(f"Using default cache location: {default_location}")
                return None
        print(f"Using custom cache location: {cache_path}")
        return cache_path


def confirm_caching_plan(
    scenarios: List[str],
    models: List[str],
    variables: List[str],
) -> bool:
    """Display caching plan and get user confirmation."""
    print("\n" + "=" * 60)
    print("CACHING PLAN SUMMARY")
    print("=" * 60)
    print(f"Scenarios ({len(scenarios)}): {', '.join(scenarios)}")
    print(f"Models ({len(models)}): {', '.join(models)}")
    print(f"Variables ({len(variables)}): {', '.join(variables)}")
    total_combinations = len(scenarios) * len(models) * len(variables)
    print(
        f"Total combinations: {len(scenarios)} × {len(models)} × {len(variables)} = {total_combinations}"
    )

    print("\nThis will download and cache monthly data for all combinations.")
    print("(Annual data is computed on-demand from monthly cache)")
    print("Depending on your internet connection, this may take considerable time.")
    print("=" * 60)

    confirm = input("Proceed with caching? (y/n): ")
    return confirm.lower() in ["y", "yes"]


def cache_data_combination(
    scenario: str,
    model: str,
    variable: str,
    cache_dir: Optional[Path],
    current: int,
    total: int,
    failed_combinations: List,
    enable_compression: bool = True,
    compression_level: int = 6,
) -> bool:
    """
    Cache data for a specific scenario/model/variable combination.

    Returns:
        True if successful, False if failed
    """
    # Progress bar
    progress = current / total * 100
    bar_length = 30
    filled_length = int(bar_length * current / total)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)

    print(
        f"\r[{bar}] {progress:5.1f}% ({current}/{total}) "
        f"Checking {scenario} - {model} - {variable}... ",
        end="",
        flush=True,
    )

    try:
        # Add a timestamp for debugging
        start_time = time.time()

        # Initialize data getter for this specific scenario/variable combo
        # This avoids the issue where models are filtered out if they don't have ALL scenarios
        compression_level_to_use = compression_level if enable_compression else 6
        if cache_dir:
            data_getter = Cmip6MeteorDataGetter(
                cache_dir=str(cache_dir),
                exps=[scenario],
                flds=[variable],
                enable_cache=True,  # Explicitly enable caching for bulk download
                enable_compression=enable_compression,
                compression_level=compression_level_to_use,
            )
        else:
            data_getter = Cmip6MeteorDataGetter(
                exps=[scenario],
                flds=[variable],
                enable_cache=True,  # Explicitly enable caching for bulk download
                enable_compression=enable_compression,
                compression_level=compression_level_to_use,
            )

            # Check if this specific data type is already cached (fast check)
            # NEW OPTIMIZATION: Check for variable-specific monthly cache instead of training data
            # This allows flexible variable combinations and avoids redundant storage
        # Check if monthly data is already cached
        cache_exists = data_getter.is_cached(
            "get_single_var_mod_data_monthly", scenario, variable, model
        )

        if cache_exists:
            elapsed = time.time() - start_time
            print(
                f"\r[{bar}] {progress:5.1f}% ({current}/{total}) "
                f"Checking {scenario} - {model} - {variable}... ✓ Cached (monthly) ({elapsed:.1f}s)"
            )
            return True

        # Data not cached - need to download
        print(
            f"\r[{bar}] {progress:5.1f}% ({current}/{total}) "
            f"Caching {scenario} - {model} - {variable} (monthly)... ",
            end="",
            flush=True,
        )

        # Add periodic heartbeat for long downloads
        heartbeat_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        heartbeat_active = True
        heartbeat_counter = [0]

        def heartbeat():
            while heartbeat_active:
                if time.time() - start_time > 2:  # Show after 2 seconds
                    char = heartbeat_chars[heartbeat_counter[0] % len(heartbeat_chars)]
                    print(
                        f"\r[{bar}] {progress:5.1f}% ({current}/{total}) "
                        f"Caching {scenario} - {model} - {variable} (monthly)... {char}",
                        end="",
                        flush=True,
                    )
                    heartbeat_counter[0] += 1
                time.sleep(0.3)

        # Start heartbeat in background for long operations
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()

        try:
            # Cache monthly data (raw source)
            # Annual data is computed on-demand (40-80x faster than caching)
            monthly_data = data_getter.get_single_var_mod_data_monthly(
                scenario, variable, model
            )

            # For validation, also ensure the data can be used for training
            # (but don't cache the training data itself - computed on-the-fly)
            _ = data_getter.make_meteor_training_data(
                scenario, model, variable, monthly=True
            )

        except KeyboardInterrupt:
            # Allow user to interrupt gracefully
            print("\n\nInterrupted by user. Progress saved.")
            heartbeat_active = False
            raise
        except Exception as download_error:
            # Handle download-specific errors
            heartbeat_active = False
            elapsed = time.time() - start_time
            error_msg = str(download_error)

            # Store detailed error for summary
            failed_combinations.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "variable": variable,
                    "reason": error_msg,
                }
            )

            # Truncate very long error messages for display
            if len(error_msg) > 100:
                error_msg = error_msg[:97] + "..."
            print(
                f"\r[{bar}] {progress:5.1f}% ({current}/{total}) "
                f"Caching {scenario} - {model} - {variable} (monthly)... ✗ Error: {error_msg}"
            )
            return False
        finally:
            heartbeat_active = False

        # Check if data was successfully retrieved
        if monthly_data is None:
            failed_combinations.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "variable": variable,
                    "reason": "No data available",
                }
            )
            print(
                f"\r[{bar}] {progress:5.1f}% ({current}/{total}) "
                f"Caching {scenario} - {model} - {variable} (monthly)... ✗ No data"
            )
            return False

        # Report success
        elapsed = time.time() - start_time
        print(
            f"\r[{bar}] {progress:5.1f}% ({current}/{total}) "
            f"Caching {scenario} - {model} - {variable}... ✓ Downloaded (monthly) ({elapsed:.1f}s)"
        )
        return True
    except Exception as e:
        heartbeat_active = False
        elapsed = time.time() - start_time if "start_time" in locals() else 0
        error_msg = str(e)

        # Store detailed error for summary
        failed_combinations.append(
            {
                "scenario": scenario,
                "model": model,
                "variable": variable,
                "reason": error_msg,
            }
        )

        # Truncate very long error messages for display
        if len(error_msg) > 100:
            error_msg = error_msg[:97] + "..."
        print(
            f"\r[{bar}] {progress:5.1f}% ({current}/{total}) "
            f"Caching {scenario} - {model} - {variable}... ✗ Error: {error_msg} ({elapsed:.1f}s)"
        )
        return False


def check_model_experiments(model_name: str, variables: List[str] = None) -> dict:
    """
    Check what experiments are available for a specific model.

    Args:
        model_name: Name of the model to check
        variables: List of variables to check (defaults to ['tas'])

    Returns:
        Dictionary with experiment availability information
    """
    if variables is None:
        variables = ["tas"]

    # Test each default experiment individually
    test_experiments = DEFAULT_SCENARIOS
    available_experiments = []
    failed_experiments = []

    print(f"\n🔍 Checking experiment availability for {model_name}...")

    for exp in test_experiments:
        for var in variables:
            try:
                # Test with minimal initialization
                temp_getter = Cmip6MeteorDataGetter(
                    exps=[exp],
                    flds=[var],
                    enable_cache=False,  # No cache needed for validation
                    enable_compression=False,  # No compression needed for validation
                )

                # Check if model is in the available models list
                if model_name in temp_getter.models:
                    if exp not in available_experiments:
                        available_experiments.append(exp)
                        print(f"  ✓ {exp} ({var})")
                else:
                    if exp not in failed_experiments:
                        failed_experiments.append(exp)
                        print(f"  ✗ {exp} ({var}) - Model not found in experiment")

            except Exception as e:
                if exp not in failed_experiments:
                    failed_experiments.append(exp)
                    print(f"  ✗ {exp} ({var}) - Error: {str(e)[:50]}...")

    return {
        "model": model_name,
        "available_experiments": available_experiments,
        "failed_experiments": failed_experiments,
        "total_tested": len(test_experiments),
    }


def check_experiment_models(experiment: str, variables: List[str] = None) -> dict:
    """
    Check what models are available for a specific experiment.

    Args:
        experiment: Name of the experiment to check
        variables: List of variables to check (defaults to ['tas'])

    Returns:
        Dictionary with model availability information
    """
    if variables is None:
        variables = ["tas"]

    print(f"\n🔍 Checking model availability for {experiment}...")

    try:
        temp_getter = Cmip6MeteorDataGetter(
            exps=[experiment],
            flds=variables,
            enable_cache=False,  # No cache needed for model listing
            enable_compression=False,  # No compression needed for validation
        )
        available_models = temp_getter.models

        print(f"  Found {len(available_models)} models with {experiment} data:")
        for i, model in enumerate(available_models[:10]):  # Show first 10
            print(f"    {i + 1:2d}. {model}")
        if len(available_models) > 10:
            print(f"    ... and {len(available_models) - 10} more")

        return {
            "experiment": experiment,
            "available_models": available_models,
            "total_models": len(available_models),
        }

    except Exception as e:
        print(f"  ✗ Error checking {experiment}: {e}")
        return {
            "experiment": experiment,
            "available_models": [],
            "total_models": 0,
            "error": str(e),
        }


def main():
    """Main interactive caching workflow."""
    parser = argparse.ArgumentParser(description="Bulk cache METEOR data")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Use all defaults without prompting",
    )
    parser.add_argument(
        "--scenarios", nargs="+", help="Scenarios to cache", default=None
    )
    parser.add_argument("--models", nargs="+", help="Models to cache", default=None)
    parser.add_argument(
        "--variables", nargs="+", help="Variables to cache", default=None
    )
    parser.add_argument(
        "--check-model",
        type=str,
        help="Check what experiments are available for a specific model",
    )
    parser.add_argument(
        "--check-experiment",
        type=str,
        help="Check what models are available for a specific experiment",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from a checkpoint file (e.g., --resume bulk_cache_checkpoint.json)",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="bulk_cache_checkpoint.json",
        help="Checkpoint file name (default: bulk_cache_checkpoint.json)",
    )
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Disable netCDF4/zlib compression for cached files (compression is enabled by default)",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=6,
        choices=range(1, 10),
        help="Compression level for netCDF4/zlib compression (1-9, default: 6). "
        "Higher values = better compression but slower performance",
    )

    args = parser.parse_args()  # Handle diagnostic commands
    if args.check_model:
        variables = parse_flexible_list(args.variables) or DEFAULT_VARIABLES
        check_model_experiments(args.check_model, variables)
        return

    if args.check_experiment:
        variables = parse_flexible_list(args.variables) or DEFAULT_VARIABLES
        check_experiment_models(args.check_experiment, variables)
        return

    # Handle resume functionality
    checkpoint_file = Path(args.checkpoint_file)
    checkpoint_data = None
    completed_combinations = []

    if args.resume:
        resume_file = Path(args.resume)
        checkpoint_data = load_checkpoint(resume_file)
        if checkpoint_data:
            print(f"📄 Resuming from checkpoint: {args.resume}")
            scenarios = checkpoint_data["scenarios"]
            models = checkpoint_data["models"]
            variables = checkpoint_data["variables"]
            completed_combinations = [
                tuple(combo) for combo in checkpoint_data["completed"]
            ]
            failed_combinations = checkpoint_data["failed_combinations"]
            print(f"   Already completed: {len(completed_combinations)} combinations")
            print(f"   Previous failures: {len(failed_combinations)} combinations")
        else:
            print(f"❌ Could not load checkpoint file: {args.resume}")
            return
    elif checkpoint_file.exists():
        # Check if there's an existing checkpoint file
        checkpoint_data = load_checkpoint(checkpoint_file)
        if checkpoint_data:
            if args.non_interactive:
                # In non-interactive mode, automatically start fresh unless --resume was specified
                print(f"📄 Found existing checkpoint file: {checkpoint_file}")
                print(
                    f"   {len(checkpoint_data.get('completed', []))} combinations already completed"
                )
                print(
                    "🗑️ Non-interactive mode: Starting fresh (use --resume to continue from checkpoint)"
                )
                checkpoint_data = None  # Clear checkpoint data to use normal flow
            else:
                # Interactive mode: ask user
                resume_choice = input(
                    f"\n📄 Found existing checkpoint file: {checkpoint_file}\n"
                    f"   {len(checkpoint_data.get('completed', []))} combinations already completed\n"
                    f"Resume from checkpoint? (y/n): "
                )
                if resume_choice.lower() in ["y", "yes"]:
                    scenarios = checkpoint_data["scenarios"]
                    models = checkpoint_data["models"]
                    variables = checkpoint_data["variables"]
                    completed_combinations = [
                        tuple(combo) for combo in checkpoint_data["completed"]
                    ]
                    failed_combinations = checkpoint_data.get("failed_combinations", [])
                    print("✅ Resuming from checkpoint...")
                else:
                    print("🗑️ Starting fresh (checkpoint file will be overwritten)")
                    checkpoint_data = None  # Clear checkpoint data to use normal flow

    print("METEOR Bulk Data Caching Tool")
    print("=" * 40)
    print("This tool will help you pre-cache CMIP6 data for METEOR analysis.")
    print("You can customize which scenarios, models, and variables to cache,")
    print("or use the defaults for comprehensive climate analysis.")

    # If not resuming, get user preferences normally
    if not checkpoint_data:
        if not args.non_interactive:
            # Interactive mode - get user preferences
            scenarios = (
                parse_flexible_list(args.scenarios)
                if args.scenarios
                else get_user_input_list(
                    "Select scenarios to cache:", DEFAULT_SCENARIOS, "scenarios"
                )
            )

            models = (
                parse_flexible_list(args.models)
                if args.models
                else get_user_input_list(
                    "Select models to cache:", DEFAULT_MODELS, "models"
                )
            )

            variables = (
                parse_flexible_list(args.variables)
                if args.variables
                else get_user_input_list(
                    "Select variables to cache:", DEFAULT_VARIABLES, "variables"
                )
            )

            cache_dir = get_cache_directory()

            # Confirm the plan
            if not confirm_caching_plan(scenarios, models, variables):
                print("Caching cancelled by user.")
                return
        else:
            # Non-interactive mode - use defaults or command line args
            scenarios = parse_flexible_list(args.scenarios) or DEFAULT_SCENARIOS
            models = parse_flexible_list(args.models) or DEFAULT_MODELS
            variables = parse_flexible_list(args.variables) or DEFAULT_VARIABLES

            cache_dir = None

            print("Non-interactive mode:")
            print(f"  Scenarios ({len(scenarios)}): {', '.join(scenarios)}")
            print(
                f"  Models ({len(models)}): {', '.join(models[:3])}{'...' if len(models) > 3 else ''}"
            )
            print(f"  Variables ({len(variables)}): {', '.join(variables)}")
            print("  Data type: Monthly only (annual computed on-demand)")
            print(
                f"  Total combinations: {len(scenarios) * len(models) * len(variables)}"
            )

            # Show default cache location
            default_cache = get_default_cache_location()
            print(f"  Cache location: {default_cache}")

        completed_combinations = []
        failed_combinations = []
    else:
        # Using checkpoint data, set cache_dir to None (use default)
        cache_dir = None

    # Get pending combinations (skip already completed ones)
    if checkpoint_data:
        pending_combinations = get_pending_combinations(
            scenarios, models, variables, completed_combinations
        )
        print("\n📋 Resuming progress:")
        print(f"   Total combinations: {len(scenarios) * len(models) * len(variables)}")
        print(f"   Already completed: {len(completed_combinations)}")
        print(f"   Remaining to process: {len(pending_combinations)}")
    else:
        pending_combinations = [
            (s, m, v) for s in scenarios for m in models for v in variables
        ]

    # Begin caching process
    total_combinations = len(scenarios) * len(models) * len(variables)
    remaining_combinations = len(pending_combinations)

    print("\nStarting bulk caching process...")
    # Begin caching process
    total_combinations = len(scenarios) * len(models) * len(variables)
    remaining_combinations = len(pending_combinations)

    print("\nStarting bulk caching process...")
    print(f"Will process {remaining_combinations} combinations total")
    print("Progress: [████████████████████████████████] 100%")
    print("Legend: ✓ = Success, ✗ = Failed/No data")
    print("-" * 80)

    successful = len(completed_combinations)  # Count previously completed as successful
    failed = len(failed_combinations)  # Count previous failures
    current = len(completed_combinations)  # Start from where we left off
    start_time = time.time()

    for scenario, model, variable in pending_combinations:
        current += 1
        success = cache_data_combination(
            scenario,
            model,
            variable,
            cache_dir,
            current,
            total_combinations,
            failed_combinations,
            not args.no_compression,  # enable_compression
            args.compression_level,  # compression_level
        )
        if success:
            successful += 1
            completed_combinations.append((scenario, model, variable))
        else:
            failed += 1

        # Save checkpoint every 5 items and at the end
        if current % 5 == 0 or current == total_combinations:
            save_checkpoint(
                checkpoint_file,
                scenarios,
                models,
                variables,
                completed_combinations,
                failed_combinations,
            )

        # Show estimated time remaining every 10 items
        if current % 10 == 0 or current == total_combinations:
            elapsed = time.time() - start_time
            if current > 0:
                avg_time = elapsed / current
                remaining = (total_combinations - current) * avg_time
                eta_mins = remaining / 60
                # Progress checkpoint
                print(
                    f"\n    Progress: {current}/{total_combinations} completed ({successful} successful, {failed} failed)"
                )
                print(f"    ETA: {eta_mins:.1f} minutes remaining")
                print(f"    Checkpoint saved: {checkpoint_file}")

    # Clean up checkpoint file on successful completion
    if checkpoint_file.exists() and failed == 0:
        checkpoint_file.unlink()
        print("\n✅ All combinations completed successfully. Checkpoint file removed.")

    # Final newline after progress bar
    print()
    print("-" * 80)

    # Summary
    print("\n" + "=" * 60)
    print("CACHING COMPLETE")
    print("=" * 60)
    print(f"Total combinations processed: {total_combinations}")
    print(f"Successfully cached: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful / total_combinations * 100:.1f}%")

    if failed > 0:
        print(f"\n📋 DETAILED FAILURE SUMMARY ({failed} failures)")
        print("=" * 60)

        # Group failures by reason
        failure_reasons = {}
        for failure in failed_combinations:
            reason = failure["reason"]
            if reason not in failure_reasons:
                failure_reasons[reason] = []
            failure_reasons[reason].append(
                f"{failure['scenario']} - {failure['model']} - {failure['variable']}"
            )

        for reason, combinations in failure_reasons.items():
            print(f"\n🔍 Reason: {reason}")
            print(f"   Affected combinations ({len(combinations)}):")
            for combo in combinations[:5]:  # Show first 5
                print(f"     • {combo}")
            if len(combinations) > 5:
                print(f"     ... and {len(combinations) - 5} more")

        print(
            "\n💡 Note: Failures are normal when model/scenario/variable combinations"
        )
        print("are not available in the CMIP6 archive. Consider using models and")
        print("scenarios with broader data availability for comprehensive analysis.")

    print("\n🎉 Cached data is now available for offline METEOR analysis!")


if __name__ == "__main__":
    main()
