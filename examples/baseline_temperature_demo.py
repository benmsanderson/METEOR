#!/usr/bin/env python3
"""
Demonstration of baseline temperature handling in METEOR components.

This script shows how different METEOR components handle baseline temperatures
and provides examples of ensuring consistency for ESM comparisons.
"""

import numpy as np
import xarray as xr
from meteor.cmip6_meteor_data_getter import Cmip6MeteorDataGetter
from meteor.noise_generator import MeteorNoiseGenerator


def demonstrate_baseline_handling():
    """Demonstrate how baselines are handled in different METEOR components."""

    print("METEOR Baseline Temperature Handling Demonstration")
    print("=" * 55)

    # Example with synthetic data to show the principles
    print("\n1. Pattern Scaling Baseline Handling:")
    print("   - Uses FULL piControl experiment as baseline")
    print("   - All experiments become anomalies relative to piControl mean")
    print("   - Code: anomaly = experiment - piControl.mean()")

    print("\n2. Noise Generation Baseline Handling:")
    print("   - Uses FIRST 42 YEARS of training data as baseline")
    print("   - Global temperature: t_glob = t_mean - t_mean[:500].mean()")
    print("   - Where 500 months = 42 years")

    # Demonstrate the difference
    print("\n3. Potential Inconsistency Example:")

    # Synthetic temperature data representing historical + future (1850-2100)
    years = np.arange(1850, 2101)
    months = len(years) * 12

    # Synthetic piControl baseline (constant pre-industrial conditions)
    pi_control_temp = 14.0  # °C

    # Synthetic historical+future warming
    warming_trajectory = (
        pi_control_temp
        + 0.01 * (years - 1850)
        + 0.2 * np.sin(2 * np.pi * (years - 1850) / 20)
    )
    monthly_warming = np.repeat(warming_trajectory, 12) + 0.5 * np.sin(
        2 * np.pi * np.arange(months) / 12
    )

    # Pattern scaling baseline (piControl mean)
    pattern_baseline = pi_control_temp
    pattern_anomaly = monthly_warming - pattern_baseline

    # Noise generation baseline (first 42 years mean)
    noise_baseline = monthly_warming[: 42 * 12].mean()
    noise_temperature = monthly_warming - noise_baseline

    print(f"   piControl baseline: {pattern_baseline:.2f} °C")
    print(f"   First 42-year baseline: {noise_baseline:.2f} °C")
    print(f"   Difference: {noise_baseline - pattern_baseline:.2f} °C")

    return {
        "pattern_baseline": pattern_baseline,
        "noise_baseline": noise_baseline,
        "monthly_warming": monthly_warming,
        "pattern_anomaly": pattern_anomaly,
        "noise_temperature": noise_temperature,
    }


def demonstrate_consistent_baseline_approach():
    """Show how to ensure consistent baselines for ESM comparisons."""

    print("\n" + "=" * 55)
    print("Consistent Baseline Approach for ESM Comparisons")
    print("=" * 55)

    print("\nRecommended Workflow:")
    print("1. Determine your comparison baseline (usually piControl climatology)")
    print("2. Apply this same baseline to both METEOR and ESM data")
    print("3. Compare the resulting anomalies")

    # Example code patterns
    print("\nExample Code Pattern:")
    print(
        """
# Step 1: Extract piControl baseline from ESM
esm_picontrol = esm_data.sel(experiment='piControl')
pi_baseline = esm_picontrol.mean(dim='year')

# Step 2: Create custom temperature relative to piControl
custom_temp = your_temperature_data - pi_baseline.mean()

# Step 3: Train noise model with consistent baseline
noise_model = data_getter.train_noise_model(
    experiments=['historical', 'ssp245'],
    model='your_model',
    variable_name='tas',
    custom_global_temp=custom_temp  # Consistent with ESM baseline
)

# Step 4: Compare outputs using same baseline
meteor_anomaly = meteor_output - pi_baseline
esm_anomaly = esm_output - pi_baseline
difference = meteor_anomaly - esm_anomaly
"""
    )


def demonstrate_noise_only_usage():
    """Demonstrate proper baseline handling for noise-only realizations."""

    print("\n" + "=" * 55)
    print("Noise-Only Realization Baseline Handling")
    print("=" * 55)

    print("\nWhen using noise_only=True:")
    print("- Removes intercept and direct temperature effects")
    print("- Keeps temperature-modulated seasonal harmonics")
    print("- Removes baseline from seasonal cycle")
    print("- Output represents variability around zero")

    print("\nProper usage:")
    print(
        """
# 1. Generate annual projection (with proper baseline)
annual_projection = meteor_patterns.predict(temperature_trajectory)

# 2. Convert to monthly (preserves baseline)
monthly_projection = annual_projection.to_monthly()

# 3. Generate noise component (relative to training baseline)
noise_realization = noise_model.generate_realization(
    temperature_trajectory, 
    noise_only=True
)

# 4. Combine (baselines are handled consistently)
full_monthly = monthly_projection + noise_realization
"""
    )


def demonstrate_custom_temperature_baselines():
    """Show how to handle baselines when using custom temperature."""

    print("\n" + "=" * 55)
    print("Custom Temperature Baseline Considerations")
    print("=" * 55)

    print("\nWhen providing custom_global_temp, consider:")
    print("1. What baseline does your temperature use?")
    print("2. Is it consistent with your intended comparison?")
    print("3. Does it match the pattern scaling baseline?")

    print("\nExample scenarios:")
    print(
        """
# Scenario A: Temperature from observations (absolute values)
obs_temp = load_observational_temperature()  # e.g., 14.5°C global mean
# Remove piControl baseline to match pattern scaling
custom_temp_A = obs_temp - 14.0  # Assuming 14.0°C piControl

# Scenario B: Temperature from climate model (already anomalies)
model_temp_anomaly = load_model_temperature_anomaly()  # Already relative to baseline
custom_temp_B = model_temp_anomaly  # Already consistent

# Scenario C: Temperature with specific smoothing
raw_temp = load_temperature_data()
smoothed_temp = apply_20year_smoothing(raw_temp)
# Ensure consistent baseline
custom_temp_C = smoothed_temp - picontrol_baseline
"""
    )


if __name__ == "__main__":
    # Run demonstrations
    baseline_info = demonstrate_baseline_handling()
    demonstrate_consistent_baseline_approach()
    demonstrate_noise_only_usage()
    demonstrate_custom_temperature_baselines()

    print("\n" + "=" * 55)
    print("Key Takeaways:")
    print("=" * 55)
    print("1. Pattern scaling uses piControl climatology as baseline")
    print("2. Noise generation uses first 42 years of training data as baseline")
    print("3. For ESM comparisons, use piControl baseline for both METEOR and ESM")
    print("4. For noise-only analysis, account for the training data baseline")
    print("5. Custom temperature should use baseline consistent with comparison goals")
    print("\nSee BASELINE_TEMPERATURE_ANALYSIS.md for detailed documentation.")
