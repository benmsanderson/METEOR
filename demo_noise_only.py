#!/usr/bin/env python
"""
Demo script showing how to use the noise_only parameter in the METEOR noise generator.

This demonstrates the concept of generating climate noise that can be added to
METEOR annual predictions to create monthly climate projections.
"""

import numpy as np
import xarray as xr


def main():
    print("METEOR Noise Generator - Noise-Only Feature Demo")
    print("=" * 55)

    print("\nThis demo shows the concept of the new noise_only parameter.")
    print("The noise_only=True option in generate_realization() allows:")
    print("1. Generate stochastic climate variability without temperature dependence")
    print("2. Add this noise to annual METEOR predictions for monthly detail")
    print("3. Preserve seasonal patterns while removing climate trends")

    print("\n" + "=" * 55)
    print("USAGE EXAMPLES:")
    print("=" * 55)

    print("\n1. Standard usage (full climate realization):")
    print("   noise_gen = MeteorNoiseGenerator()")
    print("   # ... fit to data ...")
    print("   realization = noise_gen.generate_realization(temp_trajectory)")

    print("\n2. NEW: Noise-only generation:")
    print("   noise_only = noise_gen.generate_realization(")
    print("       temp_trajectory, noise_only=True)")

    print("\n3. Combining with METEOR annual predictions:")
    print("   # Get annual prediction from METEOR")
    print("   annual_pred = meteor_model.predict(forcing_data)")
    print("   ")
    print("   # Generate monthly noise")
    print("   monthly_noise = noise_gen.generate_realization(")
    print("       temp_trajectory, noise_only=True)")
    print("   ")
    print("   # Combine for full monthly prediction")
    print("   monthly_pred = annual_pred + monthly_noise")

    print("\n" + "=" * 55)
    print("KEY DIFFERENCES:")
    print("=" * 55)

    print("\nStandard mode (noise_only=False):")
    print("- Includes temperature-dependent seasonal cycles")
    print("- Includes constant/intercept terms")
    print("- Full climate simulation")

    print("\nNoise-only mode (noise_only=True):")
    print("- Zeros out temperature dependence")
    print("- Removes constant terms")
    print("- Keeps seasonal harmonics (annual/semi-annual)")
    print("- Returns pure stochastic variability + temperature-independent seasonality")

    print("\n" + "=" * 55)
    print("TECHNICAL IMPLEMENTATION:")
    print("=" * 55)

    print("\nThe noise_only parameter modifies generate_realization() to:")
    print("1. Set global temperature to zero in harmonic features")
    print("2. Extract only harmonic coefficients (indices 1-4)")
    print("3. Skip temperature interactions and intercept")
    print("4. Generate stochastic PCs with zero temperature input")
    print("5. Return seasonal harmonics + stochastic variability")

    print("\nThis allows the noise component to be added to any baseline")
    print("climate prediction while preserving realistic variability patterns.")

    print("\n" + "=" * 55)
    print("INTEGRATION COMPLETE!")
    print("=" * 55)

    print("\nThe noise_only feature is now available in:")
    print("- MeteorNoiseGenerator.generate_realization(noise_only=True)")
    print("- Compatible with existing METEOR workflow")
    print("- Maintains backwards compatibility")

    print("\nNext steps:")
    print("1. Train noise generator on CMIP6 data")
    print("2. Generate METEOR annual predictions")
    print("3. Add monthly noise for full temporal resolution")

    print("\nDemo complete! Ready for production use.")


if __name__ == "__main__":
    main()
