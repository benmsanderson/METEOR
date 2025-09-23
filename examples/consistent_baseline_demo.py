#!/usr/bin/env python3
"""
Example demonstrating the new consistent baseline handling in METEOR.

This example shows how METEOR now automatically provides consistent
piControl baselines between pattern scaling and noise generation.
"""

import numpy as np
import xarray as xr
from meteor.cmip6_meteor_data_getter import Cmip6MeteorDataGetter
from meteor.noise_generator import MeteorNoiseGenerator


def demonstrate_consistent_baselines():
    """Demonstrate automatic consistent baseline handling."""

    print("METEOR Consistent Baseline Handling Example")
    print("=" * 50)

    print("\n🆕 NEW: Automatic piControl Baseline Consistency")
    print("METEOR now automatically uses the same piControl baseline")
    print("for both pattern scaling and noise generation components.")

    # Example workflow with automatic consistency
    print("\n1. Recommended Workflow (Automatic Consistency):")
    print(
        """
# Initialize data getter
data_getter = Cmip6MeteorDataGetter(flds=['tas', 'pr'])

# Train noise model - automatically uses piControl baseline
noise_model = data_getter.train_noise_model(
    experiments=['historical', 'ssp245'],
    model='CanESM5',
    variable_name='tas'
    # use_picontrol_baseline=True by default
)
"""
    )

    print("Expected output:")
    print("   Fetched piControl baseline for CanESM5 tas: 287.456")
    print("   Using piControl baseline: 287.456")
    print("   ✅ Noise generator fitted successfully.")

    print("\n2. Pattern Scaling Integration:")
    print("   Pattern scaling has ALWAYS used piControl baseline")
    print("   Noise generation now uses the SAME piControl baseline")
    print("   → Automatic consistency between components!")

    print("\n3. ESM Comparison Benefits:")
    print(
        """
# Now much simpler - baselines are automatically consistent
meteor_projection = create_meteor_projection(...)  # Uses piControl baseline
esm_data_anomaly = esm_data - esm_picontrol_baseline

# Direct comparison - no manual baseline adjustments needed!
difference = meteor_projection - esm_data_anomaly
"""
    )


def demonstrate_backward_compatibility():
    """Show backward compatibility with legacy baseline method."""

    print("\n" + "=" * 50)
    print("Backward Compatibility")
    print("=" * 50)

    print("\n✅ All existing code continues to work unchanged!")
    print("The new behavior is automatically better and scientifically consistent.")

    print("\n📚 Legacy Mode (if needed for exact reproduction):")
    print(
        """
# Explicitly use old method if needed
noise_model = data_getter.train_noise_model(
    experiments=['historical', 'ssp245'],
    model='CanESM5',
    variable_name='tas',
    use_picontrol_baseline=False  # Use legacy 42-year baseline
)
"""
    )

    print("Expected output:")
    print("   Using first 42 years as baseline (legacy mode)")


def demonstrate_advanced_usage():
    """Show advanced usage with custom baselines."""

    print("\n" + "=" * 50)
    print("Advanced Usage: Custom Baseline Control")
    print("=" * 50)

    print("\n🎛️ Direct baseline specification:")
    print(
        """
# For maximum control, specify baseline directly
custom_baseline = 287.5  # Your specific piControl baseline

noise_gen = MeteorNoiseGenerator()
noise_gen.fit(
    monthly_data, 
    'tas',
    picontrol_baseline=custom_baseline  # Direct specification
)
"""
    )

    print("\n🌡️ Custom temperature with consistent baseline:")
    print(
        """
# Ensure your custom temperature uses same baseline as METEOR
picontrol_temp = 287.456  # piControl baseline used by METEOR
custom_temp = your_raw_temperature - picontrol_temp

noise_model = data_getter.train_noise_model(
    experiments=['historical', 'ssp245'],
    model='CanESM5',
    variable_name='tas',
    custom_global_temp=custom_temp  # Already consistent with piControl
)
"""
    )


def demonstrate_benefits():
    """Highlight the benefits of the new approach."""

    print("\n" + "=" * 50)
    print("Benefits of Consistent Baseline Implementation")
    print("=" * 50)

    benefits = [
        "🎯 Scientific Accuracy: All METEOR components use same pre-industrial baseline",
        "🔄 Automatic Consistency: No manual baseline management required",
        "📊 Simplified ESM Comparisons: Direct comparison without baseline adjustments",
        "⚡ Backward Compatible: Existing code works unchanged with better results",
        "🎛️ User Control: Legacy mode available when needed",
        "📝 Clear Feedback: Users see which baseline is being used",
        "🔧 Flexible: Custom baseline specification when needed",
    ]

    for benefit in benefits:
        print(f"\n{benefit}")

    print("\n🚀 Result: Robust, scientifically consistent climate projections!")


def migration_guide():
    """Provide migration guidance for different user types."""

    print("\n" + "=" * 50)
    print("Migration Guide")
    print("=" * 50)

    print("\n👥 Existing Users:")
    print("   ✅ No action required - your code automatically gets better results")
    print("   📈 May see slight result improvements due to consistent baselines")
    print("   🔙 Use use_picontrol_baseline=False for exact legacy reproduction")

    print("\n🆕 New Users:")
    print("   ✅ Use default settings for automatic baseline consistency")
    print("   📊 ESM comparisons are now straightforward")
    print("   🎛️ Leverage advanced options for custom workflows")

    print("\n🔬 Researchers:")
    print("   ✅ More scientifically accurate baseline handling")
    print("   📊 Easier comparison with published ESM results")
    print("   📝 Clear documentation of baseline choices")


if __name__ == "__main__":
    demonstrate_consistent_baselines()
    demonstrate_backward_compatibility()
    demonstrate_advanced_usage()
    demonstrate_benefits()
    migration_guide()

    print("\n" + "=" * 50)
    print("🎉 METEOR now provides automatic baseline consistency!")
    print("See BASELINE_TEMPERATURE_ANALYSIS.md for complete technical details.")
    print("=" * 50)
