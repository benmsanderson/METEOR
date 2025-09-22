"""
Demonstration of the new unified noise training API in METEOR.

This shows how all noise training functionality is now consolidated in
noise_generator.py while maintaining backwards compatibility.
"""

# ===========================================================================
# NEW INTUITIVE CLASS-BASED API (Recommended for new code)
# ===========================================================================

from meteor import Cmip6MeteorDataGetter, MeteorNoiseGenerator

# Setup data getter as usual
data_getter = Cmip6MeteorDataGetter(
    exps=["piControl", "abrupt-4xCO2", "historical", "ssp245"], flds=["tas", "pr"]
)

print("=== NEW CLASS-BASED API ===\n")

# Option 1: Train single model (most intuitive)
print("1. Training single noise model:")
noise_model = MeteorNoiseGenerator.train_from_cmip6(
    data_getter=data_getter,
    experiments=["historical", "ssp245"],
    model_name="CanESM5",
    variable_name="tas",
    n_modes=8,
    cache_dir="./models",
)
print("   ✅ Single model trained\n")

# Option 2: Train multiple models at once
print("2. Training multiple models:")
noise_models = MeteorNoiseGenerator.train_multiple_from_cmip6(
    data_getter=data_getter,
    experiments=["historical", "ssp245"],
    models=["CanESM5"],  # Could be multiple models
    variables=["tas", "pr"],
    n_modes=8,
    cache_dir="./models",
)
print("   ✅ Multiple models trained\n")

# Option 3: Load previously cached model
print("3. Loading cached model:")
cached_model = MeteorNoiseGenerator.load_from_cache(
    cache_dir="./models", model_name="CanESM5", variable_name="tas"
)
print("   ✅ Model loaded from cache\n")

# ===========================================================================
# BACKWARDS COMPATIBLE API (Still works for existing code)
# ===========================================================================

print("=== BACKWARDS COMPATIBLE API ===\n")

# Old function-based approach still works
from meteor import train_noise_model_from_composite

print("4. Using legacy function (still works):")
legacy_model = train_noise_model_from_composite(
    data_getter=data_getter,
    experiments=["historical", "ssp245"],
    model_name="CanESM5",
    variable_name="tas",
    n_modes=8,
)
print("   ✅ Legacy function works\n")

# Old data getter methods still work
print("5. Using data getter methods (still works):")
dg_model = data_getter.train_noise_model(
    experiments=["historical", "ssp245"],
    model="CanESM5",
    variable_name="tas",
    n_modes=8,
)
print("   ✅ Data getter method works\n")

# ===========================================================================
# UNIFIED USAGE - ALL METHODS RETURN SAME TYPE
# ===========================================================================

print("=== ALL METHODS RETURN SAME TYPE ===\n")

# All these models are identical MeteorNoiseGenerator instances
print(f"Class method result:     {type(noise_model)}")
print(f"Batch method result:     {type(noise_models['CanESM5']['tas'])}")
print(f"Cached model result:     {type(cached_model)}")
print(f"Legacy function result:  {type(legacy_model)}")
print(f"Data getter result:      {type(dg_model)}")

print("\n✅ All return MeteorNoiseGenerator instances!")

# ===========================================================================
# USAGE RECOMMENDATION FOR NEW CODE
# ===========================================================================

print("\n=== RECOMMENDED USAGE PATTERNS ===\n")

print("For NEW CODE, use the class methods:")
print("• MeteorNoiseGenerator.train_from_cmip6() - single model")
print("• MeteorNoiseGenerator.train_multiple_from_cmip6() - batch")
print("• MeteorNoiseGenerator.load_from_cache() - load saved")

print("\nFor EXISTING CODE:")
print("• No changes needed - all old functions still work")
print("• Functions now internally use the new class methods")
print("• Same performance and functionality")

print("\nBENEFITS of new API:")
print("• All noise functionality in one place (noise_generator.py)")
print("• Intuitive class-based interface")
print("• Better IDE auto-completion")
print("• Cleaner documentation structure")
print("• Easier to extend and maintain")

print("\n🎉 Training consolidation complete!")
