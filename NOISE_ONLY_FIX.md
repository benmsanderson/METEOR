# METEOR Noise Generator - Noise-Only Mode Fix

## Issue Description
In the original `noise_only` mode implementation, the code was suppressing ALL temperature effects, including both:
1. Direct temperature response (global warming trend) ❌ Should be removed
2. Temperature-modulated seasonal harmonics (variability) ❌ Should be preserved

This was removing important seasonal variability patterns that change with temperature, which are a key feature of climate noise.

## Problem in Original Implementation
```python
# WRONG: Zeroed out temperature completely
zero_temp = np.zeros_like(global_temp_trajectory)
X = self._create_harmonic_features(time, zero_temp)
```

This removed:
- Direct temperature effect (intercept + t_glob) ✅ Good
- Pure seasonal harmonics (annual/semi-annual) ❌ Wrong, these were kept
- Temperature-modulated harmonics (t_glob * seasonals) ❌ Wrong, these were lost

## Solution: Selective Temperature Effect Removal

### Design Matrix Structure
The harmonic features matrix X has 9 columns:
```
Index 0: t_glob                    (direct temperature effect)
Index 1: annual_cos               (pure seasonal harmonics)
Index 2: annual_sin               
Index 3: semiannual_cos           
Index 4: semiannual_sin           
Index 5: t_glob * annual_cos      (temperature-modulated harmonics)
Index 6: t_glob * annual_sin      
Index 7: t_glob * semiannual_cos  
Index 8: t_glob * semiannual_sin  
```

### Fixed Implementation
```python
# CORRECT: Use actual temperature to preserve modulated harmonics
X = self._create_harmonic_features(time, global_temp_trajectory)

# Generate full seasonal cycle prediction
seasonal_cycle = self.seasonal_model.predict(X)

# Remove only intercept and direct temperature effect
intercept_effect = self.seasonal_model.intercept_
temp_effect = self.seasonal_model.coef_[:, 0] * global_temp_trajectory[:, np.newaxis]

# Keep everything except intercept and direct temperature
seasonal_cycle_adjusted = seasonal_cycle - intercept_effect - temp_effect
```

## What's Preserved in noise_only Mode

### ✅ Kept (Good for Variability):
- Pure seasonal harmonics (indices 1-4)
- Temperature-modulated seasonal harmonics (indices 5-8)
- Full stochastic component with temperature-dependent patterns
- Realistic seasonal variability that changes with warming

### ❌ Removed (Good for Additivity):
- Model intercept (baseline offset)
- Direct temperature effect (global warming trend)

## Benefits of the Fix

1. **Realistic Seasonal Patterns**: Temperature-modulated harmonics preserve the fact that seasonal cycles change with warming
2. **Enhanced Variability**: Richer noise patterns that reflect climate physics
3. **Temperature Dependence**: Seasonal amplitude and phase can vary with temperature
4. **Scientific Accuracy**: Maintains the relationship between warming and seasonal variability

## Use Case Example

```python
# Generate noise with temperature-dependent seasonal patterns
noise = noise_model.generate_realization(
    warming_trajectory,
    noise_only=True  # Now preserves temperature-modulated seasonality
)

# This noise can still be added to METEOR annual predictions
monthly_climate = annual_meteor_prediction + noise
```

## Validation

The fix should produce noise realizations that:
- Have zero long-term temperature trend ✅
- Show clear seasonal patterns ✅  
- Have seasonal patterns that depend on the input temperature trajectory ✅
- Can be cleanly added to annual predictions ✅

## Files Modified
- `src/meteor/noise_generator.py`: Updated `generate_realization()` method
- Documentation: Updated parameter descriptions

This fix ensures that `noise_only` mode generates realistic climate variability while maintaining the ability to combine with annual climate projections.