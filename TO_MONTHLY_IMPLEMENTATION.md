# METEOR to_monthly Method Implementation

## Overview
Added a new `to_monthly()` method to the `MeteorPatternScaling` class that converts annual climate predictions to monthly intervals, enabling easy combination with monthly noise generator output.

## Method Signature
```python
def to_monthly(self, annual_prediction, start_year=None):
```

## Parameters
- `annual_prediction` (xr.DataArray): Annual climate prediction with dimensions (time, lat, lon)
- `start_year` (int, optional): Starting year for monthly time coordinate. If None, uses integer indices.

## Returns
- `xr.DataArray`: Monthly climate prediction with dimensions (month, lat, lon) where each annual value is repeated 12 times

## Implementation Details

### Key Features
1. **Dimension Expansion**: Converts (time, lat, lon) to (month, lat, lon)
2. **Value Repetition**: Each annual value repeated exactly 12 times for months
3. **Coordinate Handling**: Properly creates monthly time coordinates
4. **Metadata Preservation**: Maintains original attributes with conversion notes

### Algorithm
1. Validate input DataArray has 'time' dimension
2. Calculate dimensions: n_years → n_months (12x expansion)  
3. Use `np.repeat()` to expand values along time axis
4. Create monthly coordinate array
5. Construct new DataArray with updated dimensions and coordinates
6. Add conversion metadata to attributes

### Error Handling
- Validates input is xarray DataArray
- Ensures 'time' dimension exists
- Maintains data integrity through coordinate mapping

## Usage Examples

### Basic Usage
```python
# Get annual prediction from METEOR
annual_pred = pattern_model.predict_from_combined_experiment(
    emissions_data, concentrations_data, ["tas"]
)["tas"]

# Convert to monthly
monthly_pred = pattern_model.to_monthly(annual_pred, start_year=1850)
```

### Integration with Noise Generator
```python
# Step 1: Get annual prediction
annual_prediction = pattern_model.predict_from_combined_experiment(...)["tas"]

# Step 2: Convert to monthly
monthly_annual = pattern_model.to_monthly(annual_prediction, start_year=1850)

# Step 3: Generate monthly noise (noise-only mode)
monthly_noise = noise_model.generate_realization(
    warming_trajectory, 
    noise_only=True
)

# Step 4: Combine for complete monthly climate prediction
monthly_climate = monthly_annual + monthly_noise
```

## Benefits

1. **Seamless Integration**: Direct compatibility with noise generator output
2. **Temporal Consistency**: Maintains annual climate trends while adding monthly detail
3. **Ensemble Capability**: Enables multiple realizations by varying noise seed
4. **Flexible Workflow**: Can be used with any annual climate prediction
5. **Backwards Compatible**: No changes to existing METEOR functionality

## Technical Validation

- ✅ Syntax validation passed
- ✅ Proper dimension handling verified  
- ✅ Value repetition confirmed correct
- ✅ Coordinate system properly maintained
- ✅ Integration tested with noise generator

## File Location
- Implementation: `src/meteor/meteor.py` (line ~727)
- Test demonstrations: `notebooks/CMIP6_mmdemo_with_archiving.ipynb`

## Workflow Integration

This method completes the monthly climate prediction workflow:

```
Annual Forcing → METEOR Pattern Scaling → Annual Prediction
                                               ↓
Monthly Noise ← PCA/VARX Modeling ← CMIP6 Training Data
      ↓                                      ↓
Monthly Noise (noise_only=True)    →    to_monthly()
                     ↓                       ↓
               Complete Monthly Climate Prediction
```

The implementation provides a clean, efficient way to bridge annual climate projections with monthly variability, enabling high-resolution climate simulations suitable for impact assessment and adaptation planning.