# Critical Fix: Year Offset in METEOR Predictions

## Problem Identified

The Streamlit app was showing **minimal warming until 2050** because `predict_from_combined_experiment()` returns predictions starting from **year 1750**, but we need to plot from **year 1850** onwards.

## Root Cause

Looking at the notebook plotting code:
```python
axs[j,i].plot(years_to_plot, 
              prpatt.global_mean(patterns_anomsulf[m][s][fld]).values[100:],  # <- Note the [100:]
              color="green", label="METEOR_aer")
```

The **`[100:]` slice** skips the first 100 years (1750-1850) to align with the actual historical/scenario data timeline.

## Timeline Explanation

- **1750-1850**: Spinup period in METEOR predictions (100 years)
- **1850-2014**: Historical period
- **2015-2100**: Future scenario period (SSP)

The emissions/concentrations data fed to `predict_from_combined_experiment()` starts from 1750, so the output array has:
- Index 0 = Year 1750
- Index 100 = Year 1850
- Index 250 = Year 2000
- Index 350 = Year 2100

## Fix Applied

### 1. Updated `plot_ensemble()` function:
```python
# OLD (WRONG):
annual_global = prpatt.global_mean(annual_pred).values
years = 1850 + np.arange(len(annual_global))

# NEW (CORRECT):
annual_global = prpatt.global_mean(annual_pred).values[100:]  # Skip first 100 years
years = 1850 + np.arange(len(annual_global))
```

### 2. Updated noise generation:
```python
# OLD (WRONG):
global_temp = prpatt.global_mean(annual_pred).values
monthly_temp = np.repeat(global_temp, 12)

# NEW (CORRECT):
global_temp = prpatt.global_mean(annual_pred).values[100:]  # Skip 1750-1850
monthly_temp = np.repeat(global_temp, 12)
```

### 3. Added verbose logging:
```python
log_verbose("Projection generated", {
    "shape": str(annual_pred.shape),
    "total_years": int(annual_pred.shape[-1]),
    "note": "Prediction starts at year 1750",
    "year_1750_temp": f"{global_mean_pred[0]:.2f}K",
    "year_1850_temp": f"{global_mean_pred[100]:.2f}K",
    "year_2100_temp": f"{global_mean_pred[-1]:.2f}K",
    "warming_1850_2100": f"{global_mean_pred[-1] - global_mean_pred[100]:.2f}K"
})
```

## Expected Behavior After Fix

For SSP2-4.5 scenario with CanESM5:
- ✅ Warming should start from 1850 
- ✅ Gradual increase through historical period (1850-2014)
- ✅ Continued warming through 21st century
- ✅ Total warming 1850-2100: ~3-4°C for SSP2-4.5

## Testing

Run the Streamlit app and check:
1. Enable "Verbose output" mode
2. Select SSP2-4.5 scenario
3. Verify warming starts immediately from 1850
4. Check verbose log shows correct year offsets
5. Verify warming trend annotation shows realistic values

## Files Modified

- `streamlit_app.py`: 
  - `plot_ensemble()` function
  - Noise generation section
  - Verbose logging for predictions
