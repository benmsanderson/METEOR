# METEOR Streamlit Caching System

## 🎯 Problem Solved

**Before:** First-time data downloads in Streamlit app take 2-5 minutes per model
**After:** With pre-caching, app loads in seconds

## 📦 What Gets Cached

The caching system stores two types of data:

### 1. Training Data Cache (NetCDF files)
**Location:** `streamlit_cache/training_data/`

**Files per model-scenario combination:**
```
CanESM5_base_training_data.nc           # piControl baseline (shared)
CanESM5_co2x4_training_data.nc          # CO₂ forcing patterns (shared)
CanESM5_ssp245_training_data.nc         # historical + SSP2-4.5
CanESM5_ssp585_training_data.nc         # historical + SSP5-8.5
```

**File sizes:** ~50-200 MB each
**Total size:** ~500MB-2GB for 5 models × 3 scenarios

### 2. Pattern Model Cache (Pickle files)  
**Location:** `streamlit_cache/`

**Files:**
```
CanESM5_ssp245_pattern.pkl              # Trained METEOR pattern
```

**File sizes:** ~5-10 MB each
**Created automatically** when training patterns

### 3. Noise Model Cache (Pickle files)
**Location:** `streamlit_cache/noise_models/`

**Files:**
```
CanESM5_ssp245_tas_n8_lag2_noise.pkl    # Noise model for temperature
CanESM5_ssp245_pr_n8_lag2_noise.pkl     # Noise model for precipitation
```

**File sizes:** ~10-50 MB each
**Total size:** ~500MB-1GB for 5 models × 3 scenarios × 2 variables
**Created automatically** or pre-cached with `build_noise_cache.py`
**Location:** `streamlit_cache/noise_models/`

**Files:**
```
CanESM5_tas_noise_model.pkl             # Temperature noise model
CanESM5_pr_noise_model.pkl              # Precipitation noise model
```

**Created automatically** by noise training

## 🔄 Caching Flow

### Without Pre-Caching (Slow)
```
User clicks "Run METEOR"
    ↓
Download piControl data (30-60 sec)
    ↓
Download abrupt-4xCO2 data (30-60 sec)
    ↓
Download historical+scenario data (60-120 sec)
    ↓
Train pattern model (10-20 sec)
    ↓
Train noise model (20-30 sec)
    ↓
Generate projection (5-10 sec)
    
TOTAL: 3-5 minutes
```

### With Pre-Caching (Fast)
```
User clicks "Run METEOR"
    ↓
Load cached training data (2-5 sec)
    ↓
Load/train pattern model (5-10 sec)
    ↓
Load/train noise model (10-15 sec)
    ↓
Generate projection (5-10 sec)
    
TOTAL: 30-60 seconds
```

### With Full Cache (Fastest)
```
User clicks "Run METEOR"
    ↓
Load cached training data (2 sec)
    ↓
Load cached pattern model (1 sec)
    ↓
Load cached noise model (1 sec)
    ↓
Generate projection (5-10 sec)
    
TOTAL: 10-15 seconds
```

## 🚀 Usage

### Pre-Cache Training Data
```bash
python build_streamlit_cache.py
```

This downloads and caches:
- **Models:** CanESM5, CESM2, UKESM1-0-LL, ACCESS-ESM1-5, MIROC-ES2L
- **Scenarios:** SSP1-2.6, SSP2-4.5, SSP5-8.5
- **Time:** ~10-30 minutes total

### Pre-Cache Noise Models (NEW!)
```bash
python build_noise_cache.py
```

This trains and caches noise models:
- **Models:** Same as above (5 models)
- **Scenarios:** SSP1-2.6, SSP2-4.5, SSP5-8.5
- **Variables:** tas (temperature), pr (precipitation)
- **Parameters:** n_modes=8, lag_order=2 (default)
- **Time:** ~2-5 minutes per model-scenario-variable (~60-150 min total)

### Custom Noise Cache Options
```bash
# Cache specific models only
python build_noise_cache.py --models CanESM5 CESM2

# Cache specific scenario
python build_noise_cache.py --scenarios ssp245

# Cache only temperature
python build_noise_cache.py --variables tas

# Custom parameters
python build_noise_cache.py --n-modes 10 --lag-order 3

# Force re-train
python build_noise_cache.py --force

# List available models
python build_noise_cache.py --list-models
```

### Pre-Cache Specific Models/Scenarios (Training Data)
```bash
python build_streamlit_cache.py --models CanESM5 CESM2
```

### Pre-Cache Specific Scenarios
```bash
python build_streamlit_cache.py --scenarios ssp245
```

### List Available Models
```bash
python build_streamlit_cache.py --list-models
```

### Force Re-Download
```bash
python build_streamlit_cache.py --force
```

## 🎨 Code Architecture

### Streamlit App Functions

```python
# Load cached training data if available
def load_cached_training_data(model_name, scenario, cache_dir):
    """Check for and load NetCDF training data from cache."""
    # Returns None if not cached → triggers download
    # Returns dict of xarray datasets if cached
    
# Save training data to cache
def save_training_data_cache(test_data, model_name, scenario, cache_dir):
    """Save downloaded data as NetCDF for future use."""
    # Saves: base, co2x4, sulxanom datasets
    
# Train pattern model (with caching)
@st.cache_resource
def train_pattern_model(_data_getter, model_name, scenario):
    """Load cached data if available, download if not."""
    # 1. Try load cached training data
    # 2. If not cached, download and save
    # 3. Try load cached pattern model
    # 4. If not cached, train and save
```

### Cache Builder Script

```python
# Main function
def build_cache(models, scenarios, cache_dir):
    """Download and cache data for specified models/scenarios."""
    # 1. Initialize data getter
    # 2. Check available models
    # 3. For each model-scenario pair:
    #    - Download piControl (if not cached)
    #    - Download abrupt-4xCO2 (if not cached)
    #    - Download historical+scenario
    #    - Save all as NetCDF
```

## 📊 Performance Comparison

| Scenario | First Run | Cached Training | Full Cache (with noise) |
|----------|-----------|-----------------|--------------------------|
| CanESM5 ssp245 (no noise) | 3-5 min | 30-60 sec | 10-15 sec |
| CanESM5 ssp245 (20 realizations) | 4-6 min | 90-120 sec | 15-20 sec |
| CESM2 ssp585 (no noise) | 3-5 min | 30-60 sec | 10-15 sec |
| CESM2 ssp585 (20 realizations) | 4-6 min | 90-120 sec | 15-20 sec |

**Speed improvements:**
- Training data cache: ~5-10x faster
- Pattern model cache: ~2-3x faster  
- Noise model cache: ~30-60x faster (60 sec → 1 sec)

## 🗂️ Cache Directory Structure

```
streamlit_cache/
├── training_data/                    # NetCDF training data
│   ├── CanESM5_base_training_data.nc
│   ├── CanESM5_co2x4_training_data.nc
│   ├── CanESM5_ssp245_training_data.nc
│   ├── CanESM5_ssp585_training_data.nc
│   ├── CESM2_base_training_data.nc
│   └── ...
├── noise_models/                     # Trained noise models
│   ├── CanESM5_tas_noise_model.pkl
│   ├── CanESM5_pr_noise_model.pkl
│   └── ...
└── CanESM5_ssp245_pattern.pkl       # Trained pattern models
```

## 🧹 Cache Management

### Check Cache Size
```bash
du -sh streamlit_cache/
```

### Clear All Cache
```bash
rm -rf streamlit_cache/
```

### Clear Only Training Data
```bash
rm -rf streamlit_cache/training_data/
```

### Clear Only Pattern Models
```bash
rm streamlit_cache/*_pattern.pkl
```

### Selective Cleanup
```python
# Remove old scenarios
rm streamlit_cache/training_data/*_ssp126_*

# Remove specific model
rm streamlit_cache/training_data/CanESM5_*
```

## 🔧 Customization

### Add New Scenarios
Edit `build_streamlit_cache.py`:
```python
DEFAULT_SCENARIOS = ["ssp126", "ssp245", "ssp370", "ssp585"]
```

### Change Recommended Models
Edit `build_streamlit_cache.py`:
```python
RECOMMENDED_MODELS = [
    "CanESM5",
    "CESM2",
    "Your-Favorite-Model"
]
```

### Change Cache Location
```bash
python build_streamlit_cache.py --cache-dir /data/meteor_cache
```

Then update `streamlit_app.py`:
```python
if 'cache_dir' not in st.session_state:
    st.session_state.cache_dir = "/data/meteor_cache"
```

## 🚀 Deployment Recommendations

### Local Development
- Pre-cache 1-2 models for testing
- Use default cache directory

### Production Deployment
- Pre-cache all recommended models
- Run cache builder during deployment setup
- Consider persistent storage volume for cache
- Cache location: `/app/streamlit_cache` (Streamlit Cloud)

### Streamlit Cloud
1. Include `build_streamlit_cache.py` in repo
2. Add cache directory to `.gitignore`
3. Consider GitHub Actions to build cache
4. Or: Accept first-run slowness, subsequent runs fast

## 💡 Best Practices

1. **Pre-cache before deployment**
   ```bash
   python build_streamlit_cache.py
   git add streamlit_cache/training_data/*.nc
   git commit -m "Add cached training data"
   ```

2. **Monitor cache size**
   - Limit to essential models
   - Clean old scenarios periodically

3. **Version control**
   - Add `streamlit_cache/*.pkl` to `.gitignore` (ephemeral)
   - Optionally commit `.nc` files (if < 100MB each)

4. **User communication**
   - Show cache status in debug mode
   - Display "Loading from cache" messages
   - Provide cache builder instructions

## 🐛 Troubleshooting

### Cache Not Being Used
- Check file paths match exactly
- Verify cache directory exists
- Enable debug mode to see cache status

### Downloads Still Slow
- Check internet connection
- Verify CMIP6 cloud access
- Try different models

### Cache Files Corrupt
```bash
# Force re-download
python build_streamlit_cache.py --force
```

### Out of Disk Space
```bash
# Clean cache and rebuild with fewer models
rm -rf streamlit_cache/
python build_streamlit_cache.py --models CanESM5
```

## 📈 Future Enhancements

- [ ] Automatic cache validation
- [ ] Cache expiry/versioning
- [ ] Compressed cache format
- [ ] Shared cache server
- [ ] Cache progress UI in app
- [ ] Smart cache invalidation
- [ ] Download resume capability

---

**Summary:** The caching system transforms the Streamlit app from "slow first run" to "fast always" by pre-downloading and storing CMIP6 training data locally.
