# ME## 🚀 Quick Start

### Option 1: Run with Auto-Caching (Slower First Time)
```bash
streamlit run streamlit/app.py
```
First run will download data on-demand (~3-5 minutes per model).

### Option 2: Pre-Cache Everything (Recommended for Deployment)
```bash
# Pre-cache training data (~10-30 minutes)
python streamlit/build_cache.py

# Pre-cache noise models (~60-150 minutes) - OPTIONAL but recommended
python streamlit/build_noise_cache.py

# Then run app (loads in seconds)
streamlit run streamlit/app.py
```

See [CACHING.md](CACHING.md) for detailed caching documentation. Interface

A lightweight web application for generating climate projections using METEOR without writing code.

## 🚀 Quick Start

### Option 1: With Pre-Caching (Recommended for Production)

**Pre-download training data to avoid slow first-run:**

```bash
# Install dependencies
pip install -r streamlit/requirements.txt

# Build cache for recommended models (takes 10-30 minutes)
python streamlit/build_cache.py

# Run the app (now much faster!)
streamlit run streamlit/app.py
```

The cache builder will:
- Download training data for common models
- Cache for SSP1-2.6, SSP2-4.5, and SSP5-8.5 scenarios
- Store in `./streamlit_cache/training_data/`
- Make subsequent app runs much faster

### Option 2: Quick Start (No Pre-Caching)

```bash
# Install and run directly
pip install -r streamlit/requirements.txt
streamlit run streamlit/app.py
```

⚠️ **Note:** First run will be slow as data downloads on-demand.

## 🌐 Deploy to Streamlit Cloud (Free)

1. **Fork this repository** to your GitHub account

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Click "New app"** and connect your GitHub repo

4. **Configure:**
   - Repository: `your-username/METEOR`
   - Branch: `main` (or your branch)
   - Main file path: `streamlit/app.py`
   - Python version: 3.9+

5. **Click Deploy!**

Your app will be live at `https://your-app-name.streamlit.app`

## 📋 Features

### Current (MVP)
- ✅ Select CMIP6 climate models
- ✅ Choose emission scenarios (SSP1-2.6, SSP2-4.5, SSP5-8.5)
- ✅ Generate temperature or precipitation projections
- ✅ Add realistic monthly variability
- ✅ Create ensemble projections (up to 20 members)
- ✅ Interactive visualization
- ✅ Download NetCDF data and plots
- ✅ Automatic caching for faster re-runs

### Potential Enhancements
- 🔄 Multi-scenario comparison view
- 🗺️ Spatial map visualization
- 📊 Regional analysis tools
- 📈 Climate indices (HDD, CDD, growing seasons)
- 💾 Save/load custom scenarios
- 🔐 User accounts and project management
- 📱 Mobile-responsive design improvements

## 🛠️ Configuration

### Pre-Caching Training Data

The `build_cache.py` script pre-downloads training data to speed up the app:

```bash
# Cache recommended models for default scenarios
python streamlit/build_cache.py

# Cache specific models
python streamlit/build_cache.py --models CanESM5 CESM2 UKESM1-0-LL

# Cache specific scenarios
python streamlit/build_cache.py --scenarios ssp245 ssp585

# List all available models
python streamlit/build_cache.py --list-models

# Force re-download (overwrite cache)
python streamlit/build_cache.py --force

# Custom cache directory
python streamlit/build_cache.py --cache-dir /path/to/cache
```

**What gets cached:**
- `piControl` (pre-industrial baseline) - shared across scenarios
- `abrupt-4xCO2` (CO₂ forcing patterns) - shared across scenarios  
- `historical + scenario` composite data - scenario-specific

**Cache location:**
- Default: `./streamlit_cache/training_data/`
- Typical size: ~500MB-2GB depending on number of models

**Benefits:**
- ✅ Fast app startup (seconds vs minutes)
- ✅ Better user experience
- ✅ Suitable for production deployment
- ✅ Can pre-cache overnight or during setup

### Model Cache
By default, trained models are cached in `./streamlit_cache/` to speed up subsequent runs.

To clear cache:
```bash
rm -rf streamlit_cache/
```

### Advanced Settings
Users can adjust:
- **PCA Modes** (4-20): Number of spatial patterns retained
- **VARX Lag Order** (1-4): Temporal autocorrelation complexity
- **Ensemble Size** (1-20): Number of stochastic realizations

## 📖 How It Works

1. **Pattern Scaling**: Learns climate response patterns from CMIP6 models
2. **Noise Generation**: Adds realistic month-to-month variability using PCA/VARX
3. **Fast Emulation**: Generates projections 1000x faster than running full ESM
4. **Uncertainty**: Multiple ensemble members show range of possible outcomes

## 🎯 Target Users

- **Climate Researchers**: Quick scenario exploration without coding
- **Policy Analysts**: Generate climate projections for impact assessment
- **Educators**: Demonstrate climate modeling concepts interactively
- **Students**: Learn about climate projections hands-on

## 🐛 Troubleshooting

### "Connection Error" or "Data Not Available"
- Check internet connection (app fetches data from CMIP6 cloud storage)
- Try a different model - some may have incomplete data

### "Out of Memory"
- Reduce number of realizations
- Reduce PCA modes
- Clear cache and restart

### App is Slow
- First run trains models (can take 2-3 minutes)
- Subsequent runs use cached models (much faster)
- Consider deploying on Streamlit Cloud for better resources

## 📝 Technical Details

### Architecture
```
User Interface (Streamlit)
    ↓
METEOR Pattern Scaling
    ↓
CMIP6 Data Getter (Cloud Access)
    ↓
Monthly Noise Generator (PCA/VARX)
    ↓
Results & Visualization
```

### Dependencies
- **Streamlit**: Web interface framework
- **METEOR**: Core climate emulation engine
- **CMIP6 Data**: Accessed via GCSFS (Google Cloud Storage)
- **Ciceroscm**: Scenario data and forcing calculations

### Performance
- Initial model training: 1-3 minutes
- Cached model projection: 10-30 seconds
- Memory usage: ~2-4 GB for typical run

## 🤝 Contributing

To enhance the web interface:

1. **Add new features** in `streamlit/app.py`
2. **Test locally** with `streamlit run streamlit/app.py`
3. **Submit PR** with description of changes

Ideas for contributions:
- Additional visualization types (maps, regional plots)
- Export formats (CSV, Excel, GeoTIFF)
- Custom scenario builder
- Educational tooltips and documentation
- Performance optimizations

## 📚 Resources

- [METEOR Documentation](../README.md)
- [Streamlit Docs](https://docs.streamlit.io)
- [CMIP6 Data](https://esgf-node.llnl.gov/projects/cmip6/)
- [SSP Scenarios](https://www.carbonbrief.org/explainer-how-shared-socioeconomic-pathways-explore-future-climate-change)

## 📄 License

Same as METEOR - see main repository LICENSE file.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/benmsanderson/METEOR/issues)
- **Questions**: Contact METEOR maintainers
- **Streamlit-specific**: Check [Streamlit Community Forum](https://discuss.streamlit.io)
