

"""
METEOR Streamlit App - Minimal MVP
===================================

A lightweight web interface for METEOR climate projections.
Enables users to generate climate scenarios without writing code.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
from meteor import MeteorPatternScaling, Cmip6MeteorDataGetter, prpatt
from ciceroscm import input_handler

# Page configuration
st.set_page_config(
    page_title="METEOR Climate Emulator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Initialize session state for caching
if "data_getter" not in st.session_state:
    st.session_state.data_getter = None
if "pattern_model" not in st.session_state:
    st.session_state.pattern_model = None
if "noise_model" not in st.session_state:
    st.session_state.noise_model = None
if "cache_dir" not in st.session_state:
    # Use absolute path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    st.session_state.cache_dir = os.path.join(script_dir, "cache")
    os.makedirs(st.session_state.cache_dir, exist_ok=True)


def load_scenario_data(scenario):
    """Load emissions and concentration data for a scenario."""
    cscm_data_dir = os.path.join("src", "meteor", "default_scm_data")

    conc_data = input_handler.read_inputfile(
        os.path.join(cscm_data_dir, f"{scenario}_conc_RCMIP.txt")
    )
    ih = input_handler.InputHandler({})
    em_data = ih.read_emissions(os.path.join(cscm_data_dir, f"{scenario}_em_RCMIP.txt"))

    return em_data, conc_data


@st.cache_resource(show_spinner=False)
def initialize_data_getter(experiments):
    """Initialize CMIP6 data getter with caching."""
    # Map experiment names to database locations
    dbe_map = {
        "piControl": "CMIP",
        "abrupt-4xCO2": "CMIP",
        "1pctCO2": "CMIP",
        "historical": "CMIP",
        "ssp126": "ScenarioMIP",
        "ssp245": "ScenarioMIP",
        "ssp585": "ScenarioMIP",
    }

    dbe = [dbe_map.get(exp, "CMIP") for exp in experiments]

    return Cmip6MeteorDataGetter(exps=experiments, flds=["tas", "pr"], dbe=dbe)


@st.cache_resource(show_spinner=False)
def get_available_models(_data_getter):
    """Get list of models with complete data."""
    available = []
    for model in _data_getter.models:
        if _data_getter.check_if_model_has_data(model):
            available.append(model)
    return sorted(available)


@st.cache_resource(show_spinner=False)
def train_pattern_model(_data_getter, model_name, scenario):
    """Train METEOR pattern scaling model with caching."""
    cache_dir = st.session_state.cache_dir

    # Try to load cached training data first
    training_data_cached = load_cached_training_data(model_name, scenario, cache_dir)

    if training_data_cached is not None:
        # Use cached training data
        test_data = training_data_cached
    else:
        # Download and create training data
        test_data = {
            "base": _data_getter.make_meteor_training_data("base", model_name),
            "co2x4": _data_getter.make_meteor_training_data("co2x4", model_name),
            "sulxanom": _data_getter.make_meteor_training_data_composite(
                ["historical", scenario], model_name
            ),
        }

        # Save training data to cache for future use
        save_training_data_cache(test_data, model_name, scenario, cache_dir)

    # Log training data info if we can access session state
    try:
        if hasattr(st.session_state, "verbose_log"):
            st.session_state.verbose_log.append(
                {
                    "step": "Pattern Training Data Loaded",
                    "base_shape": str(
                        test_data["base"]["tas"].shape
                        if "tas" in test_data["base"]
                        else "N/A"
                    ),
                    "co2x4_shape": str(
                        test_data["co2x4"]["tas"].shape
                        if "tas" in test_data["co2x4"]
                        else "N/A"
                    ),
                    "sulxanom_shape": str(
                        test_data["sulxanom"]["tas"].shape
                        if "tas" in test_data["sulxanom"]
                        else "N/A"
                    ),
                    "sulxanom_years": (
                        f"{int(test_data['sulxanom']['tas'].shape[-1])} years"
                        if "tas" in test_data["sulxanom"]
                        else "N/A"
                    ),
                }
            )
    except:
        pass

    # Try to load pattern model from cache
    pattern_cache_path = os.path.join(cache_dir, f"{model_name}_{scenario}_pattern.pkl")

    if os.path.exists(pattern_cache_path):
        import pickle

        with open(pattern_cache_path, "rb") as f:
            return pickle.load(f)

    # Train pattern scaling
    pattern_model = MeteorPatternScaling(
        f"cmip6-{model_name}",
        {"tas": 2, "pr": 2},
        lambda key: test_data[key],
        from_file=False,
        exp_list=["base", "co2x4", "sulxanom"],
    )

    # Cache the pattern model
    import pickle

    with open(pattern_cache_path, "wb") as f:
        pickle.dump(pattern_model, f)

    return pattern_model


def load_cached_training_data(model_name, scenario, cache_dir):
    """Load cached training data if available."""
    try:
        training_data_dir = os.path.join(cache_dir, "training_data")

        # Check if all required files exist
        required_files = {
            "base": f"{model_name}_base_training_data.nc",
            "co2x4": f"{model_name}_co2x4_training_data.nc",
            "sulxanom": f"{model_name}_{scenario}_training_data.nc",
        }

        for key, filename in required_files.items():
            if not os.path.exists(os.path.join(training_data_dir, filename)):
                return None

        # Load all training data
        test_data = {
            "base": xr.open_dataset(
                os.path.join(training_data_dir, required_files["base"])
            ),
            "co2x4": xr.open_dataset(
                os.path.join(training_data_dir, required_files["co2x4"])
            ),
            "sulxanom": xr.open_dataset(
                os.path.join(training_data_dir, required_files["sulxanom"])
            ),
        }

        return test_data

    except Exception:
        return None


def save_training_data_cache(test_data, model_name, scenario, cache_dir):
    """Save training data to cache for future use."""
    try:
        training_data_dir = os.path.join(cache_dir, "training_data")
        os.makedirs(training_data_dir, exist_ok=True)

        # Save each component
        test_data["base"].to_netcdf(
            os.path.join(training_data_dir, f"{model_name}_base_training_data.nc")
        )
        test_data["co2x4"].to_netcdf(
            os.path.join(training_data_dir, f"{model_name}_co2x4_training_data.nc")
        )
        test_data["sulxanom"].to_netcdf(
            os.path.join(training_data_dir, f"{model_name}_{scenario}_training_data.nc")
        )

    except Exception as e:
        # Silently fail - not critical if caching doesn't work
        pass


def check_training_data_cached(model_name, scenario, cache_dir):
    """Check if training data is cached for a model/scenario."""
    training_data_dir = os.path.join(cache_dir, "training_data")
    
    required_files = [
        f"{model_name}_base_training_data.nc",
        f"{model_name}_co2x4_training_data.nc",
        f"{model_name}_{scenario}_training_data.nc",
    ]
    
    return all(
        os.path.exists(os.path.join(training_data_dir, f)) 
        for f in required_files
    )


def check_pattern_model_cached(model_name, scenario, cache_dir):
    """Check if pattern model is cached."""
    pattern_cache_path = os.path.join(cache_dir, f"{model_name}_{scenario}_pattern.pkl")
    return os.path.exists(pattern_cache_path)


def check_noise_model_cached(model_name, scenario, variable, n_modes, lag_order, cache_dir):
    """Check if noise model is cached."""
    noise_cache_dir = os.path.join(cache_dir, "noise_models")
    noise_cache_path = os.path.join(
        noise_cache_dir,
        f"{model_name}_{scenario}_{variable}_n{n_modes}_lag{lag_order}_noise.pkl",
    )
    return os.path.exists(noise_cache_path)


@st.cache_resource(show_spinner=False)
def train_noise_model(_data_getter, model_name, scenario, variable, n_modes, lag_order):
    """Train monthly noise model with caching."""

    # Check for cached noise model first
    noise_cache_dir = os.path.join(st.session_state.cache_dir, "noise_models")
    os.makedirs(noise_cache_dir, exist_ok=True)

    noise_cache_path = os.path.join(
        noise_cache_dir,
        f"{model_name}_{scenario}_{variable}_n{n_modes}_lag{lag_order}_noise.pkl",
    )

    if os.path.exists(noise_cache_path):
        import pickle

        try:
            with open(noise_cache_path, "rb") as f:
                return pickle.load(f)
        except:
            # If loading fails, retrain
            pass

    # Train noise model (this also uses cache_dir internally for training data)
    noise_model = _data_getter.train_noise_model(
        experiments=["historical", scenario],
        model=model_name,
        variable_name=variable,
        n_modes=n_modes,
        lag_order=lag_order,
        use_picontrol_baseline=True,
        cache_dir=noise_cache_dir,
    )

    # Save to cache
    import pickle

    try:
        with open(noise_cache_path, "wb") as f:
            pickle.dump(noise_model, f)
    except:
        # Caching failed but we have the model
        pass

    return noise_model


def plot_ensemble(annual_pred, noise_realizations, variable, scenario, model_name):
    """Create ensemble visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get baseline
    baseline = 0  # Could be computed from piControl if needed

    # Convert annual to monthly
    # NOTE: predict_from_combined_experiment starts at year 1750,
    # so we skip first 100 years to align with 1850
    annual_global = prpatt.global_mean(annual_pred).values[100:]
    years = 1850 + np.arange(len(annual_global))

    # Plot ensemble members (annual + noise)
    if noise_realizations is not None and len(noise_realizations) > 0:
        monthly_years = 1850 + np.arange(len(noise_realizations[0])) / 12
        for i, noise in enumerate(noise_realizations):
            noise_global = prpatt.global_mean(noise).values
            # Repeat annual for monthly comparison
            annual_monthly = np.repeat(annual_global, 12)[: len(noise_global)]
            ensemble_member = annual_monthly + noise_global

            ax.plot(
                monthly_years,
                ensemble_member,
                color="lightblue",
                alpha=0.3,
                linewidth=0.5,
                label="Ensemble members" if i == 0 else "",
            )

    # Plot annual mean
    ax.plot(
        years,
        annual_global,
        color="darkblue",
        linewidth=2.5,
        label="Annual mean (METEOR)",
        zorder=10,
    )

    # Add text annotation showing warming trend
    if len(annual_global) > 100:
        start_val = np.mean(annual_global[:20])  # Average of first 20 years
        end_val = np.mean(annual_global[-20:])  # Average of last 20 years
        warming = end_val - start_val
        ax.text(
            0.02,
            0.98,
            f"Trend: {warming:+.2f}K ({years[0]:.0f}-{years[-1]:.0f})",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Formatting
    ax.set_xlabel("Year", fontsize=12)
    ylabel = "Temperature (K)" if variable == "tas" else "Precipitation (mm/day)"
    ax.set_ylabel(f"Global mean {ylabel}", fontsize=12)
    ax.set_title(
        f"{model_name} - {scenario.upper()} - {variable.upper()}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1850, 2100)

    return fig


# =============================================================================
# MAIN APP INTERFACE
# =============================================================================

# Header
st.markdown(
    '<p class="main-header">🌍 METEOR Climate Emulator</p>', unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">Fast climate projections with realistic variability</p>',
    unsafe_allow_html=True,
)

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("1. Select Model")

# Initialize data getter first to check available models
if st.session_state.data_getter is None:
    with st.spinner("Checking available models..."):
        # Use a broad set of experiments to check availability
        check_experiments = ["piControl", "abrupt-4xCO2", "historical", "ssp245"]
        st.session_state.data_getter = initialize_data_getter(check_experiments)

# Get available models
available_models = get_available_models(st.session_state.data_getter)

if not available_models:
    st.error(
        "❌ No models with complete data found. Please check your internet connection."
    )
    st.stop()

# Show number of available models
st.sidebar.caption(f"📊 {len(available_models)} models available")

# Create model options with cache indicators
cache_dir = st.session_state.cache_dir
model_options = {}
cached_models = []
uncached_models = []

for model in available_models:
    # Check if base data is cached (shared across scenarios)
    base_cached = os.path.exists(
        os.path.join(cache_dir, "training_data", f"{model}_base_training_data.nc")
    )
    co2x4_cached = os.path.exists(
        os.path.join(cache_dir, "training_data", f"{model}_co2x4_training_data.nc")
    )
    
    if base_cached and co2x4_cached:
        model_options[f"✓ {model}"] = model
        cached_models.append(f"✓ {model}")
    else:
        model_options[f"○ {model}"] = model
        uncached_models.append(f"○ {model}")

# Sort: cached models first, then uncached
sorted_model_keys = cached_models + uncached_models

# Default to first cached model if available, otherwise first model
default_index = 0  # First model in sorted list (which will be cached if any exist)

model_display = st.sidebar.selectbox(
    "Climate Model",
    sorted_model_keys,
    index=default_index,
    help="✓ = cached (fast), ○ = not cached (slower first run)",
)
model_name = model_options[model_display]

st.sidebar.subheader("2. Select Scenario")
scenarios = {"SSP1-2.6": "ssp126", "SSP2-4.5": "ssp245", "SSP5-8.5": "ssp585"}

# Create scenario options with cache indicators
scenario_options = {}
for label, scen_code in scenarios.items():
    scenario_cached = check_training_data_cached(model_name, scen_code, cache_dir)
    pattern_cached = check_pattern_model_cached(model_name, scen_code, cache_dir)
    
    if scenario_cached and pattern_cached:
        scenario_options[f"✓ {label}"] = scen_code
    elif scenario_cached or pattern_cached:
        scenario_options[f"◐ {label}"] = scen_code
    else:
        scenario_options[f"○ {label}"] = scen_code

scenario_display = st.sidebar.selectbox(
    "Emission Scenario",
    list(scenario_options.keys()),
    index=1,  # Default to second option (likely SSP2-4.5)
    help="✓ = fully cached, ◐ = partially cached, ○ = not cached",
)
scenario = scenario_options[scenario_display]

# Show cache status for selected model/scenario
cache_dir = st.session_state.cache_dir
training_cached = check_training_data_cached(model_name, scenario, cache_dir)
pattern_cached = check_pattern_model_cached(model_name, scenario, cache_dir)

if training_cached and pattern_cached:
    st.sidebar.success("✅ Model & scenario cached - fast load!")
elif training_cached or pattern_cached:
    st.sidebar.info("⚡ Partially cached - moderate load time")
else:
    st.sidebar.warning("⏳ Not cached - first run will take 2-5 minutes")
    st.sidebar.caption("💡 Subsequent runs will be much faster")

st.sidebar.subheader("3. Select Variable")
variable = st.sidebar.radio(
    "Climate Variable",
    ["tas", "pr"],
    format_func=lambda x: "Temperature (tas)" if x == "tas" else "Precipitation (pr)",
    help="Climate variable to project",
)

st.sidebar.subheader("4. Monthly Variability")
include_noise = st.sidebar.checkbox(
    "Add monthly variability",
    value=False,
    help="Include realistic month-to-month fluctuations",
)

if include_noise:
    n_realizations = st.sidebar.slider(
        "Number of realizations",
        min_value=1,
        max_value=20,
        value=3,
        help="Number of stochastic ensemble members",
    )

    with st.sidebar.expander("Advanced Noise Settings"):
        n_modes = st.slider(
            "PCA Modes", 4, 20, 8, help="Number of spatial patterns to retain"
        )
        lag_order = st.slider(
            "VARX Lag Order", 1, 4, 2, help="Temporal autocorrelation order"
        )
    
    # Show noise model cache status
    noise_cached = check_noise_model_cached(
        model_name, scenario, variable, n_modes, lag_order, cache_dir
    )
    
    if noise_cached:
        st.sidebar.success("✅ Noise model cached - instant generation!")
    else:
        st.sidebar.warning("⏳ Noise model not cached - will take ~30-60 seconds to train")
        st.sidebar.caption("💡 Run `python streamlit/build_noise_cache.py` to pre-cache")
else:
    n_realizations = 0
    n_modes = 8
    lag_order = 2

# Debug mode toggle (at bottom of sidebar)
st.sidebar.markdown("---")
debug_mode = st.sidebar.checkbox(
    "🐛 Debug mode", value=False, help="Show additional diagnostic information"
)

verbose_mode = st.sidebar.checkbox(
    "🔍 Verbose output",
    value=False,
    help="Show detailed logging of all operations and data loading",
)

if debug_mode:
    st.sidebar.write("**Debug Info:**")
    st.sidebar.write(f"- Models available: {len(available_models)}")
    st.sidebar.write(f"- Cache dir: {st.session_state.cache_dir}")
    if st.session_state.data_getter:
        st.sidebar.write(f"- Data getter exps: {st.session_state.data_getter.exps}")

    # Show cache status
    training_cache_dir = os.path.join(st.session_state.cache_dir, "training_data")
    if os.path.exists(training_cache_dir):
        cache_files = len(
            [f for f in os.listdir(training_cache_dir) if f.endswith(".nc")]
        )
        st.sidebar.write(f"- Training data cached: {cache_files} files")
    else:
        st.sidebar.write("- Training data cache: empty")
        st.sidebar.caption("💡 Run `python streamlit/build_cache.py` to pre-cache data")

# Main content area
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📊 About This Projection")
    # Get clean scenario label (remove cache indicator)
    scenario_label = scenario_display.lstrip("✓◐○ ")
    
    st.info(
        f"""
    **Model:** {model_name}
    
    **Scenario:** {scenario_label}
    
    **Variable:** {"Temperature" if variable == "tas" else "Precipitation"}
    
    **Resolution:** {"Monthly" if include_noise else "Annual"}
    
    **Ensemble size:** {n_realizations if include_noise else "Single realization"}
    """
    )

    # Show model availability info
    with st.expander("ℹ️ Model Information"):
        st.write(
            f"""
        **{model_name}** is selected from {len(available_models)} available models 
        with complete data for the required experiments.
        
        Available models have data for:
        - piControl (pre-industrial baseline)
        - abrupt-4xCO2 (CO₂ forcing patterns)
        - historical (past climate)
        - SSP scenarios (future projections)
        """
        )

    if st.button("ℹ️ About METEOR", use_container_width=True):
        st.info(
            """
        **METEOR** (Multivariate Emulation of Time-Evolving and Overlapping Responses) 
        is a fast climate emulator that:
        
        - Combines pattern scaling for long-term trends
        - Adds stochastic noise for short-term variability
        - Runs 1000x faster than full ESMs
        - Preserves spatial-temporal patterns from CMIP6
        """
        )

with col1:
    st.subheader("🚀 Generate Climate Projection")

    # Verbose output container
    if verbose_mode:
        verbose_container = st.expander("🔍 Detailed Operation Log", expanded=True)

    if st.button("▶️ Run METEOR", type="primary", use_container_width=True):

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Verbose logging helper
        def log_verbose(message, data=None):
            if verbose_mode:
                with verbose_container:
                    st.write(f"**{message}**")
                    if data is not None:
                        if isinstance(data, dict):
                            st.json(data)
                        else:
                            st.text(str(data))

        try:
            # Step 1: Initialize/update data getter with needed experiments
            status_text.text("Connecting to CMIP6 cloud data...")
            progress_bar.progress(10)

            log_verbose(
                "Step 1: Initialize Data Getter",
                {
                    "selected_model": model_name,
                    "selected_scenario": scenario,
                    "variable": variable,
                },
            )

            # Update data getter if needed experiments changed
            needed_experiments = ["piControl", "abrupt-4xCO2", "historical", scenario]
            current_experiments = getattr(st.session_state.data_getter, "exps", [])

            log_verbose(
                "Checking required experiments",
                {
                    "needed": needed_experiments,
                    "current": current_experiments,
                    "needs_update": set(needed_experiments) != set(current_experiments),
                },
            )

            # Check if we need to reinitialize with different experiments
            if set(needed_experiments) != set(current_experiments):
                status_text.text("Updating data getter for selected scenario...")
                log_verbose("Reinitializing data getter with new experiments...")
                st.session_state.data_getter = initialize_data_getter(
                    needed_experiments
                )

            # Verify model has complete data
            if not st.session_state.data_getter.check_if_model_has_data(model_name):
                st.error(
                    f"""
                ❌ **Model {model_name} does not have complete data**
                
                This model is missing one or more required experiments:
                - piControl (pre-industrial control)
                - abrupt-4xCO2 (CO₂ quadrupling)
                - historical (historical run)
                - {scenario} (future scenario)
                
                Please try selecting a different model from the sidebar.
                
                **Available models:** {', '.join(available_models[:5])}
                {f'and {len(available_models)-5} more...' if len(available_models) > 5 else ''}
                """
                )
                progress_bar.empty()
                status_text.empty()
                st.stop()

            log_verbose("Model data verification passed", {"model": model_name})

            # Step 2: Train pattern model
            status_text.text(f"Training pattern scaling for {model_name}...")
            progress_bar.progress(30)

            log_verbose(
                "Step 2: Training Pattern Model",
                {
                    "training_experiments": [
                        "base (piControl)",
                        "co2x4 (abrupt-4xCO2)",
                        f"sulxanom (historical+{scenario})",
                    ],
                    "pattern_type": "anomsulf (includes aerosol residuals)",
                    "note": "Pattern learns relationships from historical+scenario composite",
                },
            )

            pattern_model = train_pattern_model(
                st.session_state.data_getter, model_name, scenario
            )

            log_verbose(
                "Pattern model trained successfully",
                {
                    "model_name": f"cmip6-{model_name}",
                    "experiments_used": ["base", "co2x4", "sulxanom"],
                },
            )

            # Step 3: Load scenario data
            status_text.text(f"Loading {scenario_display.lstrip('✓◐○ ')} scenario data...")
            progress_bar.progress(50)

            log_verbose(
                "Step 3: Loading Emissions and Concentrations",
                {
                    "scenario": scenario,
                    "files": [f"{scenario}_em_RCMIP.txt", f"{scenario}_conc_RCMIP.txt"],
                    "purpose": "Apply pattern to these forcings to generate projection",
                },
            )

            em_data, conc_data = load_scenario_data(scenario)

            log_verbose(
                "Scenario data loaded",
                {
                    "emissions_shape": (
                        str(em_data.shape) if hasattr(em_data, "shape") else "loaded"
                    ),
                    "concentrations_shape": (
                        str(conc_data.shape)
                        if hasattr(conc_data, "shape")
                        else "loaded"
                    ),
                },
            )

            # Step 4: Generate annual projection
            status_text.text("Generating annual climate projection...")
            progress_bar.progress(60)

            log_verbose(
                "Step 4: Predicting Climate Response",
                {
                    "method": "predict_from_combined_experiment",
                    "inputs": ["emissions", "concentrations"],
                    "variable": variable,
                    "note": "Applying learned pattern to scenario forcings",
                },
            )

            annual_pred = pattern_model.predict_from_combined_experiment(
                em_data, conc_data, [variable]
            )[variable]

            # Log some stats about the prediction
            global_mean_pred = prpatt.global_mean(annual_pred).values
            log_verbose(
                "Projection generated",
                {
                    "shape": str(annual_pred.shape),
                    "total_years": int(annual_pred.shape[-1]),
                    "note": "Prediction starts at year 1750",
                    "year_1750_temp": f"{global_mean_pred[0]:.2f}K",
                    "year_1850_temp": f"{global_mean_pred[100]:.2f}K",
                    "year_2100_temp": (
                        f"{global_mean_pred[-1]:.2f}K"
                        if len(global_mean_pred) > 350
                        else "N/A"
                    ),
                    "warming_1850_2100": (
                        f"{global_mean_pred[-1] - global_mean_pred[100]:.2f}K"
                        if len(global_mean_pred) > 350
                        else "N/A"
                    ),
                },
            )

            noise_realizations = []

            # Step 5: Add monthly noise if requested
            if include_noise:
                status_text.text("Training monthly noise model...")
                progress_bar.progress(70)

                log_verbose(
                    "Step 5: Training Noise Model",
                    {
                        "experiments": ["historical", scenario],
                        "n_modes": n_modes,
                        "lag_order": lag_order,
                        "monthly": True,
                    },
                )

                noise_model = train_noise_model(
                    st.session_state.data_getter,
                    model_name,
                    scenario,
                    variable,
                    n_modes,
                    lag_order,
                )

                log_verbose("Noise model trained")

                status_text.text(f"Generating {n_realizations} noise realizations...")
                progress_bar.progress(80)

                # Get temperature trajectory
                # Skip first 100 years (1750-1850) to align with scenario start
                global_temp = prpatt.global_mean(annual_pred).values[100:]
                monthly_temp = np.repeat(global_temp, 12)

                log_verbose(
                    "Generating noise realizations",
                    {
                        "n_realizations": n_realizations,
                        "monthly_points": len(monthly_temp),
                        "years_span": f"1850-{1850 + len(global_temp)}",
                        "noise_only": True,
                        "note": "Adding stochastic variability to smooth projection",
                    },
                )

                # Generate noise ensemble
                for i in range(n_realizations):
                    noise = noise_model.generate_realization(
                        monthly_temp, noise_only=True, random_seed=42 + i
                    )
                    noise_realizations.append(noise)

                log_verbose(f"Generated {len(noise_realizations)} realizations")

            # Step 6: Create visualization
            status_text.text("Creating visualization...")
            progress_bar.progress(90)

            log_verbose(
                "Step 6: Creating Visualization",
                {
                    "plot_type": "ensemble" if include_noise else "single trajectory",
                    "variable": variable,
                    "scenario": scenario,
                },
            )

            fig = plot_ensemble(
                annual_pred, noise_realizations, variable, scenario, model_name
            )

            progress_bar.progress(100)
            status_text.text("✅ Complete!")

            # Display results
            st.pyplot(fig)

            # Show workflow summary in verbose mode
            if verbose_mode:
                with verbose_container:
                    st.success("**Workflow Summary:**")
                    st.write(
                        """
                    1. ✅ Loaded training data: `base` (piControl), `co2x4` (abrupt-4xCO2), `sulxanom` (historical+{scenario})
                    2. ✅ Trained pattern model with aerosol residuals using experiments: [base, co2x4, sulxanom]
                    3. ✅ Loaded emissions and concentrations for {scenario}
                    4. ✅ Applied pattern to forcings using `predict_from_combined_experiment()`
                    5. {noise_step}
                    6. ✅ Generated visualization
                    """.format(
                            scenario=scenario,
                            noise_step=(
                                f"✅ Added {n_realizations} stochastic noise realizations"
                                if include_noise
                                else "⏭️ Skipped noise (annual only)"
                            ),
                        )
                    )

            # Download options
            st.subheader("💾 Download Results")

            col_a, col_b = st.columns(2)

            with col_a:
                # Create annual data in memory for download
                # Trim to start from year 1850 (skip first 100 years of spinup from 1750)
                import io
                annual_pred_trimmed = annual_pred.isel(time=slice(100, None))
                
                # xarray closes the buffer after writing, so we need to get bytes differently
                # Option 1: Use a temporary file path that writes to BytesIO
                # Option 2: Write to netcdf and read back
                netcdf_bytes = annual_pred_trimmed.to_netcdf()
                
                st.download_button(
                    "📥 Download Annual Data (NetCDF)",
                    netcdf_bytes,
                    file_name=f"meteor_{model_name}_{scenario}_{variable}_annual.nc",
                    mime="application/netcdf",
                    help="Annual mean data from 1850-2100",
                )

            with col_b:
                # Create plot in memory for download
                png_buffer = io.BytesIO()
                fig.savefig(png_buffer, format='png', dpi=300, bbox_inches="tight")
                png_buffer.seek(0)
                png_data = png_buffer.getvalue()
                
                st.download_button(
                    "📥 Download Plot (PNG)",
                    png_data,
                    file_name=f"meteor_{model_name}_{scenario}_{variable}.png",
                    mime="image/png",
                )

            # Statistics
            if include_noise:
                st.subheader("📈 Projection Statistics")

                # Calculate ensemble statistics
                final_year_values = []
                for noise in noise_realizations:
                    noise_global = prpatt.global_mean(noise).values
                    annual_monthly = np.repeat(global_temp, 12)[: len(noise_global)]
                    ensemble_member = annual_monthly + noise_global
                    final_year_values.append(
                        np.mean(ensemble_member[-12:])
                    )  # Last year average

                col_x, col_y, col_z = st.columns(3)

                with col_x:
                    st.metric(
                        "Ensemble Mean (2100)",
                        f"{np.mean(final_year_values):.2f}",
                        help="Average across all realizations",
                    )

                with col_y:
                    st.metric(
                        "Ensemble Std Dev",
                        f"{np.std(final_year_values):.2f}",
                        help="Spread of ensemble members",
                    )

                with col_z:
                    st.metric(
                        "Range",
                        f"{np.max(final_year_values) - np.min(final_year_values):.2f}",
                        help="Max - Min across ensemble",
                    )

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

        finally:
            progress_bar.empty()
            status_text.empty()

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>METEOR: Multivariate Emulation of Time-Evolving and Overlapping Responses</p>
    <p><a href='https://github.com/benmsanderson/METEOR'>GitHub</a> | 
    <a href='https://doi.org/10.5281/zenodo.15732955'>DOI: 10.5281/zenodo.15732955</a></p>
</div>
""",
    unsafe_allow_html=True,
)
