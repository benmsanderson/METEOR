from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR

ds=xr.open_dataset('CanESM5_ssp245_monthly.nc')
t_globm=ds['tas'].mean(dim=['lat','lon','ens'])
t_globm=t_globm-t_globm[:500].mean()

t_glob = t_globm.rolling(month=60, center=True).mean().interpolate_na('month',method='nearest',fill_value="extrapolate").values

time=ds['month'].values
# ---
## 📈 Part 1: Modulated Harmonic Regression for Seasonal Cycle

# We need to create a "design matrix" (X) for the regression.
# Features: intercept, T_glob, sin/cos for annual & semi-annual, and interaction terms.

# Create harmonic features
months_per_year = 12
annual_cos = np.cos(2 * np.pi * time / months_per_year)
annual_sin = np.sin(2 * np.pi * time / months_per_year)
semiannual_cos = np.cos(4 * np.pi * time / months_per_year)
semiannual_sin = np.sin(4 * np.pi * time / months_per_year)

# Stack all features into the design matrix X (shape: n_time x n_features)
# We use T_glob * harmonic to create the interaction terms.
X = np.vstack([
    t_glob,
    annual_cos,
    annual_sin,
    semiannual_cos,
    semiannual_sin,
    t_glob * annual_cos,
    t_glob * annual_sin,
    t_glob * semiannual_cos,
    t_glob * semiannual_sin
]).T

# The temperature data needs to be flattened from (time, lat, lon) to (time, space)
# where space = lat * lon.

Y_xr = ds['tas'].mean(dim=['ens']).stack(space=('lat', 'lon'))
Y=Y_xr.data

# Fit a linear regression. Scikit-learn handles adding an intercept.
# This efficiently fits one model for each of the (n_lat * n_lon) grid cells at once.
model = LinearRegression(fit_intercept=True)
model.fit(X, Y)

# The model now contains all the coefficients (betas, alphas, gammas from our equation)
# model.intercept_ has shape (n_space,)
# model.coef_ has shape (n_space, n_features)

# Reconstruct the fitted seasonal cycle for all time
seasonal_cycle_fit = model.predict(X)

# Reshape both the fit and the residuals back to the original (time, lat, lon) grid
seasonal_cycle_fit_xr = xr.DataArray(
    seasonal_cycle_fit,
    coords=Y_xr.coords,
    dims=Y_xr.dims
).unstack('space')

anomalies = ds['tas'].mean(dim=['ens']) - seasonal_cycle_fit_xr
ds['anomalies'] = anomalies

print("\nSeasonal cycle extracted. Anomalies calculated.")
print("Example anomaly value at month=0, lat=0, lon=0:", ds['anomalies'].isel(month=0, lat=0, lon=0).item())
print("-" * 20)

# ---
## 🌀 Step 1: Standard PCA on Full Anomaly Dataset

# We need the anomaly data as a 2D numpy array (time, space)
anomalies_flat = ds['anomalies'].stack(space=('lat', 'lon')).data

# Initialize and fit the PCA model
n_modes = 10  # Let's retain the top 10 modes
pca = PCA(n_components=n_modes)
pcs = pca.fit_transform(anomalies_flat) # Principal Components (time series)
eofs = pca.components_                    # EOFs (spatial patterns)

print(f"✅ Step 1 complete: PCA performed.")
print(f"   - PC shape: {pcs.shape}")
print(f"   - EOF shape: {eofs.shape}")
print(f"   - Variance explained by {n_modes} modes: {pca.explained_variance_ratio_.sum():.2%}")
print("-" * 30)


# ---
## 📈 Step 2: Fit the VARX Model to the PCs

# The PCs are our target variables (endogenous)
# T_glob and month are our drivers (exogenous)

# Create the exogenous variable matrix
# We reuse the harmonic terms to represent the month

# The VAR model from statsmodels
# endog = our PC time series
# exog = our external drivers
model_var = VAR(endog=pcs, exog=X[:,:3])

# Fit the model. We must choose a lag order. For a demo, L=2 is fine.
# In a real study, you'd use model_var.select_order() to find the best lag.
lag_order = 2
results = model_var.fit(lag_order)

print(f"✅ Step 2 complete: VARX(2) model fitted to the {n_modes} PC time series.")
# Uncomment the line below to see a detailed statistical summary of the fit
# print(results.summary())
print("-" * 30)




# ---
## 🎲 Step 3: Simulate an Alternative Realisation (CORRECTED)

# We will build the new time series step-by-step in a loop.
n_time=pcs.shape[0]
n_simulation = n_time
synthetic_pcs = np.zeros_like(pcs)

# Use the historical data for the initial `lag_order` steps.
synthetic_pcs[:lag_order] = pcs[:lag_order]

# Get the covariance of the model's residuals (the "shocks")
# This is the key to generating new random behavior.
residual_cov = results.sigma_u
mean_shock = np.zeros(n_modes)

X_var=X[:n_simulation,:3]
# Loop from the first prediction point to the end
for t in range(lag_order, n_simulation):
    # 1. Get the inputs for the forecast
    current_initial_conditions = synthetic_pcs[t-lag_order:t]
    current_exog = X_var[t:t+1] # Must be 2D
    
    # 2. Get the deterministic forecast (the mean prediction)
    mean_forecast = results.forecast(y=current_initial_conditions, steps=1, exog_future=current_exog)
    
    # 3. Create a new random shock
    random_shock = np.random.multivariate_normal(mean_shock, residual_cov)
    
    # 4. Add the shock to the mean forecast to get the new stochastic value
    synthetic_pcs[t] = mean_forecast + random_shock


# Reconstruct the anomalies by multiplying the synthetic PCs by the EOFs
# This is the inverse of the PCA transformation
reconstructed_anomalies_flat = synthetic_pcs @ eofs

# Reshape the data back to the original grid (time, lat, lon)
# We can create a new xarray DataArray for this
reconstructed_anomalies_xr = xr.DataArray(
    reconstructed_anomalies_flat.reshape(n_time, len(ds['lat']), len(ds['lon'])),
    coords={
        "time": ds.coords['month'],
        "lat": ds.coords['lat'],
        "lon": ds.coords['lon']
    },
    dims=("month", "lat", "lon")
)
