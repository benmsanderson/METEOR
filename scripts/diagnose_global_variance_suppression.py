"""
Diagnostic: where does the METEOR noise model lose global-mean variability?

Reproduces MeteorNoiseGenerator.fit() preprocessing exactly (seasonal model ->
anomalies -> EOF/PCA) and the generator (VAR-X) in pure numpy, then decomposes
the global-mean internal variance through the pipeline:

    V_full      truth: area-weighted global mean variance of the anomaly field
    V_retained  ceiling: same, reconstructed from the retained n_modes EOFs
                 (in-sample PCs, includes forced/exog component) -> EOF-basis loss
    V_exog      variance of the global mean carried by the deterministic exog term
                 (B . t_glob) -> the part replaced by a smooth curve at generation
    V_gen       what the generator actually produces as ensemble spread

The key identity used throughout:
    global_mean(field)_t = sum_k PC_k(t) * gbar_k ,   gbar_k = <EOF_k>_area
so V_global = gbar^T Cov(PC) gbar, and point/regional variance sample a *different*
projection of Cov(PC) -- which is why point variance can look fine while the
global mean is suppressed.

No statsmodels / no meteor import required: VAR-X is fit by plain multi-output OLS
(identical to statsmodels VAR equation-by-equation OLS) and simulated exactly as
MeteorNoiseGenerator._generate_stochastic_pcs does (zero init, batched MVN shocks).

Run:
    python scripts/diagnose_global_variance_suppression.py
"""

import argparse

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

CACHE = ".cache/cmip6"
ROLLING_WINDOW = 60  # months, matches fit()


# --------------------------------------------------------------------------- #
# 1. Data + preprocessing -- faithful to MeteorNoiseGenerator.fit()
# --------------------------------------------------------------------------- #
def load_composite(model, var, exps):
    parts = []
    for exp in exps:
        ds = xr.open_dataset(f"{CACHE}/{model}_{exp}_{var}_monthly.nc")
        parts.append(ds[var])
    da = xr.concat(parts, dim="month")
    da = da.assign_coords(month=np.arange(da.sizes["month"]))
    return da


def make_harmonic_features(time, t_glob):
    """Exact copy of MeteorNoiseGenerator._create_harmonic_features."""
    m = 12
    ac, as_ = np.cos(2 * np.pi * time / m), np.sin(2 * np.pi * time / m)
    sc, ss = np.cos(4 * np.pi * time / m), np.sin(4 * np.pi * time / m)
    return np.vstack(
        [t_glob, ac, as_, sc, ss, t_glob * ac, t_glob * as_, t_glob * sc, t_glob * ss]
    ).T


def area_weight_vector(lat, n_lon):
    """w over flattened (lat,lon) space s.t. field @ w == area-weighted global mean,
    matching _weighted_mean_over_region's global branch (cos-lat, no lon weighting)."""
    coslat = np.cos(np.deg2rad(lat))
    total = coslat.sum() * n_lon
    w_cell = np.repeat(coslat / total, n_lon)  # (n_lat*n_lon,)
    return w_cell, coslat


def preprocess(model, var, exps):
    da = load_composite(model, var, exps).mean(dim="ens")  # ens=1 -> no-op
    lat = da["lat"].values
    lon = da["lon"].values
    n_lat, n_lon = len(lat), len(lon)
    time = da["month"].values.astype(float)

    # piControl baseline (scalar), as train_noise_model_from_cmip6 does
    pic = xr.open_dataset(f"{CACHE}/{model}_piControl_{var}_monthly.nc")[var]
    coslat_da = np.cos(np.deg2rad(pic["lat"]))
    baseline = float(pic.weighted(coslat_da).mean(dim=("lat", "lon")).mean().values)

    # t_glob: area-weighted global mean, minus baseline, 60-mo centered rolling
    glob = da.weighted(np.cos(np.deg2rad(da["lat"]))).mean(dim=("lat", "lon"))
    glob = glob - baseline
    t_glob = (
        glob.rolling(month=ROLLING_WINDOW, center=True, min_periods=1).mean().values
    )

    X = make_harmonic_features(time, t_glob)

    Y = da.stack(space=("lat", "lon")).data.astype(np.float64)  # (T, space)
    seasonal = LinearRegression(fit_intercept=True).fit(X, Y)
    anomalies = Y - seasonal.predict(X) - baseline  # (T, space); const drops in var

    return dict(
        anomalies=anomalies,
        X=X,
        t_glob=t_glob,
        lat=lat,
        lon=lon,
        n_lat=n_lat,
        n_lon=n_lon,
        n_ens=load_composite(model, var, exps).sizes["ens"],
    )


# --------------------------------------------------------------------------- #
# 2. VAR-X by OLS (matches statsmodels VAR) + exact generator simulation
# --------------------------------------------------------------------------- #
def fit_varx_ols(pcs, x_exog, lag):
    """y_t = c + sum A_i y_{t-i} + B x_t + e_t, fit by OLS. Returns params for the
    manual generator loop plus residual covariance (df-adjusted, like sigma_u)."""
    T, k = pcs.shape
    rows_y, rows_Z = [], []
    for t in range(lag, T):
        z = [1.0]
        for i in range(lag):
            z.extend(pcs[t - i - 1])
        if x_exog is not None:
            z.extend(np.atleast_1d(x_exog[t]))
        rows_Z.append(z)
        rows_y.append(pcs[t])
    Z = np.asarray(rows_Z)
    Yt = np.asarray(rows_y)
    beta, *_ = np.linalg.lstsq(Z, Yt, rcond=None)  # (nreg, k)
    resid = Yt - Z @ beta
    sigma_u = resid.T @ resid / (Z.shape[0] - Z.shape[1])

    intercept = beta[0]
    A = [beta[1 + i * k : 1 + (i + 1) * k].T for i in range(lag)]
    n_exog = 0 if x_exog is None else np.atleast_2d(x_exog.T).shape[0]
    B = beta[-n_exog:].T if n_exog else None
    return intercept, A, B, sigma_u


def simulate_pcs(intercept, A, B, sigma_u, x_exog, n_time, n_real, rng):
    """Exact reproduction of _generate_stochastic_pcs (zero init, batched shocks)."""
    k = len(intercept)
    lag = len(A)
    out = np.zeros((n_real, n_time, k))
    for r in range(n_real):
        shocks = rng.multivariate_normal(np.zeros(k), sigma_u, size=n_time)
        y = np.zeros((n_time, k))
        for t in range(lag, n_time):
            f = intercept.copy()
            for i in range(lag):
                f = f + A[i] @ y[t - i - 1]
            if B is not None:
                f = f + B @ np.atleast_1d(x_exog[t])
            y[t] = f + shocks[t]
        out[r] = y
    return out


# --------------------------------------------------------------------------- #
# 3. Decomposition
# --------------------------------------------------------------------------- #
def run(model, var, exps, mode_list, lag, n_real, seed):
    pp = preprocess(model, var, exps)
    A_anom = pp["anomalies"]
    w, coslat = area_weight_vector(pp["lat"], pp["n_lon"])
    T = A_anom.shape[0]

    V_full = np.var(A_anom @ w)
    print("=" * 78)
    print(f"GLOBAL-MEAN VARIANCE DIAGNOSTIC  |  {model} {var}  exps={exps}")
    print("=" * 78)
    print(f"grid {pp['n_lat']}x{pp['n_lon']}={pp['n_lat']*pp['n_lon']} cells | "
          f"T={T} months | n_ens={pp['n_ens']} (fit uses ens-mean)")
    print(f"V_full  (truth, area-weighted global-mean anomaly variance) = {V_full:.6e}")
    print()

    # area weighting for proper-metric PCA: column scale sqrt(cos lat)
    sqrt_area = np.sqrt(np.repeat(coslat, pp["n_lon"]))

    hdr = (f"{'n_modes':>7} | {'EOF basis (unwtd)':>18} | "
           f"{'EOF basis (area-wtd)':>20} | {'cum var-expl (unwtd)':>20}")
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for nm in mode_list:
        # --- unweighted PCA (what METEOR does) ---
        pca = PCA(n_components=nm).fit(A_anom)
        pcs = pca.transform(A_anom)
        comps = pca.components_
        gbar = comps @ w
        recon_gm = pcs @ gbar
        V_ret = np.var(recon_gm)
        ve = pca.explained_variance_ratio_.sum()

        # --- area-weighted PCA (proper EOF metric) ---
        pca_w = PCA(n_components=nm).fit(A_anom * sqrt_area)
        pcs_w = pca_w.transform(A_anom * sqrt_area)
        comps_w_phys = pca_w.components_ / sqrt_area  # back to physical units
        gbar_w = comps_w_phys @ w
        V_ret_w = np.var(pcs_w @ gbar_w)

        print(f"{nm:>7} | {V_ret/V_full:>17.1%} | {V_ret_w/V_full:>19.1%} | "
              f"{ve:>19.1%}")
        results[nm] = dict(pca=pca, pcs=pcs, comps=comps, gbar=gbar, V_ret=V_ret)

    # ---- VAR-X stage at the production mode count (largest requested) ----
    nm = mode_list[-1]
    r = results[nm]
    pcs, gbar = r["pcs"], r["gbar"]
    x_exog = pp["X"][:, 0]  # use_exog='temp_only' -> t_glob only
    intercept, Amats, B, sigma_u = fit_varx_ols(pcs, x_exog, lag)

    # in-sample exog-driven global component (the part that becomes a smooth curve)
    exog_contrib = (x_exog[:, None] * B[:, 0][None, :]) @ gbar  # (T,)
    V_exog = np.var(exog_contrib)

    rng = np.random.default_rng(seed)
    gen = simulate_pcs(intercept, Amats, B, sigma_u, x_exog, T, n_real, rng)
    V_gen = np.mean([np.var(gen[i] @ gbar) for i in range(n_real)])

    print()
    print("-" * 78)
    print(f"VAR-X stage (n_modes={nm}, lag={lag}, exog='temp_only', "
          f"{n_real} realizations)")
    print("-" * 78)
    print(f"V_full                                  = {V_full:.4e}  (100%)")
    print(f"V_retained (EOF basis ceiling)          = {r['V_ret']:.4e}  "
          f"({r['V_ret']/V_full:6.1%} of truth)   <- EOF-basis loss")
    print(f"V_exog (deterministic B*t_glob part)    = {V_exog:.4e}  "
          f"({V_exog/r['V_ret']:6.1%} of retained) <- drained at generation")
    print(f"V_gen (actual generator ensemble spread)= {V_gen:.4e}  "
          f"({V_gen/V_full:6.1%} of truth)         <- what you ship")
    print()
    print(f"  EOF-basis retains          : {r['V_ret']/V_full:6.1%} of global variance")
    print(f"  generator/basis ratio      : {V_gen/r['V_ret']:6.1%} "
          f"(VAR-X + exog + burn-in loss)")
    print(f"  TOTAL global suppression   : generator reproduces "
          f"{V_gen/V_full:.1%} of target global variance")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NorESM2-MM")
    ap.add_argument("--var", default="tas")
    ap.add_argument("--exps", nargs="+", default=["historical", "ssp245"])
    ap.add_argument("--modes", nargs="+", type=int,
                    default=[5, 10, 20, 40, 80, 160])
    ap.add_argument("--lag", type=int, default=2)
    ap.add_argument("--n-real", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.model, args.var, args.exps, sorted(args.modes), args.lag,
        args.n_real, args.seed)


if __name__ == "__main__":
    main()
