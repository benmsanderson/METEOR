"""
Diagnostic: why is METEOR's annual global-mean tas variability ~2.5x too small?

The EOF-weighting fix made the *monthly* global variance correct, but the
*annual* global variance collapses because the generated noise is temporally
too white. This script localizes the lost low-frequency power by measuring the
"redness" of the global-mean signal at three stages of the pipeline:

  STAGE A  raw ESM           : deseasonalized global-mean anomaly (the target)
  STAGE B  training anomaly  : after seasonal model + t_glob response removed
                               (what PCA/VAR are actually asked to reproduce)
  STAGE C  VAR-X generated   : the synthetic noise the model emits

Redness metrics (all on the area-weighted global mean):
  - annual / monthly variance ratio  (white noise = 1/12 ~ 0.083; red > that)
  - lag-1..12 month autocorrelation

Reading the result:
  * If B is still red (ratio >> white, lag-1 positive) but C is white/negative,
    the VAR(lag_order) temporal model is whitening it  -> fix = richer temporal
    model (longer/structured lags, or an explicit low-frequency global mode).
  * If B is already white, the seasonal + 60-month-smoothed-t_glob removal has
    stripped the low-frequency internal variability before the noise model sees
    it -> fix = change the forced/internal split (smoothing window / regressor).

Pure numpy/sklearn + a tiny OLS VAR-X (matches statsmodels); no meteor import.

Run:
    python scripts/diagnose_temporal_persistence.py --model CanESM5
"""

import argparse

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

CACHE = "cache/cmip6"  # maurad layout; override with --cache
ROLLING_WINDOW = 60


def gmean(da):
    return da.weighted(np.cos(np.deg2rad(da.lat))).mean(("lat", "lon"))


def deseasonalize(x):
    """Remove cubic drift then per-calendar-month climatology (x is monthly)."""
    n = len(x) // 12 * 12
    x = x[:n].astype(float)
    t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 3), t)
    xm = x.reshape(-1, 12)
    return (xm - xm.mean(0)).reshape(-1)


def redness(x, label):
    """Print annual/monthly variance ratio and autocorrelation for a 1D monthly series."""
    n = len(x) // 12 * 12
    x = x[:n]
    vm = x.var()
    ann = x.reshape(-1, 12).mean(1)
    va = ann.var()

    def acf(k):
        a, b = x[:-k], x[k:]
        return np.mean((a - a.mean()) * (b - b.mean())) / x.var()

    ac = {k: acf(k) for k in (1, 3, 6, 12)}
    print(f"  {label}")
    print(f"     monthly std = {vm**0.5:.4f}   annual std = {va**0.5:.4f}   "
          f"ann/mon var ratio = {va/vm:.3f}  (white={1/12:.3f})")
    print(f"     autocorr  lag1={ac[1]:+.2f}  lag3={ac[3]:+.2f}  "
          f"lag6={ac[6]:+.2f}  lag12={ac[12]:+.2f}")
    return dict(monthly=vm**0.5, annual=va**0.5, ratio=va / vm, acf1=ac[1])


def fit_varx_ols(pcs, x_exog, lag):
    T, k = pcs.shape
    Z, Y = [], []
    for t in range(lag, T):
        z = [1.0]
        for i in range(lag):
            z.extend(pcs[t - i - 1])
        if x_exog is not None:
            z.extend(np.atleast_1d(x_exog[t]))
        Z.append(z)
        Y.append(pcs[t])
    Z, Y = np.asarray(Z), np.asarray(Y)
    beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    resid = Y - Z @ beta
    sigma = resid.T @ resid / (Z.shape[0] - Z.shape[1])
    intercept = beta[0]
    A = [beta[1 + i * k:1 + (i + 1) * k].T for i in range(lag)]
    n_exog = 0 if x_exog is None else np.atleast_2d(x_exog.T).shape[0]
    B = beta[-n_exog:].T if n_exog else None
    return intercept, A, B, sigma


def simulate(intercept, A, B, sigma, x_exog, n_time, rng):
    k = len(intercept)
    lag = len(A)
    y = np.zeros((n_time, k))
    shocks = rng.multivariate_normal(np.zeros(k), sigma, size=n_time)
    for t in range(lag, n_time):
        f = intercept.copy()
        for i in range(lag):
            f = f + A[i] @ y[t - i - 1]
        if B is not None:
            f = f + B @ np.atleast_1d(x_exog[t])
        y[t] = f + shocks[t]
    return y


def harmonic(time, tg):
    m = 12
    ac, as_ = np.cos(2 * np.pi * time / m), np.sin(2 * np.pi * time / m)
    sc, ss = np.cos(4 * np.pi * time / m), np.sin(4 * np.pi * time / m)
    return np.vstack(
        [tg, ac, as_, sc, ss, tg * ac, tg * as_, tg * sc, tg * ss]
    ).T


def run(model, var, exps, n_modes, lag, weight, smooth, seed, use_exog):
    print("=" * 78)
    print(f"TEMPORAL PERSISTENCE DIAGNOSTIC  |  {model} {var}  "
          f"n_modes={n_modes} lag={lag} weight_eofs={weight} t_glob_smooth={smooth}mo")
    print("=" * 78)

    # ---- STAGE A: raw ESM internal variability (piControl) ----
    pic = xr.open_dataset(f"{CACHE}/{model}_piControl_{var}_monthly.nc")
    pic_v = pic[[v for v in pic.data_vars][0]]
    if "ens" in pic_v.dims:
        pic_v = pic_v.isel(ens=0)
    g_pic = deseasonalize(gmean(pic_v).values)
    print("STAGE A  raw ESM (piControl, deseasonalized) = TARGET")
    rA = redness(g_pic, "raw global-mean anomaly")
    print()

    # ---- build training composite & replicate fit() preprocessing ----
    parts = [xr.open_dataset(f"{CACHE}/{model}_{e}_{var}_monthly.nc")[var] for e in exps]
    da = xr.concat(parts, dim="month").mean("ens")
    da = da.assign_coords(month=np.arange(da.sizes["month"]))
    time = da["month"].values.astype(float)

    base = float(gmean(pic_v).mean().values)
    glob = gmean(da) - base
    tg = glob.rolling(month=smooth, center=True, min_periods=1).mean().values

    X = harmonic(time, tg)
    Y = da.stack(space=("lat", "lon")).data.astype(np.float64)
    seasonal = LinearRegression().fit(X, Y)
    anom = Y - seasonal.predict(X)  # (T, space)

    # STAGE B: redness of the TRAINING anomaly global mean
    w = np.cos(np.deg2rad(np.repeat(da.lat.values, da.sizes["lon"])))
    w = w / w.sum()
    print("STAGE B  training anomaly (seasonal + t_glob response removed)")
    rB = redness(anom @ w, "what PCA/VAR must reproduce")
    print()

    # ---- fit weighted PCA + VAR-X, generate, measure STAGE C ----
    lat_cell = np.repeat(da.lat.values, da.sizes["lon"])
    if weight:
        sw = np.sqrt(np.clip(np.cos(np.deg2rad(lat_cell)), 1e-6, None))
        Aw = anom * sw
    else:
        sw = None
        Aw = anom
    pca = PCA(n_components=n_modes).fit(Aw)
    pcs = pca.transform(Aw)
    comps = pca.components_ if sw is None else pca.components_ / sw
    gbar = comps @ w  # global projection of physical EOFs

    if use_exog == "all":
        x_exog = X[:, :3]  # t_glob, annual_cos, annual_sin
    elif use_exog == "temp_only":
        x_exog = X[:, :1]
    else:
        x_exog = None
    intercept, A, B, sigma = fit_varx_ols(pcs, x_exog, lag)
    rng = np.random.default_rng(seed)
    # Generate an ensemble and measure the ensemble-mean-removed noise, exactly
    # as the shipped output is measured. This cancels any deterministic seasonal
    # cycle that use_exog='all' injects via B*[cos,sin], isolating the stochastic
    # part (the actual internal variability the model emits).
    n_real = 10
    g_ens = np.array(
        [simulate(intercept, A, B, sigma, x_exog, len(time), rng) @ gbar
         for _ in range(n_real)]
    )
    g_noise = (g_ens - g_ens.mean(0))[0]
    print(f"STAGE C  VAR-X generated noise (lag={lag}, use_exog={use_exog})")
    rC = redness(g_noise, "synthetic global-mean noise (ens-mean removed)")
    print()

    print("-" * 78)
    print("INTERPRETATION")
    print(f"  annual global std   A(target)={rA['annual']:.4f}  "
          f"B(training)={rB['annual']:.4f}  C(generated)={rC['annual']:.4f} K")
    print(f"  ann/mon redness     A={rA['ratio']:.3f}  B={rB['ratio']:.3f}  "
          f"C={rC['ratio']:.3f}   (white={1/12:.3f})")
    b_red = rB["ratio"] > 1.5 / 12
    c_red = rC["ratio"] > 1.5 / 12
    if b_red and not c_red:
        verdict = ("Training anomaly is RED but generated noise is WHITE -> the "
                   f"VAR(lag={lag}) is whitening it. Fix = richer temporal model.")
    elif not b_red:
        verdict = ("Training anomaly is already WHITE -> seasonal + "
                   f"{smooth}mo-smoothed t_glob removal stripped the low-frequency "
                   "internal variability before the noise model. Fix = forced/"
                   "internal split.")
    else:
        verdict = ("Both training and generated retain redness; deficit is "
                   "milder than expected for this config.")
    print(f"  VERDICT: {verdict}")
    print("=" * 78)


def main():
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CanESM5")
    ap.add_argument("--var", default="tas")
    ap.add_argument("--exps", nargs="+", default=["historical", "ssp245"])
    ap.add_argument("--n-modes", type=int, default=40)
    ap.add_argument("--lag", type=int, default=2)
    ap.add_argument("--no-weight", action="store_true")
    ap.add_argument("--smooth", type=int, default=ROLLING_WINDOW)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-exog", default="all",
                    choices=["all", "temp_only", "none"],
                    help="exog set for the VAR-X (her tas default is 'all')")
    args = ap.parse_args()
    CACHE = args.cache
    run(args.model, args.var, args.exps, args.n_modes, args.lag,
        not args.no_weight, args.smooth, args.seed, args.use_exog)


if __name__ == "__main__":
    main()
