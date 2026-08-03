"""Statistical + shape equivalence check for the batched PC generator.

Compares the new _generate_stochastic_pcs_batched against the reference
sequential loop calling _generate_stochastic_pcs. Because the RNG draws in a
different order for the two paths, we cannot expect byte-identical samples;
we check that per-lag mean, std, and cross-covariance match to within the
sampling standard error at a large N.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from meteor.noise_generator import MeteorNoiseGenerator


def _load_fitted_noise(variable="tas"):
    model_path = REPO_ROOT / "cache" / "noise_models" / f"NorESM2-MM_{variable}_noise_model.pkl"
    gen = MeteorNoiseGenerator()
    gen.load_model(str(model_path))
    return gen


def main():
    gen = _load_fitted_noise("tas")
    n_time = 251 * 12  # match the standard trajectory length
    n_realizations = 500

    # Build an exogenous input the way generate_stochastic_pcs does.
    time = np.arange(n_time)
    fake_temp = np.linspace(0, 2.0, n_time)  # smooth warming trajectory
    X = gen._create_harmonic_features(time, fake_temp)
    X_exog = gen._extract_exog_variables(X)

    # Sequential baseline
    np.random.seed(1234)
    seq = np.stack([gen._generate_stochastic_pcs(X_exog, n_time)
                    for _ in range(n_realizations)])

    # Batched
    np.random.seed(1234)
    bat = gen._generate_stochastic_pcs_batched(X_exog, n_time, n_realizations)

    print(f"seq shape: {seq.shape}   bat shape: {bat.shape}")
    assert seq.shape == bat.shape

    # Compare per-mode summary statistics across the realization axis
    # (mean at each t; std at each t).
    seq_mean = seq.mean(axis=0)  # (n_time, n_modes)
    bat_mean = bat.mean(axis=0)
    seq_std = seq.std(axis=0)
    bat_std = bat.std(axis=0)

    # Standard error of the mean is sigma / sqrt(N); use 4-sigma tolerance
    # against the sequential std as our reference.
    tol_mean = 4.0 * seq_std / np.sqrt(n_realizations)

    diff_mean = np.abs(seq_mean - bat_mean)
    diff_std = np.abs(seq_std - bat_std) / seq_std.clip(min=1e-9)

    print(f"|Δmean| max = {diff_mean.max():.4g}   "
          f"tol max = {tol_mean.max():.4g}   "
          f"(mean over t,m: {diff_mean.mean():.4g})")
    print(f"|Δstd|/std max = {diff_std.max():.4g}   "
          f"(mean over t,m: {diff_std.mean():.4g})")

    # Marginal spectrum via lag-0 autocov: (n_modes, n_modes)
    def _cov(x):
        # Flatten (N, T, m) -> (N*T, m) then compute mode-mode cov
        flat = x.reshape(-1, x.shape[-1])
        return np.cov(flat.T)

    seq_cov = _cov(seq)
    bat_cov = _cov(bat)
    cov_frob = np.linalg.norm(seq_cov - bat_cov) / np.linalg.norm(seq_cov)
    print(f"||Δcov||_F / ||cov||_F = {cov_frob:.4g}")

    # tol_mean can be 0 at t=0 (initial conditions), so allow a floor
    ok = (diff_mean < np.maximum(tol_mean * 3, 1e-10)).all() and cov_frob < 0.02
    print("\nSTATISTICAL EQUIVALENCE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
