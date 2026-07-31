"""Parameterized profiling harness for METEOR training and generation.

Runs a named workload under cProfile, captures wall time and peak RSS, and
writes .prof + summary JSON for later analysis.

Usage:
    python scripts/profiling/run_profile.py --workload gen_global_ts --n 100
    python scripts/profiling/run_profile.py --workload train_fresh
    python scripts/profiling/run_profile.py --list

Output layout under scripts/profiling/results/:
    <workload>_<param-tag>_<timestamp>.prof   (raw cProfile output)
    <workload>_<param-tag>_<timestamp>.json   (wall time, peak RSS, params)
"""

import argparse
import cProfile
import gc
import json
import os
import pstats
import shutil
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

REAL_CACHE = REPO_ROOT / "cache"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _psutil_rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        return None


def _make_fresh_cache_dir(tag):
    """Create a temp cache dir with cmip6/ symlinked from the real cache.

    Model pkl files (pattern_scaling/, noise_models/) start empty so training
    is forced. CMIP6 raw data is reused to avoid cloud downloads.
    """
    fresh = REPO_ROOT / "cache_profile" / tag
    if fresh.exists():
        shutil.rmtree(fresh)
    fresh.mkdir(parents=True)
    (fresh / "cmip6").symlink_to(REAL_CACHE / "cmip6", target_is_directory=True)
    (fresh / "pattern_scaling").mkdir()
    (fresh / "noise_models").mkdir()
    return fresh


def _make_emulator(cache_dir, variables=("tas", "pr"), data_getter_kwargs=None):
    from meteor import MeteorInterface
    return MeteorInterface(
        model="NorESM2-MM",
        variables=list(variables),
        cache_dir=str(cache_dir),
        data_getter_kwargs=data_getter_kwargs,
    )


# --- workloads ---------------------------------------------------------------

def workload_train_fresh(params):
    """Force cache miss for pattern scaling + noise models."""
    fresh = _make_fresh_cache_dir(f"train_fresh_{params['tag']}")
    emu = _make_emulator(fresh)
    emu.train(verbose=False)
    return {"cache_dir": str(fresh)}


def workload_gen_global_ts(params):
    """Ex.1: single global-mean timeseries, vary n_realizations."""
    emu = _make_emulator(REAL_CACHE)
    emu.train(verbose=False)
    gc.collect()
    out = emu.generate_ensemble_outputs(
        scenario="ssp245",
        start_year=1850,
        end_year=2100,
        n_realizations=params["n"],
        timeseries=["global"],
        verbose=False,
    )
    return {"tas_shape": tuple(out["tas"].timeseries["global"].shape)}


def workload_gen_multi_scale(params):
    """Ex.2: 5 aggregations, N=100."""
    emu = _make_emulator(REAL_CACHE)
    emu.train(verbose=False)
    gc.collect()
    out = emu.generate_ensemble_outputs(
        scenario="ssp245",
        start_year=1850,
        end_year=2100,
        n_realizations=params["n"],
        timeseries=[
            "global",
            "regional:NEU",
            "regional:EAS",
            "point:59.9,10.8",
            "point:28.6,77.2",
        ],
        verbose=False,
    )
    return {"aggregations": list(out["tas"].timeseries.keys())}


def workload_gen_impacts(params):
    """Ex.3: degree-days at a single point."""
    emu = _make_emulator(REAL_CACHE)
    emu.train(verbose=False)
    gc.collect()
    out = emu.generate_ensemble_outputs(
        scenario="ssp245",
        start_year=1850,
        end_year=2100,
        n_realizations=params["n"],
        timeseries=["point:59.9,10.8"],
        impacts={"tas": {"degree_days": {"hdd_base": 18.0, "cdd_base": 18.0}}},
        verbose=False,
    )
    return {"hdd_shape": tuple(out["tas"].impacts["hdd"]["point:59.9,10.8"].shape)}


def workload_gen_no_noise(params):
    """Ex.5: climatology only (no stochastic PC generation)."""
    emu = _make_emulator(REAL_CACHE)
    emu.train(verbose=False)
    gc.collect()
    out = emu.generate_ensemble_outputs(
        scenario="ssp245",
        start_year=1850,
        end_year=2100,
        n_realizations=1,
        timeseries=["global", "regional:NEU"],
        include_noise=False,
        verbose=False,
    )
    return {"tas_shape": tuple(out["tas"].timeseries["global"].shape)}


def workload_gen_gridded(params):
    """Ex.4: gridded output at capped N (<=10 per user constraint)."""
    n = min(params["n"], 10)
    emu = _make_emulator(REAL_CACHE)
    emu.train(verbose=False)
    gc.collect()
    out = emu.generate_ensemble_outputs(
        scenario="ssp245",
        start_year=1850,
        end_year=2100,
        n_realizations=n,
        timeseries=["global"],
        gridded={"annual": [2030, 2050, 2100], "monthly": [2100]},
        verbose=False,
    )
    grid_2100 = out["tas"].gridded["annual"][2100]
    return {"gridded_annual_shape": tuple(grid_2100.shape)}


WORKLOADS = {
    "train_fresh": workload_train_fresh,
    "gen_global_ts": workload_gen_global_ts,
    "gen_multi_scale": workload_gen_multi_scale,
    "gen_impacts": workload_gen_impacts,
    "gen_no_noise": workload_gen_no_noise,
    "gen_gridded": workload_gen_gridded,
}


def run(workload_name, params):
    fn = WORKLOADS[workload_name]
    tag = params["tag"]

    profiler = cProfile.Profile()
    tracemalloc.start()
    rss_before = _psutil_rss_mb()

    profiler.enable()
    t0 = time.perf_counter()
    result = fn(params)
    wall_s = time.perf_counter() - t0
    profiler.disable()

    _, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _psutil_rss_mb()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prof_path = RESULTS_DIR / f"{workload_name}_{tag}_{ts}.prof"
    json_path = RESULTS_DIR / f"{workload_name}_{tag}_{ts}.json"
    profiler.dump_stats(str(prof_path))

    stats = pstats.Stats(profiler)
    all_rows = []
    for func, (cc, nc, tt, ct, _) in stats.stats.items():
        all_rows.append({
            "file": func[0], "line": func[1], "name": func[2],
            "ncalls": nc, "tottime": tt, "cumtime": ct,
        })
    top_by_cum = sorted(all_rows, key=lambda r: r["cumtime"], reverse=True)[:50]
    top_by_tot = sorted(all_rows, key=lambda r: r["tottime"], reverse=True)[:50]

    summary = {
        "workload": workload_name,
        "params": params,
        "timestamp": ts,
        "wall_seconds": wall_s,
        "tracemalloc_peak_mb": tracemalloc_peak / 1024 / 1024,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": (rss_after - rss_before) if rss_before is not None else None,
        "result": result,
        "prof_path": str(prof_path.relative_to(REPO_ROOT)),
        "top_by_cumtime": top_by_cum,
        "top_by_tottime": top_by_tot,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[{workload_name}:{tag}] wall={wall_s:.2f}s "
          f"peak_trace={tracemalloc_peak/1024/1024:.0f}MB "
          f"rss_delta={summary['rss_delta_mb']}")
    print(f"  prof: {prof_path.relative_to(REPO_ROOT)}")
    print(f"  json: {json_path.relative_to(REPO_ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=list(WORKLOADS) + ["all"],
                    help="Which workload to run.")
    ap.add_argument("--n", type=int, default=100,
                    help="n_realizations for generation workloads.")
    ap.add_argument("--tag", type=str, default=None,
                    help="Optional label appended to output filenames.")
    ap.add_argument("--list", action="store_true", help="List workloads and exit.")
    args = ap.parse_args()

    if args.list:
        for name in WORKLOADS:
            print(name)
        return

    if args.workload is None:
        ap.error("--workload is required (or --list)")

    params = {"n": args.n, "tag": args.tag or f"n{args.n}"}
    if args.workload == "all":
        for name in WORKLOADS:
            run(name, params)
    else:
        run(args.workload, params)


if __name__ == "__main__":
    main()
