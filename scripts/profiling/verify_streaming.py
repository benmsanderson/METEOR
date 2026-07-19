"""Ad-hoc memory/timing benchmark for the streaming gridded builder.

Runs ``MeteorInterface._build_ensemble_slices_streaming`` directly at several
``chunk_size`` values with the trained NorESM2-MM pr model and prints wall time
and peak Python-tracked memory. Equivalence against the previous whole-ensemble
path was verified when that path still existed (see the streaming refactor
commit); this script is retained for future memory-scaling investigations.

Usage:
    python scripts/profiling/verify_streaming.py --n 5 --chunk 1
"""

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meteor import MeteorInterface


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=None,
                    help="Chunk size. Default: auto (~5 GB per chunk).")
    args = ap.parse_args()

    emu = MeteorInterface(
        model="NorESM2-MM",
        variables=["tas", "pr"],
        cache_dir=str(REPO_ROOT / "cache"),
    )
    emu.train(verbose=False)

    variable = "pr"
    gen_inputs = emu._prepare_generation(
        variable=variable,
        scenario="ssp245",
        start_year=1850,
        end_year=2100,
        n_realizations=args.n,
        include_noise=True,
        verbose=False,
    )
    transform_config = emu._get_transform_config(variable)

    start_year, end_year = 1850, 2100

    def yr_to_idx(y):
        return (y - start_year) * 12

    slice_specs = [
        ((yr_to_idx(2030), yr_to_idx(2030) + 12, True), yr_to_idx(2030), yr_to_idx(2030) + 12, True),
        ((yr_to_idx(2050), yr_to_idx(2050) + 12, True), yr_to_idx(2050), yr_to_idx(2050) + 12, True),
        ((yr_to_idx(2100), yr_to_idx(2100) + 12, True), yr_to_idx(2100), yr_to_idx(2100) + 12, True),
        ((yr_to_idx(2100), yr_to_idx(2100) + 12, False), yr_to_idx(2100), yr_to_idx(2100) + 12, False),
    ]

    tracemalloc.start()
    t0 = time.perf_counter()
    streamed = emu._build_ensemble_slices_streaming(
        variable,
        gen_inputs.pattern.monthly_prediction,
        gen_inputs.pattern.full_monthly_warming[
            gen_inputs.pattern.start_month_idx : gen_inputs.pattern.end_month_idx
        ],
        emu.noise_models[variable],
        gen_inputs.stochastic_pcs,
        start_year,
        end_year,
        transform_config,
        include_noise=True,
        slice_specs=slice_specs,
        chunk_size=args.chunk,
        verbose=False,
    )
    t = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"N={args.n} chunk={args.chunk}: wall={t:.1f}s peak_trace={peak/1024/1024:.0f}MB "
          f"slices={len(streamed)}")


if __name__ == "__main__":
    sys.exit(main())
