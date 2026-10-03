"""Aggregate profiling runs into a comparable table.

Reads .prof files directly (authoritative) and the .json summaries for
wall-time / memory. Prints a summary table, a cross-workload hotspot
leaderboard, and a drill-down for a single run.

Usage:
    python scripts/profiling/analyze.py                       # summary + leaderboard
    python scripts/profiling/analyze.py --run <substring>     # single run drill-down
    python scripts/profiling/analyze.py --meteor-only         # filter to meteor package
    python scripts/profiling/analyze.py --exclude 'noise_generator.py'
"""

import argparse
import json
import pstats
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_run_summaries():
    runs = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        with open(p) as f:
            summary = json.load(f)
        prof_path = RESULTS_DIR / f"{p.stem}.prof"
        if not prof_path.exists():
            continue
        runs.append((p, summary, prof_path))
    return runs


def extract_rows(prof_path, top=50):
    stats = pstats.Stats(str(prof_path))
    rows = []
    for func, (cc, nc, tt, ct, _) in stats.stats.items():
        rows.append({
            "file": func[0], "line": func[1], "name": func[2],
            "ncalls": nc, "tottime": tt, "cumtime": ct,
        })
    by_cum = sorted(rows, key=lambda r: r["cumtime"], reverse=True)[:top]
    by_tot = sorted(rows, key=lambda r: r["tottime"], reverse=True)[:top]
    return by_cum, by_tot


def summary_table(runs):
    header = f"{'workload':<18} {'tag':<12} {'wall_s':>8} {'peak_MB':>8} {'rss_dMB':>8}"
    print(header)
    print("-" * len(header))
    for _, r, _ in runs:
        rss = r.get("rss_delta_mb")
        rss_str = f"{rss:8.0f}" if isinstance(rss, (int, float)) else f"{'?':>8}"
        print(f"{r['workload']:<18} {r['params'].get('tag', ''):<12} "
              f"{r['wall_seconds']:8.2f} {r['tracemalloc_peak_mb']:8.0f} {rss_str}")


def _is_meteor(row):
    return "/meteor/" in row["file"] or row["file"].endswith("/meteor.py")


def _row_key(row):
    return (Path(row["file"]).name, row["line"], row["name"])


def drilldown(prof_path, wall_s, tag, meteor_only=False, exclude=None, top=25):
    by_cum, by_tot = extract_rows(prof_path, top=200)

    def _keep(r):
        if meteor_only and not _is_meteor(r):
            return False
        if exclude and exclude in r["file"]:
            return False
        return True

    print(f"\n=== {tag} (wall={wall_s:.2f}s) ===")
    print("\n-- top by cumulative time --")
    for r in [x for x in by_cum if _keep(x)][:top]:
        loc = f"{Path(r['file']).name}:{r['line']}"
        print(f"  {r['cumtime']:8.3f}s  {loc:<40} {r['name']}  (ncalls={r['ncalls']})")

    print("\n-- top by self time --")
    for r in [x for x in by_tot if _keep(x)][:top]:
        loc = f"{Path(r['file']).name}:{r['line']}"
        print(f"  {r['tottime']:8.3f}s  {loc:<40} {r['name']}  (ncalls={r['ncalls']})")


def hotspot_leaderboard(runs, meteor_only=True, top=25):
    agg = defaultdict(lambda: {"tottime": 0.0, "ncalls": 0, "workloads": set()})
    for _, summary, prof_path in runs:
        _, by_tot = extract_rows(prof_path, top=100)
        for r in by_tot:
            if meteor_only and not _is_meteor(r):
                continue
            k = _row_key(r)
            agg[k]["tottime"] += r["tottime"]
            agg[k]["ncalls"] += r["ncalls"]
            agg[k]["workloads"].add(summary["workload"])
    rows = sorted(agg.items(), key=lambda kv: kv[1]["tottime"], reverse=True)[:top]
    print("\n=== Cross-workload hotspot leaderboard (summed self time, meteor only) ===")
    for (fname, ln, name), v in rows:
        loc = f"{fname}:{ln}"
        print(f"  {v['tottime']:8.2f}s  {loc:<40} {name}  "
              f"[{','.join(sorted(v['workloads']))}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="Substring of a run filename for drill-down.")
    ap.add_argument("--meteor-only", action="store_true",
                    help="Filter drill-down to meteor package only.")
    ap.add_argument("--exclude", help="File-path substring to exclude from drill-down.")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    runs = load_run_summaries()
    if not runs:
        print("No results in", RESULTS_DIR)
        return

    if args.run:
        for p, summary, prof in runs:
            if args.run in p.name:
                drilldown(prof, summary["wall_seconds"], p.stem,
                          meteor_only=args.meteor_only, exclude=args.exclude,
                          top=args.top)
    else:
        summary_table(runs)
        hotspot_leaderboard(runs, meteor_only=True, top=args.top)


if __name__ == "__main__":
    main()
