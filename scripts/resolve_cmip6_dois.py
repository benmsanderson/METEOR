#!/usr/bin/env python
"""Resolve source DOIs for the cached CMIP6 netCDF files.

The METEOR cache strips original CMIP6 global attributes, so the DOI is not in
the files. We reconstruct (source_id, experiment_id) from the filenames, map to
(activity_id, institution_id) via the Pangeo catalog CSV, then query the DKRZ
CMIP6 citation service for the dataset DOI.
"""
import csv
import glob
import json
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "notebooks/cache/cmip6/cmip6-zarr-consolidated-stores.csv")
API = ("https://cera-www.dkrz.de/WDCC/ui/cerasearch/cerarest/"
       "exportcmip6?input=CMIP6.{act}.{inst}.{src}.{exp}&wt=json")

MODELS = {"CanESM5", "CESM2-WACCM", "CNRM-ESM2-1", "IPSL-CM6A-LR",
          "NorESM2-MM", "UKESM1-0-LL"}
EXPERIMENTS = {"piControl", "abrupt-4xCO2", "historical",
               "ssp126", "ssp245", "ssp370", "ssp534-over", "ssp585"}


def collect_combos():
    """Unique (source, experiment) pairs from cmip6 cache filenames."""
    combos = set()
    for path in glob.glob(os.path.join(ROOT, "**/cache/cmip6/**/*.nc"),
                          recursive=True):
        base = os.path.basename(path)
        # split off model prefix, then find the experiment token
        for model in MODELS:
            if base.startswith(model + "_"):
                rest = base[len(model) + 1:]
                for exp in EXPERIMENTS:
                    if rest.startswith(exp + "_") or rest == exp + ".nc":
                        combos.add((model, exp))
                break
    return sorted(combos)


def build_catalog_map():
    """(source, experiment) -> (activity_id, institution_id) from the CSV."""
    m = {}
    with open(CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["source_id"], row["experiment_id"])
            if key[0] in MODELS and key[1] in EXPERIMENTS and key not in m:
                m[key] = (row["activity_id"], row["institution_id"])
    return m


def fetch_doi(act, inst, src, exp):
    url = API.format(act=act, inst=inst, src=src, exp=exp)
    try:
        with urllib.request.urlopen(url, timeout=40) as r:
            d = json.load(r)
        ident = d.get("identifier") or {}
        return ident.get("id"), (d.get("titles") or [""])[0]
    except Exception as e:  # noqa: BLE001
        return None, f"ERROR: {e}"


def main():
    combos = collect_combos()
    catalog = build_catalog_map()
    print(f"# {len(combos)} unique (model, experiment) combos\n", file=sys.stderr)

    rows = []
    for src, exp in combos:
        mapped = catalog.get((src, exp))
        if not mapped:
            rows.append((src, exp, None, None, None, "NOT IN CATALOG"))
            continue
        act, inst = mapped
        doi, title = fetch_doi(act, inst, src, exp)
        rows.append((src, exp, act, inst, doi, title))
        print(f"{src:14s} {exp:14s} {act:12s} {inst:10s} "
              f"{doi or 'NO DOI':28s} {title}", file=sys.stderr)

    dois = sorted({r[4] for r in rows if r[4]})
    print("\n# ==== UNIQUE SOURCE DOIs ====")
    for d in dois:
        print(f"https://doi.org/{d}")

    out = os.path.join(ROOT, "scripts/cmip6_doi_map.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "experiment_id", "activity_id",
                    "institution_id", "doi", "title"])
        w.writerows(rows)
    print(f"\n# wrote per-combo map -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
