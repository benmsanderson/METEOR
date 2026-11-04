from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OMNIA_FILE = Path(
    "/div/no-backup-nac/users/masan/GRAFITE/temp_indata/"
    "baseline_v1_2_new+softlink_omnia_em_gases_vupdate_2024_WMO_added_new_em_DiamondWP-scenarios.txt"
)
DEFAULT_SCEN7_FILE = Path(
    "/div/no-backup-nac/users/maurad/METEOR/data/FASTMIP_phase2/scenario_data/"
    "scen7-M_em_gases_vupdate_2022_AR6.txt"
)
DEFAULT_OUTPUT_DIR = Path("../data/emissions_comparison_plots")
DEFAULT_SKIPROWS = [1, 2, 3]
SKIPROWS_CANDIDATES = (DEFAULT_SKIPROWS,)
SPECIES_PER_PAGE = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one Omnia emissions file against one scen7 emissions file "
            "and write paginated 3x3 species plots."
        )
    )
    parser.add_argument("--omnia-file", type=Path, default=DEFAULT_OMNIA_FILE)
    parser.add_argument("--scen7-file", type=Path, default=DEFAULT_SCEN7_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument(
        "--skiprows",
        type=int,
        nargs="*",
        default=DEFAULT_SKIPROWS,
        help="Explicit row numbers to skip for both files. Default: 1 2 3.",
    )
    return parser.parse_args()


def _load_emissions_table_once(file_path: Path, skiprows: int | list[int]) -> pd.DataFrame:
    frame = pd.read_csv(file_path, sep=r"\s+", skiprows=skiprows, engine="python")
    frame = frame.dropna(axis=1, how="all")
    if frame.empty or frame.shape[1] < 2:
        raise ValueError("table is empty or has fewer than two columns")

    frame.columns = [str(column).strip() for column in frame.columns]
    lower_to_original = {column.lower(): column for column in frame.columns}
    year_column = lower_to_original.get("year", frame.columns[0])

    year_values = pd.to_numeric(frame[year_column], errors="coerce")
    if year_values.isna().all():
        raise ValueError("could not identify a numeric year column")

    frame = frame.assign(year=year_values)
    frame = frame.loc[frame["year"].notna()].copy()
    if frame.empty:
        raise ValueError("all rows were removed while parsing the year column")

    frame["year"] = frame["year"].astype(int)
    frame = frame.drop_duplicates(subset="year", keep="first")
    frame = frame.set_index("year")

    if year_column in frame.columns:
        frame = frame.drop(columns=[year_column])

    numeric_columns = {}
    for column in frame.columns:
        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        if numeric_values.notna().any():
            numeric_columns[column.strip()] = numeric_values

    if not numeric_columns:
        raise ValueError("no numeric species columns were found")

    result = pd.DataFrame(numeric_columns, index=frame.index).sort_index()
    if result.index.min() < 1000 or result.index.max() > 4000:
        raise ValueError("parsed year range is implausible")
    return result


def load_emissions_table(
    file_path: Path, skiprows: int | list[int] | None = None
) -> tuple[pd.DataFrame, int | list[int]]:
    if not file_path.exists():
        raise FileNotFoundError(f"Emissions file not found: {file_path}")

    candidates = (skiprows,) if skiprows is not None else SKIPROWS_CANDIDATES
    errors: list[str] = []
    for candidate in candidates:
        try:
            table = _load_emissions_table_once(file_path, candidate)
            return table, candidate
        except Exception as exc:  # noqa: BLE001
            errors.append(f"skiprows={candidate}: {exc}")

    message = "\n".join(errors)
    raise ValueError(f"Failed to parse emissions table {file_path}\n{message}")


def load_emissions_tables_with_shared_skiprows(
    omnia_file: Path,
    scen7_file: Path,
    skiprows: int | list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int | list[int]]:
    candidates = (skiprows,) if skiprows is not None else SKIPROWS_CANDIDATES
    errors: list[str] = []

    for candidate in candidates:
        try:
            omnia, _ = load_emissions_table(omnia_file, candidate)
            scen7, _ = load_emissions_table(scen7_file, candidate)
            return omnia, scen7, candidate
        except Exception as exc:  # noqa: BLE001
            errors.append(f"skiprows={candidate}: {exc}")

    message = "\n".join(errors)
    raise ValueError(
        "Failed to parse both emissions tables with a shared skiprows value\n"
        f"{message}"
    )


def align_tables(
    omnia: pd.DataFrame,
    scen7: pd.DataFrame,
    start_year: int | None,
    end_year: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_years = omnia.index.intersection(scen7.index)
    if start_year is not None:
        common_years = common_years[common_years >= start_year]
    if end_year is not None:
        common_years = common_years[common_years <= end_year]
    if len(common_years) == 0:
        raise ValueError("No overlapping years found between the two emissions tables")

    common_species = sorted(set(omnia.columns).intersection(scen7.columns))
    if not common_species:
        raise ValueError("No overlapping species columns found between the two emissions tables")

    omnia_aligned = omnia.loc[common_years, common_species]
    scen7_aligned = scen7.loc[common_years, common_species]
    difference = omnia_aligned - scen7_aligned
    return omnia_aligned, scen7_aligned, difference


def build_summary(
    omnia: pd.DataFrame,
    scen7: pd.DataFrame,
    difference: pd.DataFrame,
) -> pd.DataFrame:
    summary = pd.DataFrame(index=difference.columns)
    summary.index.name = "species"
    summary["omnia_mean"] = omnia.mean(axis=0)
    summary["scen7_mean"] = scen7.mean(axis=0)
    summary["difference_mean"] = difference.mean(axis=0)
    summary["difference_abs_mean"] = difference.abs().mean(axis=0)
    summary["difference_abs_max"] = difference.abs().max(axis=0)
    summary["difference_min"] = difference.min(axis=0)
    summary["difference_max"] = difference.max(axis=0)
    return summary.sort_values("difference_abs_max", ascending=False)


def _format_page_species(species_names: list[str], page_index: int, total_pages: int) -> str:
    first_species = species_names[0]
    last_species = species_names[-1]
    return f"Page {page_index}/{total_pages}: {first_species} to {last_species}"


def plot_species_pages(
    omnia: pd.DataFrame,
    scen7: pd.DataFrame,
    difference: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    omnia_label: str,
    scen7_label: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    species_names = list(summary.index)
    total_pages = math.ceil(len(species_names) / SPECIES_PER_PAGE)
    written_paths: list[Path] = []

    for page_number in range(total_pages):
        page_species = species_names[
            page_number * SPECIES_PER_PAGE : (page_number + 1) * SPECIES_PER_PAGE
        ]
        fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
        axes_flat = axes.flat

        for axis, species in zip(axes_flat, page_species):
            axis.plot(
                omnia.index,
                omnia[species],
                label=omnia_label,
                color="C0",
                linewidth=1.5,
            )
            axis.plot(
                scen7.index,
                scen7[species],
                label=scen7_label,
                color="C1",
                linewidth=1.5,
            )
            axis.plot(
                difference.index,
                difference[species],
                label="difference",
                color="black",
                linewidth=1.0,
                linestyle="--",
                alpha=0.8,
            )
            axis.axhline(0.0, color="0.6", linewidth=0.8, linestyle=":")
            axis.set_title(species)
            axis.grid(True, alpha=0.25)

            max_abs_diff = summary.loc[species, "difference_abs_max"]
            mean_abs_diff = summary.loc[species, "difference_abs_mean"]
            axis.text(
                0.02,
                0.98,
                f"max |Δ|={max_abs_diff:.3g}\nmean |Δ|={mean_abs_diff:.3g}",
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

        for axis in list(axes_flat)[len(page_species) :]:
            axis.set_visible(False)

        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", ncol=3)

        fig.suptitle(
            "Omnia vs scen7 emissions comparison\n"
            + _format_page_species(page_species, page_number + 1, total_pages),
            ha="left",
            x=0.06,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        output_path = output_dir / f"comparison_page_{page_number + 1:02d}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        written_paths.append(output_path)

    return written_paths


def write_summary(summary: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "species_difference_summary.csv"
    summary.to_csv(summary_path)
    return summary_path


def main() -> None:
    args = parse_args()

    omnia, scen7, shared_skiprows = load_emissions_tables_with_shared_skiprows(
        args.omnia_file,
        args.scen7_file,
        args.skiprows,
    )
    print(omnia.head())
    print(scen7.head())
    print(shared_skiprows)
    omnia_aligned, scen7_aligned, difference = align_tables(
        omnia,
        scen7,
        args.start_year,
        args.end_year,
    )
    summary = build_summary(omnia_aligned, scen7_aligned, difference)

    output_dir = args.output_dir
    written_plots = plot_species_pages(
        omnia_aligned,
        scen7_aligned,
        difference,
        summary,
        output_dir,
        omnia_label=args.omnia_file.stem,
        scen7_label=args.scen7_file.stem,
    )
    summary_path = write_summary(summary, output_dir)

    print(f"Loaded both files with skiprows={shared_skiprows}")
    print(f"  Omnia: {args.omnia_file}")
    print(f"  scen7: {args.scen7_file}")
    print(
        f"Compared {len(summary)} common species over years "
        f"{omnia_aligned.index.min()}-{omnia_aligned.index.max()}"
    )
    print(f"Wrote summary: {summary_path}")
    for plot_path in written_plots:
        print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()