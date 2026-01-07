#!/usr/bin/env python3
"""
Combined plot of all DOE data: Argonne + Oak Ridge models.
"""

import warnings
warnings.filterwarnings("ignore")

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(base_dirs: list[Path], model: str, scenario: str) -> xr.DataArray:
    """Load and compute global annual mean from downloaded files."""
    # Try multiple possible directory structures
    possible_paths = []
    for base in base_dirs:
        possible_paths.extend([
            base / model / scenario,
            base / model / scenario / "Argonne",
            base / model / scenario / "Oak Ridge",
        ])

    files = []
    for path in possible_paths:
        if path.exists():
            files = list(path.glob("*.nc"))
            if files:
                break

    if not files:
        return None

    ds = xr.open_mfdataset(files, combine="by_coords")
    tas = ds["tas"]

    weights = np.cos(np.deg2rad(ds.lat))
    global_mean = tas.weighted(weights).mean(dim=["lat", "lon"])
    annual = global_mean.groupby("time.year").mean("time")

    return annual.compute()


def main():
    print("Loading data from Argonne and Oak Ridge...")

    # All data directories
    all_dirs = [
        Path("data/doe_nodes"),
        Path("data/argonne"),
        Path("data/ornl"),
    ]

    # All models with their source attribution
    models = {
        # Argonne models
        "GFDL-ESM4": "Argonne",
        "MIROC6": "Argonne",
        "NorESM2-LM": "Argonne",
        # Oak Ridge models
        "GISS-E2-1-H": "Oak Ridge",
        "UKESM1-0-LL": "Oak Ridge",
    }

    scenarios = ["ssp126", "ssp585"]

    # Load all data
    results = {}
    for model, source in models.items():
        results[model] = {"source": source}
        for scenario in scenarios:
            data = load_data(all_dirs, model, scenario)
            if data is not None:
                results[model][scenario] = data
                print(f"  {model} {scenario}: {len(data)} years")

    # Create plot
    print("\nGenerating combined plot...")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Colors by scenario
    colors = {"ssp126": "#2166ac", "ssp585": "#b2182b"}

    # Line styles by model
    linestyles = {
        "GFDL-ESM4": "-",
        "MIROC6": "--",
        "NorESM2-LM": ":",
        "GISS-E2-1-H": "-.",
        "UKESM1-0-LL": (0, (3, 1, 1, 1)),  # densely dashdotted
    }

    # Markers by source
    markers = {"Argonne": "o", "Oak Ridge": "s"}

    for model, data in results.items():
        source = data["source"]
        for scenario in scenarios:
            if scenario not in data:
                continue

            annual = data[scenario]
            temp_c = np.squeeze(annual.values) - 273.15
            years = annual.year.values

            # Limit to 2015-2100
            mask = (years >= 2015) & (years <= 2100)
            years = years[mask]
            temp_c = temp_c[mask]

            ax.plot(
                years,
                temp_c,
                color=colors[scenario],
                linestyle=linestyles[model],
                linewidth=2,
                alpha=0.85,
                label=f"{model} ({scenario}) [{source}]",
                marker=markers[source],
                markevery=15,
                markersize=5,
            )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Global Mean Temperature (°C)", fontsize=12)
    ax.set_title(
        "CMIP6 Global Temperature Projections\n"
        "Data from DOE: Argonne National Lab (ALCF) + Oak Ridge National Lab (ORNL)",
        fontsize=14
    )

    # Create legend with two columns
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2015, 2100)

    # Add scenario annotations
    ax.text(0.98, 0.12, "Low emissions (SSP1-2.6)",
            transform=ax.transAxes, ha="right", va="bottom",
            color=colors["ssp126"], fontsize=11, fontweight="bold")
    ax.text(0.98, 0.88, "High emissions (SSP5-8.5)",
            transform=ax.transAxes, ha="right", va="top",
            color=colors["ssp585"], fontsize=11, fontweight="bold")

    plt.tight_layout()

    output_path = Path("temperature_combined_doe.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {output_path.absolute()}")

    # Summary table
    print("\n" + "=" * 70)
    print("Summary: Warming by 2100 (relative to 2015-2025)")
    print("=" * 70)
    print(f"{'Model':<18} {'Source':<12} {'SSP1-2.6':>10} {'SSP5-8.5':>10}")
    print("-" * 70)

    for model in ["GFDL-ESM4", "MIROC6", "NorESM2-LM", "GISS-E2-1-H", "UKESM1-0-LL"]:
        data = results.get(model, {})
        source = data.get("source", "?")

        warmings = []
        for scenario in scenarios:
            if scenario in data:
                annual = data[scenario]
                years = annual.year.values
                mask_end = (years >= 2090) & (years <= 2100)
                mask_start = (years >= 2015) & (years <= 2025)

                temp_end = float(np.squeeze(annual.values[mask_end]).mean()) - 273.15
                temp_start = float(np.squeeze(annual.values[mask_start]).mean()) - 273.15
                warming = temp_end - temp_start
                warmings.append(f"+{warming:.1f}°C")
            else:
                warmings.append("N/A")

        print(f"{model:<18} {source:<12} {warmings[0]:>10} {warmings[1]:>10}")


if __name__ == "__main__":
    main()
