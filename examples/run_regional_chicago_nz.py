#!/usr/bin/env python3
"""
Regional temperature analysis for Chicago and New Zealand.

Uses data already downloaded from Argonne and Oak Ridge.
"""

import warnings
warnings.filterwarnings("ignore")

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Regional definitions
REGIONS = {
    "Chicago, IL": {
        "lat_min": 41.0,
        "lat_max": 43.0,
        "lon_min": -89.0,
        "lon_max": -87.0,
        "description": "Chicago area (41-43°N, 87-89°W)",
    },
    "New Zealand": {
        "lat_min": -47.0,
        "lat_max": -34.0,
        "lon_min": 166.0,
        "lon_max": 179.0,
        "description": "New Zealand (34-47°S, 166-179°E)",
    },
}

# Models and their data locations
MODELS = {
    "GFDL-ESM4": "Argonne",
    "MIROC6": "Argonne",
    "NorESM2-LM": "Argonne",
    "GISS-E2-1-H": "Oak Ridge",
    "UKESM1-0-LL": "Oak Ridge",
}

SCENARIOS = ["ssp126", "ssp585"]

# Data directories
DATA_DIRS = [
    Path("data/doe_nodes"),
    Path("data/argonne"),
    Path("data/ornl"),
]


def find_data_files(model: str, scenario: str) -> list[Path]:
    """Find NetCDF files for a model/scenario."""
    for base in DATA_DIRS:
        for subpath in [
            base / model / scenario,
            base / model / scenario / "Argonne",
            base / model / scenario / "Oak Ridge",
        ]:
            if subpath.exists():
                files = list(subpath.glob("*.nc"))
                if files:
                    return sorted(files)
    return []


def normalize_lon(lon_min: float, lon_max: float, ds_lon: xr.DataArray) -> tuple[float, float]:
    """Convert longitude to match dataset convention (0-360 or -180-180)."""
    if ds_lon.min() >= 0:  # Dataset uses 0-360
        if lon_min < 0:
            lon_min += 360
        if lon_max < 0:
            lon_max += 360
    return lon_min, lon_max


def extract_regional_mean(files: list[Path], region: dict) -> xr.DataArray:
    """Extract area-weighted regional mean temperature."""
    ds = xr.open_mfdataset(files, combine="by_coords")
    tas = ds["tas"]

    # Get longitude coordinate name
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    lat_name = "lat" if "lat" in ds.coords else "latitude"

    # Normalize longitude to dataset convention
    lon_min, lon_max = normalize_lon(
        region["lon_min"],
        region["lon_max"],
        ds[lon_name]
    )

    # Subset to region
    regional = tas.sel(
        **{
            lat_name: slice(region["lat_min"], region["lat_max"]),
            lon_name: slice(lon_min, lon_max),
        }
    )

    # Area-weighted mean
    weights = np.cos(np.deg2rad(ds[lat_name]))
    weights = weights.sel(**{lat_name: slice(region["lat_min"], region["lat_max"])})

    regional_mean = regional.weighted(weights).mean(dim=[lat_name, lon_name])

    # Annual mean
    annual = regional_mean.groupby("time.year").mean("time")

    return annual.compute()


def main():
    print("=" * 70)
    print("Regional Temperature Analysis")
    print("Chicago, IL vs New Zealand")
    print("=" * 70)

    # Store results by region
    results = {region: {} for region in REGIONS}

    for region_name, region_def in REGIONS.items():
        print(f"\n## {region_name}: {region_def['description']}")

        for model in MODELS:
            results[region_name][model] = {}

            for scenario in SCENARIOS:
                files = find_data_files(model, scenario)

                if not files:
                    continue

                print(f"  {model} {scenario}:", end=" ", flush=True)

                try:
                    annual = extract_regional_mean(files, region_def)
                    results[region_name][model][scenario] = annual

                    years = annual.year.values
                    print(f"OK ({int(years.min())}-{int(years.max())})")

                except Exception as e:
                    print(f"error: {str(e)[:40]}")

    # Create plots
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors = {"ssp126": "#2166ac", "ssp585": "#b2182b"}
    linestyles = {
        "GFDL-ESM4": "-",
        "MIROC6": "--",
        "NorESM2-LM": ":",
        "GISS-E2-1-H": "-.",
        "UKESM1-0-LL": (0, (3, 1, 1, 1)),
    }

    for idx, (region_name, region_def) in enumerate(REGIONS.items()):
        ax = axes[idx]

        for model in MODELS:
            model_data = results[region_name].get(model, {})

            for scenario in SCENARIOS:
                if scenario not in model_data:
                    continue

                data = model_data[scenario]
                temp_c = np.squeeze(data.values) - 273.15
                years = data.year.values

                # Limit to 2015-2100
                mask = (years >= 2015) & (years <= 2100)
                years = years[mask]
                temp_c = temp_c[mask]

                ax.plot(
                    years,
                    temp_c,
                    color=colors[scenario],
                    linestyle=linestyles[model],
                    linewidth=1.8,
                    alpha=0.85,
                    label=f"{model} ({scenario})",
                )

        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Temperature (°C)", fontsize=11)
        ax.set_title(f"{region_name}\n{region_def['description']}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2015, 2100)

    # Set same y-axis range for both plots
    all_ylims = [ax.get_ylim() for ax in axes]
    y_min = min(ylim[0] for ylim in all_ylims)
    y_max = max(ylim[1] for ylim in all_ylims)
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    # Add scenario labels after setting ylim
    for ax in axes:
        ax.text(0.98, 0.08, "SSP1-2.6", transform=ax.transAxes,
                ha="right", color=colors["ssp126"], fontsize=10, fontweight="bold")
        ax.text(0.98, 0.92, "SSP5-8.5", transform=ax.transAxes,
                ha="right", color=colors["ssp585"], fontsize=10, fontweight="bold")

    # Single legend for both plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8,
               bbox_to_anchor=(0.99, 0.5), framealpha=0.9)

    plt.suptitle("CMIP6 Regional Temperature Projections\nChicago, IL vs New Zealand",
                 fontsize=14, y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(right=0.82)

    output_path = Path("Artifacts/temperature_chicago_nz.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {output_path.absolute()}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary: Warming by 2100 (relative to 2015-2025)")
    print("=" * 70)

    for region_name in REGIONS:
        print(f"\n{region_name}:")
        print(f"{'Model':<18} {'SSP1-2.6':>10} {'SSP5-8.5':>10}")
        print("-" * 40)

        for model in MODELS:
            model_data = results[region_name].get(model, {})
            warmings = []

            for scenario in SCENARIOS:
                if scenario in model_data:
                    data = model_data[scenario]
                    years = data.year.values

                    mask_end = (years >= 2090) & (years <= 2100)
                    mask_start = (years >= 2015) & (years <= 2025)

                    if mask_end.any() and mask_start.any():
                        temp_end = float(np.squeeze(data.values[mask_end]).mean()) - 273.15
                        temp_start = float(np.squeeze(data.values[mask_start]).mean()) - 273.15
                        warming = temp_end - temp_start
                        warmings.append(f"+{warming:.1f}°C")
                    else:
                        warmings.append("N/A")
                else:
                    warmings.append("N/A")

            if warmings[0] != "N/A" or warmings[1] != "N/A":
                print(f"{model:<18} {warmings[0]:>10} {warmings[1]:>10}")


if __name__ == "__main__":
    main()
