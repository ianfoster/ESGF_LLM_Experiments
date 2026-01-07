#!/usr/bin/env python3
"""
Temperature projection analysis using Argonne ALCF data via Globus HTTPS.

Downloads NetCDF files from Argonne's Globus endpoint and analyzes them.
"""

import warnings
warnings.filterwarnings("ignore")

import requests
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os


# ESGF Search API
ESGF_SEARCH = "https://esgf-node.llnl.gov/esg-search/search"

# Use ONLY Argonne node
DATA_NODE = "eagle.alcf.anl.gov"

# Models with data at Argonne
MODELS = ["MPI-ESM1-2-LR", "CanESM5", "ACCESS-ESM1-5"]
SCENARIOS = ["ssp126", "ssp585"]
VARIABLE = "tas"
TABLE = "Amon"


def get_http_urls(source_id: str, experiment_id: str) -> tuple[list[str], str]:
    """Get HTTPServer URLs from Argonne for a model/experiment."""
    params = {
        "project": "CMIP6",
        "source_id": source_id,
        "experiment_id": experiment_id,
        "variable_id": VARIABLE,
        "table_id": TABLE,
        "data_node": DATA_NODE,
        "type": "File",
        "format": "application/solr+json",
        "latest": "true",
        "limit": 100,
    }

    response = requests.get(ESGF_SEARCH, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    # Group by member_id
    by_member = {}
    for doc in data.get("response", {}).get("docs", []):
        mem = doc.get("member_id", ["unknown"])[0]
        if mem not in by_member:
            by_member[mem] = []

        for url_entry in doc.get("url", []):
            parts = url_entry.split("|")
            if len(parts) >= 3 and parts[2] == "HTTPServer":
                by_member[mem].append(parts[0])

    if not by_member:
        return [], None

    # Prefer r1i1p1f1
    chosen = "r1i1p1f1" if "r1i1p1f1" in by_member else sorted(by_member.keys())[0]
    return sorted(set(by_member[chosen])), chosen


def download_file(url: str, dest_dir: Path) -> Path:
    """Download a single file."""
    filename = url.split("/")[-1]
    dest_path = dest_dir / filename

    if dest_path.exists():
        return dest_path

    response = requests.get(url, timeout=300)
    response.raise_for_status()

    dest_path.write_bytes(response.content)
    return dest_path


def download_files(urls: list[str], dest_dir: Path) -> list[Path]:
    """Download files in parallel."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(download_file, url, dest_dir) for url in urls]
        for future in futures:
            try:
                paths.append(future.result())
            except Exception as e:
                print(f"Download error: {e}")

    return sorted(paths)


def compute_global_mean(files: list[Path]) -> xr.DataArray:
    """Load files and compute area-weighted global mean."""
    ds = xr.open_mfdataset(files, combine="by_coords")
    tas = ds[VARIABLE]

    # Area weighting
    weights = np.cos(np.deg2rad(ds.lat))
    global_mean = tas.weighted(weights).mean(dim=["lat", "lon"])

    return global_mean


def main():
    print("=" * 70)
    print(f"Temperature Analysis - Argonne ALCF ({DATA_NODE})")
    print("=" * 70)

    # Use temp directory for downloads
    data_dir = Path("data/argonne")
    data_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for model in MODELS:
        results[model] = {}
        model_dir = data_dir / model

        for scenario in SCENARIOS:
            print(f"\n{model} / {scenario}:", end=" ")

            urls, member = get_http_urls(model, scenario)
            if not urls:
                print("no data")
                continue

            print(f"{len(urls)} files ({member})")

            # Download
            print(f"  Downloading...", end=" ", flush=True)
            scenario_dir = model_dir / scenario
            files = download_files(urls, scenario_dir)

            if not files:
                print("failed")
                continue

            total_mb = sum(f.stat().st_size for f in files) / 1e6
            print(f"{total_mb:.1f} MB")

            # Process
            print(f"  Processing...", end=" ", flush=True)
            try:
                global_mean = compute_global_mean(files)
                annual = global_mean.groupby("time.year").mean("time")
                annual = annual.compute()
                results[model][scenario] = annual

                years = annual.year.values
                print(f"done ({int(years.min())}-{int(years.max())})")
            except Exception as e:
                print(f"error: {str(e)[:50]}")

    # Check results
    total = sum(len(s) for s in results.values())
    if total == 0:
        print("\nNo data processed successfully.")
        return

    # Plot
    print("\n" + "=" * 70)
    print("Generating plot...")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {"ssp126": "#2166ac", "ssp585": "#b2182b"}
    linestyles = {"MPI-ESM1-2-LR": "-", "CanESM5": "--", "ACCESS-ESM1-5": ":"}

    for model, scenarios in results.items():
        for scenario, data in scenarios.items():
            if data is not None and len(data) > 0:
                temp_c = np.squeeze(data.values) - 273.15
                years = data.year.values

                ax.plot(
                    years,
                    temp_c,
                    color=colors[scenario],
                    linestyle=linestyles.get(model, "-"),
                    linewidth=2,
                    alpha=0.9,
                    label=f"{model} ({scenario})",
                )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Global Mean Temperature (°C)", fontsize=12)
    ax.set_title(
        f"CMIP6 Temperature Projections\nData from Argonne ALCF ({DATA_NODE})",
        fontsize=13
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2015, 2100)

    plt.tight_layout()

    output_path = Path("temperature_argonne.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {output_path.absolute()}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Warming by 2100 (relative to 2015-2025)")
    print("=" * 70)

    for model in MODELS:
        scenarios = results.get(model, {})
        for scenario in SCENARIOS:
            data = scenarios.get(scenario)
            if data is not None and len(data) > 0:
                temp_end = float(np.squeeze(data.sel(year=slice(2090, 2100)).mean().values)) - 273.15
                temp_start = float(np.squeeze(data.sel(year=slice(2015, 2025)).mean().values)) - 273.15
                warming = temp_end - temp_start
                print(f"  {model:18} {scenario}: +{warming:.1f}°C")


if __name__ == "__main__":
    main()
