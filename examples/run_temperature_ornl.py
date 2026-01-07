#!/usr/bin/env python3
"""
Temperature projection analysis using Oak Ridge National Lab (ORNL) data.

Downloads models not available/used from Argonne.
"""

import warnings
warnings.filterwarnings("ignore")

import requests
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


ESGF_SEARCH = "https://esgf-node.llnl.gov/esg-search/search"
DATA_NODE = "esgf-node.ornl.gov"

# Models with good coverage at ORNL (not already used from Argonne)
MODELS = {
    "GISS-E2-1-H": "r1i1p1f2",    # NASA GISS
    "UKESM1-0-LL": "r4i1p1f2",    # UK Met Office
}

SCENARIOS = ["ssp126", "ssp585"]
VARIABLE = "tas"
TABLE = "Amon"

DATA_DIR = Path("data/ornl")


def get_http_urls(source_id: str, experiment_id: str, member_id: str) -> list[str]:
    """Get HTTPServer URLs from ORNL."""
    params = {
        "project": "CMIP6",
        "source_id": source_id,
        "experiment_id": experiment_id,
        "variable_id": VARIABLE,
        "table_id": TABLE,
        "member_id": member_id,
        "data_node": DATA_NODE,
        "type": "File",
        "format": "application/solr+json",
        "latest": "true",
        "limit": 100,
    }

    response = requests.get(ESGF_SEARCH, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    urls = []
    for doc in data.get("response", {}).get("docs", []):
        for url_entry in doc.get("url", []):
            parts = url_entry.split("|")
            if len(parts) >= 3 and parts[2] == "HTTPServer":
                urls.append(parts[0])

    return sorted(set(urls))


def download_file(url: str, dest_dir: Path) -> Path:
    """Download a single file."""
    filename = url.split("/")[-1]
    dest_path = dest_dir / filename

    if dest_path.exists():
        return dest_path

    response = requests.get(url, timeout=600, stream=True)
    response.raise_for_status()

    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return dest_path


def download_files(urls: list[str], dest_dir: Path) -> list[Path]:
    """Download files sequentially."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i, url in enumerate(urls):
        try:
            path = download_file(url, dest_dir)
            paths.append(path)
            print(".", end="", flush=True)
        except Exception as e:
            print(f"x", end="", flush=True)

    return sorted(paths)


def compute_global_annual_mean(files: list[Path]) -> xr.DataArray:
    """Load and compute area-weighted global annual mean."""
    ds = xr.open_mfdataset(files, combine="by_coords")
    tas = ds[VARIABLE]

    weights = np.cos(np.deg2rad(ds.lat))
    global_mean = tas.weighted(weights).mean(dim=["lat", "lon"])

    annual = global_mean.groupby("time.year").mean("time")
    return annual.compute()


def main():
    print("=" * 70)
    print(f"Temperature Analysis - Oak Ridge (ORNL)")
    print(f"Data node: {DATA_NODE}")
    print("=" * 70)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for model, member in MODELS.items():
        results[model] = {}

        for scenario in SCENARIOS:
            print(f"\n{model} / {scenario} ({member}):", end=" ")

            urls = get_http_urls(model, scenario, member)
            if not urls:
                print("no data")
                continue

            print(f"{len(urls)} files")

            # Download
            dest_dir = DATA_DIR / model / scenario
            print(f"  Downloading ", end="", flush=True)
            files = download_files(urls, dest_dir)
            print()

            if not files:
                print("  Failed")
                continue

            total_mb = sum(f.stat().st_size for f in files) / 1e6
            print(f"  Downloaded: {total_mb:.1f} MB")

            # Process
            print(f"  Processing...", end=" ", flush=True)
            try:
                annual = compute_global_annual_mean(files)
                results[model][scenario] = annual
                years = annual.year.values
                print(f"done ({int(years.min())}-{int(years.max())})")
            except Exception as e:
                print(f"error: {str(e)[:50]}")

    # Check results
    total = sum(len(s) for s in results.values())
    if total == 0:
        print("\nNo data processed.")
        return

    # Plot
    print("\n" + "=" * 70)
    print("Generating plot...")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {"ssp126": "#2166ac", "ssp585": "#b2182b"}
    linestyles = {"GISS-E2-1-H": "-", "UKESM1-0-LL": "--"}

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
        f"CMIP6 Temperature Projections\nData from Oak Ridge National Lab ({DATA_NODE})",
        fontsize=13
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2015, 2100)

    plt.tight_layout()

    output_path = Path("temperature_ornl.png")
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
                print(f"  {model:15} {scenario}: +{warming:.1f}°C")


if __name__ == "__main__":
    main()
