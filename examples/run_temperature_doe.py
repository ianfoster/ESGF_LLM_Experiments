#!/usr/bin/env python3
"""
Temperature projection analysis using DOE data centers:
- Argonne National Lab (ALCF)
- Oak Ridge National Lab (ORNL)

Uses models with complete 2015-2100 coverage in single or few files.
"""

import warnings
warnings.filterwarnings("ignore")

import requests
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


# ESGF Search API
ESGF_SEARCH = "https://esgf-node.llnl.gov/esg-search/search"

# DOE data nodes
NODES = {
    "Argonne": "eagle.alcf.anl.gov",
    "Oak Ridge": "esgf-node.ornl.gov",
}

# Models with efficient file structure (full 2015-2100 in single/few files)
MODELS = ["GFDL-ESM4", "MIROC6", "NorESM2-LM"]
SCENARIOS = ["ssp126", "ssp585"]
VARIABLE = "tas"
TABLE = "Amon"

# Local data directory
DATA_DIR = Path("data/doe_nodes")


def get_http_urls(source_id: str, experiment_id: str, data_node: str) -> tuple[list[str], str]:
    """Get HTTPServer URLs for a model/experiment from a specific node."""
    params = {
        "project": "CMIP6",
        "source_id": source_id,
        "experiment_id": experiment_id,
        "variable_id": VARIABLE,
        "table_id": TABLE,
        "data_node": data_node,
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

    response = requests.get(url, timeout=600, stream=True)
    response.raise_for_status()

    # Stream to file
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return dest_path


def download_files(urls: list[str], dest_dir: Path) -> list[Path]:
    """Download files (sequentially to avoid overwhelming servers)."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for url in urls:
        try:
            path = download_file(url, dest_dir)
            paths.append(path)
        except Exception as e:
            print(f"    Download error: {e}")

    return sorted(paths)


def compute_global_annual_mean(files: list[Path]) -> xr.DataArray:
    """Load files and compute area-weighted global annual mean."""
    ds = xr.open_mfdataset(files, combine="by_coords")
    tas = ds[VARIABLE]

    # Area weighting
    weights = np.cos(np.deg2rad(ds.lat))
    global_mean = tas.weighted(weights).mean(dim=["lat", "lon"])

    # Annual mean
    annual = global_mean.groupby("time.year").mean("time")
    return annual.compute()


def main():
    print("=" * 70)
    print("Temperature Analysis - DOE Data Centers")
    print("Argonne (ALCF) + Oak Ridge (ORNL)")
    print("=" * 70)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Try to get data from either node (prefer Argonne, fallback to Oak Ridge)
    results = {}

    for model in MODELS:
        results[model] = {}

        for scenario in SCENARIOS:
            print(f"\n{model} / {scenario}:")

            # Try each node
            for node_name, node_host in NODES.items():
                urls, member = get_http_urls(model, scenario, node_host)

                if not urls:
                    continue

                print(f"  {node_name}: {len(urls)} files ({member})")

                # Download
                dest_dir = DATA_DIR / model / scenario / node_name
                print(f"    Downloading...", end=" ", flush=True)
                files = download_files(urls, dest_dir)

                if not files:
                    print("failed")
                    continue

                total_mb = sum(f.stat().st_size for f in files) / 1e6
                print(f"{total_mb:.1f} MB")

                # Process
                print(f"    Processing...", end=" ", flush=True)
                try:
                    annual = compute_global_annual_mean(files)
                    results[model][scenario] = {
                        "data": annual,
                        "node": node_name,
                    }
                    years = annual.year.values
                    print(f"done ({int(years.min())}-{int(years.max())})")
                    break  # Success, don't try other nodes

                except Exception as e:
                    print(f"error: {str(e)[:40]}")
                    continue

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
    linestyles = {"GFDL-ESM4": "-", "MIROC6": "--", "NorESM2-LM": ":"}
    markers = {"Argonne": "o", "Oak Ridge": "s"}

    for model, scenarios in results.items():
        for scenario, info in scenarios.items():
            data = info["data"]
            node = info["node"]

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
                    label=f"{model} ({scenario}) [{node}]",
                    marker=markers.get(node, ""),
                    markevery=10,
                    markersize=4,
                )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Global Mean Temperature (°C)", fontsize=12)
    ax.set_title(
        "CMIP6 Temperature Projections\n"
        "Data from DOE: Argonne (ALCF) & Oak Ridge (ORNL)",
        fontsize=13
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2015, 2100)

    plt.tight_layout()

    output_path = Path("temperature_doe.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {output_path.absolute()}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Warming by 2100 (relative to 2015-2025)")
    print("=" * 70)

    for model in MODELS:
        scenarios = results.get(model, {})
        for scenario in SCENARIOS:
            info = scenarios.get(scenario)
            if info:
                data = info["data"]
                node = info["node"]
                temp_end = float(np.squeeze(data.sel(year=slice(2090, 2100)).mean().values)) - 273.15
                temp_start = float(np.squeeze(data.sel(year=slice(2015, 2025)).mean().values)) - 273.15
                warming = temp_end - temp_start
                print(f"  {model:15} {scenario}: +{warming:.1f}°C  [{node}]")


if __name__ == "__main__":
    main()
