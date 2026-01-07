#!/usr/bin/env python3
"""
GPP (Gross Primary Production) change analysis using CESM2 under SSP5-8.5.

Compares present-day (1995-2014) vs end-of-century (2081-2100) GPP.
"""

import warnings
warnings.filterwarnings("ignore")

import requests
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


ESGF_SEARCH = "https://esgf-node.llnl.gov/esg-search/search"

# Analysis configuration
MODEL = "CESM2"
MEMBER = "r11i1p1f1"
VARIABLE = "gpp"
TABLE = "Lmon"

# Time periods for comparison
PRESENT = (1995, 2014)
FUTURE = (2081, 2100)

DATA_DIR = Path("data/gpp")


def search_files(experiment: str) -> list[str]:
    """Search for GPP files at DOE nodes."""
    params = {
        "project": "CMIP6",
        "source_id": MODEL,
        "variable_id": VARIABLE,
        "table_id": TABLE,
        "experiment_id": experiment,
        "member_id": MEMBER,
        "type": "File",
        "format": "application/solr+json",
        "latest": "true",
        "limit": 50,
    }

    response = requests.get(ESGF_SEARCH, params=params, timeout=60)
    data = response.json()

    urls = []
    for doc in data.get("response", {}).get("docs", []):
        node = doc.get("data_node", "")
        # Prefer DOE nodes
        if "ornl" in node or "anl" in node:
            for url_entry in doc.get("url", []):
                parts = url_entry.split("|")
                if len(parts) >= 3 and parts[2] == "HTTPServer":
                    urls.append(parts[0])

    return sorted(set(urls))


def download_file(url: str, dest_dir: Path) -> Path:
    """Download a file."""
    filename = url.split("/")[-1]
    dest_path = dest_dir / filename

    if dest_path.exists():
        return dest_path

    print(f"    Downloading {filename}...", end=" ", flush=True)
    response = requests.get(url, timeout=600, stream=True)
    response.raise_for_status()

    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    size_mb = dest_path.stat().st_size / 1e6
    print(f"{size_mb:.1f} MB")
    return dest_path


def download_files(urls: list[str], dest_dir: Path) -> list[Path]:
    """Download multiple files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for url in urls:
        try:
            path = download_file(url, dest_dir)
            paths.append(path)
        except Exception as e:
            print(f"    Error: {e}")
    return sorted(paths)


def compute_period_mean(files: list[Path], start_year: int, end_year: int) -> xr.DataArray:
    """Compute mean GPP for a time period."""
    ds = xr.open_mfdataset(files, combine="by_coords")
    gpp = ds[VARIABLE]

    # Select time period
    gpp_period = gpp.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    # Compute annual mean, then period mean
    annual = gpp_period.groupby("time.year").mean("time")
    period_mean = annual.mean("year")

    return period_mean.compute()


def main():
    print("=" * 70)
    print("GPP Change Analysis: CESM2 SSP5-8.5")
    print(f"Present: {PRESENT[0]}-{PRESENT[1]} vs Future: {FUTURE[0]}-{FUTURE[1]}")
    print("=" * 70)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download historical data
    print("\n1. Historical data (for present-day baseline):")
    hist_urls = search_files("historical")
    print(f"   Found {len(hist_urls)} files")
    hist_files = download_files(hist_urls, DATA_DIR / "historical")

    # Download SSP585 data
    print("\n2. SSP5-8.5 data (for future projection):")
    ssp_urls = search_files("ssp585")
    print(f"   Found {len(ssp_urls)} files")
    ssp_files = download_files(ssp_urls, DATA_DIR / "ssp585")

    if not hist_files or not ssp_files:
        print("\nError: Could not download required data")
        return

    # Compute period means
    print("\n3. Computing GPP means...")
    print(f"   Present ({PRESENT[0]}-{PRESENT[1]})...", end=" ", flush=True)
    gpp_present = compute_period_mean(hist_files, PRESENT[0], PRESENT[1])
    print("done")

    print(f"   Future ({FUTURE[0]}-{FUTURE[1]})...", end=" ", flush=True)
    gpp_future = compute_period_mean(ssp_files, FUTURE[0], FUTURE[1])
    print("done")

    # Compute change
    print("\n4. Computing changes...")

    # Absolute change (kg C m-2 s-1)
    gpp_change = gpp_future - gpp_present

    # Percent change
    # Avoid division by zero for areas with no vegetation
    gpp_pct_change = xr.where(
        gpp_present > 1e-12,
        (gpp_change / gpp_present) * 100,
        np.nan
    )

    # Convert to more intuitive units: g C m-2 day-1
    # 1 kg C m-2 s-1 = 86400 * 1000 g C m-2 day-1 = 86400000 g C m-2 day-1
    # Actually GPP is typically ~1e-8 to 1e-6 kg C m-2 s-1
    # Let's convert to g C m-2 day-1 for readability
    scale = 86400 * 1000  # kg/s to g/day
    gpp_present_gday = gpp_present * scale
    gpp_future_gday = gpp_future * scale
    gpp_change_gday = gpp_change * scale

    # Global statistics (land only, where GPP > 0)
    land_mask = gpp_present > 1e-12

    # Area-weighted global mean
    weights = np.cos(np.deg2rad(gpp_present.lat))

    present_global = float(gpp_present_gday.where(land_mask).weighted(weights).mean().values)
    future_global = float(gpp_future_gday.where(land_mask).weighted(weights).mean().values)
    change_global = future_global - present_global
    pct_change_global = (change_global / present_global) * 100

    print(f"\n   Global land mean GPP:")
    print(f"   Present: {present_global:.2f} g C m⁻² day⁻¹")
    print(f"   Future:  {future_global:.2f} g C m⁻² day⁻¹")
    print(f"   Change:  {change_global:+.2f} g C m⁻² day⁻¹ ({pct_change_global:+.1f}%)")

    # Create plots
    print("\n5. Generating maps...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                             subplot_kw={'projection': None})

    # Get lat/lon
    lon = gpp_present.lon.values
    lat = gpp_present.lat.values

    # 1. Present-day GPP
    ax = axes[0, 0]
    im = ax.pcolormesh(lon, lat, gpp_present_gday.values,
                       cmap='YlGn', vmin=0, vmax=15, shading='auto')
    ax.set_title(f'Present-day GPP ({PRESENT[0]}-{PRESENT[1]})', fontsize=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, label='g C m⁻² day⁻¹', shrink=0.8)

    # 2. Future GPP
    ax = axes[0, 1]
    im = ax.pcolormesh(lon, lat, gpp_future_gday.values,
                       cmap='YlGn', vmin=0, vmax=15, shading='auto')
    ax.set_title(f'Future GPP ({FUTURE[0]}-{FUTURE[1]}, SSP5-8.5)', fontsize=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, label='g C m⁻² day⁻¹', shrink=0.8)

    # 3. Absolute change
    ax = axes[1, 0]
    im = ax.pcolormesh(lon, lat, gpp_change_gday.values,
                       cmap='RdBu', vmin=-3, vmax=3, shading='auto')
    ax.set_title('GPP Change (Future - Present)', fontsize=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, label='Δ g C m⁻² day⁻¹', shrink=0.8)

    # 4. Percent change
    ax = axes[1, 1]
    im = ax.pcolormesh(lon, lat, gpp_pct_change.values,
                       cmap='RdBu', vmin=-50, vmax=50, shading='auto')
    ax.set_title('GPP Percent Change', fontsize=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, label='% change', shrink=0.8)

    plt.suptitle(f'Gross Primary Production (GPP) Change: CESM2 SSP5-8.5\n'
                 f'Global land mean: {present_global:.1f} → {future_global:.1f} g C m⁻² day⁻¹ '
                 f'({pct_change_global:+.1f}%)',
                 fontsize=14, y=1.02)

    plt.tight_layout()

    output_path = Path("gpp_change_cesm2.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {output_path.absolute()}")

    # Regional breakdown
    print("\n" + "=" * 70)
    print("Regional GPP Changes")
    print("=" * 70)

    regions = {
        "Tropics (23S-23N)": {"lat_min": -23, "lat_max": 23},
        "Northern mid-lat (30N-60N)": {"lat_min": 30, "lat_max": 60},
        "Southern mid-lat (30S-60S)": {"lat_min": -60, "lat_max": -30},
        "Arctic (>60N)": {"lat_min": 60, "lat_max": 90},
    }

    print(f"\n{'Region':<30} {'Present':>12} {'Future':>12} {'Change':>12}")
    print("-" * 70)

    for region_name, bounds in regions.items():
        lat_mask = (gpp_present.lat >= bounds["lat_min"]) & (gpp_present.lat <= bounds["lat_max"])

        present_reg = gpp_present_gday.where(lat_mask & land_mask)
        future_reg = gpp_future_gday.where(lat_mask & land_mask)

        weights_reg = weights.where(lat_mask).fillna(0)

        p = float(present_reg.weighted(weights_reg).mean().values)
        f = float(future_reg.weighted(weights_reg).mean().values)
        c = f - p
        pct = (c / p) * 100 if p > 0 else 0

        print(f"{region_name:<30} {p:>10.2f} {f:>10.2f} {c:>+10.2f} ({pct:+.0f}%)")


if __name__ == "__main__":
    main()
