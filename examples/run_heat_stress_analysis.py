#!/usr/bin/env python3
"""
Heat-Humidity Compound Events: Human Heat Stress Analysis

Analyzes changes in dangerous heat stress conditions using Wet Bulb Temperature (WBT),
the key metric for human physiological heat tolerance.

Key thresholds:
- WBT > 28°C: Dangerous for strenuous activity
- WBT > 32°C: Dangerous even at rest for acclimatized individuals
- WBT > 35°C: Fatal threshold - human body cannot cool itself

Uses CMIP6 temperature (tas) and relative humidity (hurs) data from ESGF.
"""

import warnings
warnings.filterwarnings("ignore")

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import json
import requests

# Configuration
ESGF_SEARCH = "https://esgf-node.llnl.gov/esg-search/search"

DOE_NODES = {
    "Argonne": "eagle.alcf.anl.gov",
    "Oak Ridge": "esgf-node.ornl.gov",
}

DATA_DIR = Path("data/heat_stress")
COMPOUND_DATA_DIR = Path("data/compound_hazard")  # Reuse existing tas data
ARTIFACTS_DIR = Path("Artifacts")

MODEL = "GFDL-ESM4"

PERIODS = {
    "baseline": (1985, 2014),
    "historical": (1995, 2014),
    "mid_century": (2040, 2069),
    "end_century": (2070, 2099),
}

# Wet Bulb Temperature thresholds (°C)
# Note: These are adjusted for monthly mean data, where daily peaks are smoothed
# Daily WBT of 35°C (fatal) corresponds roughly to monthly mean ~28-30°C
WBT_HOT = 24  # Hot conditions (monthly mean)
WBT_DANGEROUS = 26  # Dangerous for sustained outdoor work
WBT_VERY_DANGEROUS = 28  # Very dangerous - chronic heat stress
WBT_EXTREME = 30  # Extreme - life-threatening for prolonged exposure

REGIONS = {
    "South_Asia": {"lat": (5, 35), "lon": (60, 100), "title": "South Asia", "pop": 1.9e9},
    "Middle_East": {"lat": (15, 40), "lon": (25, 65), "title": "Middle East", "pop": 0.4e9},
    "West_Africa": {"lat": (0, 20), "lon": (-20, 20), "title": "West Africa", "pop": 0.4e9},
    "Southeast_Asia": {"lat": (-10, 25), "lon": (95, 145), "title": "Southeast Asia", "pop": 0.7e9},
    "US_Gulf": {"lat": (25, 35), "lon": (-100, -80), "title": "US Gulf Coast", "pop": 0.05e9},
    "Amazon": {"lat": (-20, 10), "lon": (-80, -40), "title": "Amazon Basin", "pop": 0.03e9},
}


def search_esgf_files(source_id, experiment_id, variable_id, data_node, table_id="Amon"):
    """Search ESGF for files matching criteria."""
    params = {
        "project": "CMIP6",
        "source_id": source_id,
        "experiment_id": experiment_id,
        "variable_id": variable_id,
        "table_id": table_id,
        "data_node": data_node,
        "type": "File",
        "format": "application/solr+json",
        "latest": "true",
        "limit": 200,
    }

    try:
        response = requests.get(ESGF_SEARCH, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"    Search error: {e}")
        return [], None

    # Group files by member_id
    by_member = {}
    for doc in data.get("response", {}).get("docs", []):
        member = doc.get("member_id", ["unknown"])[0]
        if member not in by_member:
            by_member[member] = []

        for url_entry in doc.get("url", []):
            parts = url_entry.split("|")
            if len(parts) >= 3 and parts[2] == "HTTPServer":
                by_member[member].append(parts[0])

    if not by_member:
        return [], None

    # Prefer r1i1p1f1, then first available
    chosen = "r1i1p1f1" if "r1i1p1f1" in by_member else sorted(by_member.keys())[0]
    return sorted(set(by_member[chosen])), chosen


def download_file(url, dest_path):
    """Download a single file."""
    if dest_path.exists():
        return dest_path

    try:
        print(f"    Downloading: {dest_path.name}")
        response = requests.get(url, timeout=600, stream=True)
        response.raise_for_status()

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return dest_path
    except Exception as e:
        print(f"    Download error: {e}")
        return None


def download_heat_stress_data():
    """Download humidity data (reuse existing temperature data)."""
    experiments = ["historical", "ssp585", "ssp126"]

    # First, link or copy existing temperature data
    for exp in experiments:
        tas_src = COMPOUND_DATA_DIR / MODEL / exp / "tas"
        tas_dst = DATA_DIR / MODEL / exp / "tas"

        if tas_src.exists() and not tas_dst.exists():
            tas_dst.mkdir(parents=True, exist_ok=True)
            for f in tas_src.glob("*.nc"):
                dst_file = tas_dst / f.name
                if not dst_file.exists():
                    # Create symlink
                    dst_file.symlink_to(f.resolve())
                    print(f"  Linked: {f.name}")

    # Download humidity data
    for exp in experiments:
        output_dir = DATA_DIR / MODEL / exp / "hurs"
        output_dir.mkdir(parents=True, exist_ok=True)

        existing = list(output_dir.glob("*.nc"))
        if existing:
            print(f"  {MODEL}/{exp}/hurs: Using {len(existing)} existing files")
            continue

        print(f"\nSearching {MODEL}/{exp}/hurs...")

        # Try DOE nodes
        for node_name, node_host in DOE_NODES.items():
            urls, member = search_esgf_files(MODEL, exp, "hurs", node_host)
            if urls:
                print(f"  Found {len(urls)} files at {node_name} ({member})")
                for url in urls:
                    filename = url.split("/")[-1]
                    download_file(url, output_dir / filename)
                break
        else:
            print(f"  No files found for hurs/{exp}")


def load_data(files, variable, start_year, end_year):
    """Load and concatenate NetCDF files."""
    datasets = []
    for f in sorted(files):
        try:
            ds = xr.open_dataset(f)
            datasets.append(ds)
        except:
            continue

    if not datasets:
        return None

    ds = xr.concat(datasets, dim="time").sortby("time")

    # Remove duplicate times
    _, idx = np.unique(ds.time.values, return_index=True)
    ds = ds.isel(time=sorted(idx))

    da = ds[variable].sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    return da if da.time.size > 0 else None


def calculate_wet_bulb_temperature(tas, hurs):
    """
    Calculate Wet Bulb Temperature using Stull (2011) approximation.

    This formula is accurate to within 0.3°C for RH > 5% and T between -20°C and 50°C.

    WBT = T * arctan(0.151977 * (RH + 8.313659)^0.5) + arctan(T + RH)
          - arctan(RH - 1.676331) + 0.00391838 * RH^1.5 * arctan(0.023101 * RH)
          - 4.686035

    Parameters:
        tas: Temperature in Kelvin
        hurs: Relative humidity in %

    Returns:
        Wet bulb temperature in °C
    """
    # Convert Kelvin to Celsius
    T = tas - 273.15
    RH = hurs

    # Stull (2011) formula
    WBT = (T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
           + np.arctan(T + RH)
           - np.arctan(RH - 1.676331)
           + 0.00391838 * np.power(RH, 1.5) * np.arctan(0.023101 * RH)
           - 4.686035)

    return WBT


def convert_lon_to_180(da):
    """Convert longitude from 0-360 to -180 to 180."""
    lon = da.lon.values
    lon_180 = np.where(lon > 180, lon - 360, lon)
    da = da.assign_coords(lon=lon_180)
    da = da.sortby("lon")
    return da


def analyze_heat_stress(model=MODEL):
    """Run heat stress analysis."""
    print(f"\n{'='*70}")
    print(f"HEAT STRESS ANALYSIS: {model}")
    print(f"{'='*70}")

    # Load data paths
    hist_tas = sorted((DATA_DIR / model / "historical" / "tas").glob("*.nc"))
    hist_hurs = sorted((DATA_DIR / model / "historical" / "hurs").glob("*.nc"))
    ssp585_tas = sorted((DATA_DIR / model / "ssp585" / "tas").glob("*.nc"))
    ssp585_hurs = sorted((DATA_DIR / model / "ssp585" / "hurs").glob("*.nc"))
    ssp126_tas = sorted((DATA_DIR / model / "ssp126" / "tas").glob("*.nc"))
    ssp126_hurs = sorted((DATA_DIR / model / "ssp126" / "hurs").glob("*.nc"))

    if not all([hist_tas, hist_hurs, ssp585_tas, ssp585_hurs]):
        print("Missing required data files")
        return None

    results = {}

    # Process each period and scenario
    scenarios = {
        "historical": (hist_tas, hist_hurs, PERIODS["historical"]),
        "end_century_ssp585": (ssp585_tas, ssp585_hurs, PERIODS["end_century"]),
    }

    if ssp126_tas and ssp126_hurs:
        scenarios["end_century_ssp126"] = (ssp126_tas, ssp126_hurs, PERIODS["end_century"])

    for scenario_name, (tas_files, hurs_files, (start, end)) in scenarios.items():
        print(f"\nProcessing {scenario_name} ({start}-{end})...")

        tas = load_data(tas_files, "tas", start, end)
        hurs = load_data(hurs_files, "hurs", start, end)

        if tas is None or hurs is None:
            print(f"  Could not load data for {scenario_name}")
            continue

        # Align grids if needed
        if tas.lat.size != hurs.lat.size or tas.lon.size != hurs.lon.size:
            print("  Interpolating humidity to temperature grid...")
            hurs = hurs.interp(lat=tas.lat, lon=tas.lon)

        # Align time if needed
        common_times = np.intersect1d(tas.time.values, hurs.time.values)
        if len(common_times) < tas.time.size:
            print(f"  Aligning time axes ({len(common_times)} common months)")
            tas = tas.sel(time=common_times)
            hurs = hurs.sel(time=common_times)

        print("  Calculating Wet Bulb Temperature...")
        wbt = calculate_wet_bulb_temperature(tas, hurs)

        # Calculate exceedance frequencies
        print("  Computing exceedance frequencies...")

        hot = (wbt > WBT_HOT).mean(dim="time") * 100
        dangerous = (wbt > WBT_DANGEROUS).mean(dim="time") * 100
        very_dangerous = (wbt > WBT_VERY_DANGEROUS).mean(dim="time") * 100
        extreme = (wbt > WBT_EXTREME).mean(dim="time") * 100

        # Global statistics
        weights = np.cos(np.deg2rad(dangerous.lat))
        global_hot = float(hot.weighted(weights).mean().values)
        global_dangerous = float(dangerous.weighted(weights).mean().values)
        global_very_dangerous = float(very_dangerous.weighted(weights).mean().values)
        global_extreme = float(extreme.weighted(weights).mean().values)

        print(f"  Global mean WBT>{WBT_HOT}°C: {global_hot:.2f}% of time")
        print(f"  Global mean WBT>{WBT_DANGEROUS}°C: {global_dangerous:.2f}% of time")
        print(f"  Global mean WBT>{WBT_VERY_DANGEROUS}°C: {global_very_dangerous:.2f}% of time")
        print(f"  Global mean WBT>{WBT_EXTREME}°C: {global_extreme:.2f}% of time")

        # Regional statistics
        regional_stats = {}
        for region_name, bounds in REGIONS.items():
            lat_min, lat_max = bounds["lat"]
            lon_min, lon_max = bounds["lon"]

            # Handle longitude
            if lon_min < 0:
                lon_min_sel = lon_min + 360
            else:
                lon_min_sel = lon_min
            if lon_max < 0:
                lon_max_sel = lon_max + 360
            else:
                lon_max_sel = lon_max

            try:
                if lon_min_sel > lon_max_sel:
                    # Crosses date line
                    region_h = xr.concat([
                        hot.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_sel, 360)),
                        hot.sel(lat=slice(lat_min, lat_max), lon=slice(0, lon_max_sel))
                    ], dim="lon")
                    region_d = xr.concat([
                        dangerous.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_sel, 360)),
                        dangerous.sel(lat=slice(lat_min, lat_max), lon=slice(0, lon_max_sel))
                    ], dim="lon")
                    region_vd = xr.concat([
                        very_dangerous.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_sel, 360)),
                        very_dangerous.sel(lat=slice(lat_min, lat_max), lon=slice(0, lon_max_sel))
                    ], dim="lon")
                    region_e = xr.concat([
                        extreme.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_sel, 360)),
                        extreme.sel(lat=slice(lat_min, lat_max), lon=slice(0, lon_max_sel))
                    ], dim="lon")
                else:
                    region_h = hot.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_sel, lon_max_sel))
                    region_d = dangerous.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_sel, lon_max_sel))
                    region_vd = very_dangerous.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_sel, lon_max_sel))
                    region_e = extreme.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_sel, lon_max_sel))

                weights_r = np.cos(np.deg2rad(region_d.lat))
                regional_stats[region_name] = {
                    "hot": float(region_h.weighted(weights_r).mean().values),
                    "dangerous": float(region_d.weighted(weights_r).mean().values),
                    "very_dangerous": float(region_vd.weighted(weights_r).mean().values),
                    "extreme": float(region_e.weighted(weights_r).mean().values),
                    "population": bounds["pop"]
                }
            except Exception as e:
                print(f"  Error with region {region_name}: {e}")
                continue

        results[scenario_name] = {
            "hot": hot,
            "dangerous": dangerous,
            "very_dangerous": very_dangerous,
            "extreme": extreme,
            "global": {
                "hot": global_hot,
                "dangerous": global_dangerous,
                "very_dangerous": global_very_dangerous,
                "extreme": global_extreme
            },
            "regional": regional_stats
        }

    return results


def plot_heat_stress_results(results):
    """Create heat stress visualizations."""
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*70}")

    if "historical" not in results or "end_century_ssp585" not in results:
        print("Missing required scenarios for plotting")
        return

    # Figure 1: Global maps - use DANGEROUS threshold (26°C) for better land signal
    fig = plt.figure(figsize=(16, 10))

    # Historical - dangerous threshold
    ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.Robinson())
    ax1.set_global()

    hist_d = convert_lon_to_180(results["historical"]["dangerous"])

    im1 = ax1.pcolormesh(
        hist_d.lon, hist_d.lat, hist_d,
        cmap="YlOrRd", vmin=0, vmax=50,
        transform=ccrs.PlateCarree(),
        shading="auto"
    )
    # Add ocean mask ON TOP to hide ocean data
    ax1.add_feature(cfeature.OCEAN, facecolor='white', zorder=2)
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--", zorder=3)
    ax1.set_title(f"Historical (1995-2014)\nWBT > {WBT_DANGEROUS}°C Frequency", fontsize=11)
    plt.colorbar(im1, ax=ax1, orientation="horizontal", pad=0.05, shrink=0.6, label="% of time")

    # Future SSP5-8.5 - dangerous threshold
    ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.Robinson())
    ax2.set_global()

    future_d = convert_lon_to_180(results["end_century_ssp585"]["dangerous"])

    im2 = ax2.pcolormesh(
        future_d.lon, future_d.lat, future_d,
        cmap="YlOrRd", vmin=0, vmax=50,
        transform=ccrs.PlateCarree(),
        shading="auto"
    )
    ax2.add_feature(cfeature.OCEAN, facecolor='white', zorder=2)
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--", zorder=3)
    ax2.set_title(f"End-Century SSP5-8.5 (2070-2099)\nWBT > {WBT_DANGEROUS}°C Frequency", fontsize=11)
    plt.colorbar(im2, ax=ax2, orientation="horizontal", pad=0.05, shrink=0.6, label="% of time")

    # Change map - dangerous threshold
    ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.Robinson())
    ax3.set_global()

    change_d = future_d - hist_d

    im3 = ax3.pcolormesh(
        change_d.lon, change_d.lat, change_d,
        cmap="Reds", vmin=0, vmax=40,
        transform=ccrs.PlateCarree(),
        shading="auto"
    )
    ax3.add_feature(cfeature.OCEAN, facecolor='white', zorder=2)
    ax3.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax3.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--", zorder=3)
    ax3.set_title(f"Change in WBT > {WBT_DANGEROUS}°C Frequency\n(SSP5-8.5 minus Historical)", fontsize=11)
    plt.colorbar(im3, ax=ax3, orientation="horizontal", pad=0.05, shrink=0.6, label="Δ percentage points")

    # Very dangerous threshold map (future) - 28°C
    ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.Robinson())
    ax4.set_global()

    future_vd = convert_lon_to_180(results["end_century_ssp585"]["very_dangerous"])

    im4 = ax4.pcolormesh(
        future_vd.lon, future_vd.lat, future_vd,
        cmap="hot_r", vmin=0, vmax=15,
        transform=ccrs.PlateCarree(),
        shading="auto"
    )
    ax4.add_feature(cfeature.OCEAN, facecolor='white', zorder=2)
    ax4.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax4.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--", zorder=3)
    ax4.set_title(f"End-Century SSP5-8.5\nWBT > {WBT_VERY_DANGEROUS}°C (Very Dangerous)", fontsize=11)
    plt.colorbar(im4, ax=ax4, orientation="horizontal", pad=0.05, shrink=0.6, label="% of time")

    plt.suptitle("Human Heat Stress: Wet Bulb Temperature Exceedance (Land Only)\nGFDL-ESM4 CMIP6", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = ARTIFACTS_DIR / "heat_stress_global.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")

    # Figure 2: Regional bar charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Dangerous threshold by region
    ax = axes[0]
    regions = list(REGIONS.keys())
    x = np.arange(len(regions))
    width = 0.35

    hist_vals = [results["historical"]["regional"].get(r, {}).get("dangerous", 0) for r in regions]
    future_vals = [results["end_century_ssp585"]["regional"].get(r, {}).get("dangerous", 0) for r in regions]

    ax.bar(x - width/2, hist_vals, width, label="Historical", color="#2166ac", alpha=0.8)
    ax.bar(x + width/2, future_vals, width, label="End-Century SSP5-8.5", color="#b2182b", alpha=0.8)

    ax.set_ylabel(f"% of time WBT > {WBT_DANGEROUS}°C")
    ax.set_title(f"Dangerous Heat Stress (WBT > {WBT_DANGEROUS}°C)\nBy Region", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIONS[r]["title"] for r in regions], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Very dangerous threshold
    ax = axes[1]

    hist_vals = [results["historical"]["regional"].get(r, {}).get("very_dangerous", 0) for r in regions]
    future_vals = [results["end_century_ssp585"]["regional"].get(r, {}).get("very_dangerous", 0) for r in regions]

    ax.bar(x - width/2, hist_vals, width, label="Historical", color="#2166ac", alpha=0.8)
    ax.bar(x + width/2, future_vals, width, label="End-Century SSP5-8.5", color="#b2182b", alpha=0.8)

    ax.set_ylabel(f"% of time WBT > {WBT_VERY_DANGEROUS}°C")
    ax.set_title(f"Very Dangerous Heat Stress (WBT > {WBT_VERY_DANGEROUS}°C)\nBy Region", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIONS[r]["title"] for r in regions], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    output_path = ARTIFACTS_DIR / "heat_stress_regional.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")

    # Figure 3: Population exposure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate person-hours of dangerous exposure
    regions_sorted = sorted(regions, key=lambda r: results["end_century_ssp585"]["regional"].get(r, {}).get("very_dangerous", 0), reverse=True)

    hist_exposure = []
    future_exposure = []
    labels = []

    for r in regions_sorted:
        pop = REGIONS[r]["pop"]
        hist_pct = results["historical"]["regional"].get(r, {}).get("very_dangerous", 0)
        future_pct = results["end_century_ssp585"]["regional"].get(r, {}).get("very_dangerous", 0)

        # Convert to billion person-hours per year
        hist_exposure.append(pop * hist_pct / 100 * 8760 / 1e9)
        future_exposure.append(pop * future_pct / 100 * 8760 / 1e9)
        labels.append(REGIONS[r]["title"])

    x = np.arange(len(labels))

    ax.bar(x - width/2, hist_exposure, width, label="Historical", color="#2166ac", alpha=0.8)
    ax.bar(x + width/2, future_exposure, width, label="End-Century SSP5-8.5", color="#b2182b", alpha=0.8)

    ax.set_ylabel("Billion Person-Hours per Year")
    ax.set_title(f"Population Exposure to Very Dangerous Heat Stress\n(WBT > {WBT_VERY_DANGEROUS}°C)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    output_path = ARTIFACTS_DIR / "heat_stress_exposure.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def save_summary(results):
    """Save results summary to JSON."""
    summary = {
        "thresholds": {
            "hot": WBT_HOT,
            "dangerous": WBT_DANGEROUS,
            "very_dangerous": WBT_VERY_DANGEROUS,
            "extreme": WBT_EXTREME
        },
        "global_results": {},
        "regional_results": {}
    }

    for scenario, data in results.items():
        summary["global_results"][scenario] = data["global"]
        summary["regional_results"][scenario] = data["regional"]

    output_path = ARTIFACTS_DIR / "heat_stress_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {output_path}")

    return summary


def main():
    print("=" * 70)
    print("HUMAN HEAT STRESS ANALYSIS")
    print("Wet Bulb Temperature Under Climate Change")
    print("=" * 70)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Download data
    print("\n" + "=" * 70)
    print("DOWNLOADING DATA")
    print("=" * 70)
    download_heat_stress_data()

    # Run analysis
    results = analyze_heat_stress()

    if results:
        # Create visualizations
        plot_heat_stress_results(results)

        # Save summary
        summary = save_summary(results)

        # Print summary
        print("\n" + "=" * 70)
        print("HEAT STRESS SUMMARY")
        print("=" * 70)

        if "historical" in results and "end_century_ssp585" in results:
            hist = results["historical"]["global"]
            future = results["end_century_ssp585"]["global"]

            print(f"\nGlobal WBT > {WBT_HOT}°C (hot):")
            print(f"  Historical: {hist['hot']:.2f}%")
            print(f"  End-Century: {future['hot']:.2f}%")
            print(f"  Change: +{future['hot'] - hist['hot']:.2f} pp")

            print(f"\nGlobal WBT > {WBT_DANGEROUS}°C (dangerous):")
            print(f"  Historical: {hist['dangerous']:.2f}%")
            print(f"  End-Century: {future['dangerous']:.2f}%")
            print(f"  Change: +{future['dangerous'] - hist['dangerous']:.2f} pp")

            print(f"\nGlobal WBT > {WBT_VERY_DANGEROUS}°C (very dangerous):")
            print(f"  Historical: {hist['very_dangerous']:.2f}%")
            print(f"  End-Century: {future['very_dangerous']:.2f}%")
            print(f"  Change: +{future['very_dangerous'] - hist['very_dangerous']:.2f} pp")

            print(f"\nGlobal WBT > {WBT_EXTREME}°C (extreme):")
            print(f"  Historical: {hist['extreme']:.2f}%")
            print(f"  End-Century: {future['extreme']:.2f}%")
            print(f"  Change: +{future['extreme'] - hist['extreme']:.2f} pp")

            print(f"\nRegional Results (Dangerous WBT > {WBT_DANGEROUS}°C):")
            for region in REGIONS:
                if region in results["historical"]["regional"] and region in results["end_century_ssp585"]["regional"]:
                    h = results["historical"]["regional"][region]["dangerous"]
                    f = results["end_century_ssp585"]["regional"][region]["dangerous"]
                    print(f"  {REGIONS[region]['title']}: {h:.1f}% → {f:.1f}% (+{f-h:.1f} pp)")

    print("\n" + "=" * 70)
    print("Heat stress analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
