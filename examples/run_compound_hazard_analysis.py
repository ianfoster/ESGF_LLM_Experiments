#!/usr/bin/env python3
"""
Compound Climate Hazard Analysis: Heat-Drought Concurrence

Research Question:
Does climate change disproportionately increase compound heat-drought events,
with amplification being non-linear and spatially heterogeneous?

Methodology:
1. Download temperature (tas) and precipitation (pr) for historical + SSP5-8.5
2. Define baseline period (1985-2014) percentile thresholds
3. Identify compound events: temp > 90th percentile AND precip < 10th percentile
4. Compare frequency: historical vs mid-century vs end-century
5. Analyze spatial patterns of compound risk amplification
6. Test non-linearity: compound risk vs product of individual risks
"""

import warnings
warnings.filterwarnings("ignore")

import requests
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import json


# Configuration
ESGF_SEARCH = "https://esgf-node.llnl.gov/esg-search/search"

DOE_NODES = {
    "Argonne": "eagle.alcf.anl.gov",
    "Oak Ridge": "esgf-node.ornl.gov",
}

# Use models with good data availability at DOE
MODELS = ["GFDL-ESM4", "MIROC6", "NorESM2-LM"]
VARIABLES = ["tas", "pr"]
TABLE = "Amon"

# Time periods for analysis
PERIODS = {
    "baseline": (1985, 2014),      # Historical baseline for percentiles
    "historical": (1995, 2014),    # Recent historical for comparison
    "mid_century": (2040, 2069),   # Mid-century projection
    "end_century": (2070, 2099),   # End-century projection
}

# Percentile thresholds for compound events
HEAT_PERCENTILE = 90   # Hot: above 90th percentile
DROUGHT_PERCENTILE = 10  # Dry: below 10th percentile

DATA_DIR = Path("data/compound_hazard")
ARTIFACTS_DIR = Path("Artifacts")


def search_esgf_files(
    source_id: str,
    experiment_id: str,
    variable_id: str,
    data_node: str,
    table_id: str = "Amon"
) -> tuple[list[str], Optional[str]]:
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


def download_file(url: str, dest_dir: Path) -> Optional[Path]:
    """Download a single file with progress indication."""
    filename = url.split("/")[-1]
    dest_path = dest_dir / filename

    if dest_path.exists():
        return dest_path

    try:
        response = requests.get(url, timeout=600, stream=True)
        response.raise_for_status()

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return dest_path
    except Exception as e:
        print(f"    Download error for {filename}: {e}")
        return None


def download_dataset(
    source_id: str,
    experiment_id: str,
    variable_id: str,
) -> Optional[list[Path]]:
    """Download all files for a model/experiment/variable combination."""

    dest_dir = DATA_DIR / source_id / experiment_id / variable_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_files = list(dest_dir.glob("*.nc"))
    if existing_files:
        print(f"  Using cached: {len(existing_files)} files")
        return sorted(existing_files)

    # Try DOE nodes
    for node_name, node_host in DOE_NODES.items():
        urls, member = search_esgf_files(
            source_id, experiment_id, variable_id, node_host
        )

        if not urls:
            continue

        print(f"  {node_name}: {len(urls)} files ({member})")

        files = []
        for url in urls:
            f = download_file(url, dest_dir)
            if f:
                files.append(f)

        if files:
            total_mb = sum(f.stat().st_size for f in files) / 1e6
            print(f"    Downloaded: {total_mb:.1f} MB")
            return sorted(files)

    return None


def load_and_process_data(
    files: list[Path],
    variable: str,
    start_year: int,
    end_year: int
) -> Optional[xr.DataArray]:
    """Load NetCDF files and extract time period."""
    try:
        # Load files individually and concatenate (more robust for multi-file datasets)
        datasets = []
        for f in sorted(files):
            try:
                d = xr.open_dataset(f)
                datasets.append(d)
            except Exception:
                continue

        if not datasets:
            return None

        ds = xr.concat(datasets, dim="time")

        # Sort by time to ensure monotonic index
        ds = ds.sortby("time")

        # Remove duplicate time values (keep first occurrence)
        _, unique_indices = np.unique(ds.time.values, return_index=True)
        ds = ds.isel(time=sorted(unique_indices))

        da = ds[variable]

        # Select time period
        da = da.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

        # Ensure we have data
        if da.time.size == 0:
            return None

        return da
    except Exception as e:
        print(f"    Load error: {e}")
        return None


def compute_percentile_thresholds(
    tas_baseline: xr.DataArray,
    pr_baseline: xr.DataArray,
    heat_pct: int = 90,
    drought_pct: int = 10
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute percentile thresholds for each grid cell and calendar month.
    This accounts for seasonal cycles in the baseline.
    """
    print("  Computing percentile thresholds...")

    # Group by month to get seasonal thresholds
    tas_threshold = tas_baseline.groupby("time.month").quantile(
        heat_pct / 100.0, dim="time"
    )
    pr_threshold = pr_baseline.groupby("time.month").quantile(
        drought_pct / 100.0, dim="time"
    )

    return tas_threshold, pr_threshold


def identify_compound_events(
    tas: xr.DataArray,
    pr: xr.DataArray,
    tas_threshold: xr.DataArray,
    pr_threshold: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Identify months with compound heat-drought events.

    Returns:
        compound: Boolean array of compound events (hot AND dry)
        heat_only: Boolean array of heat-only events
        drought_only: Boolean array of drought-only events
    """
    # Align thresholds with data by month
    tas_thresh_aligned = tas_threshold.sel(month=tas.time.dt.month)
    pr_thresh_aligned = pr_threshold.sel(month=pr.time.dt.month)

    # Drop the month coordinate to enable comparison
    tas_thresh_aligned = tas_thresh_aligned.drop_vars("month")
    pr_thresh_aligned = pr_thresh_aligned.drop_vars("month")

    # Identify events
    is_hot = tas > tas_thresh_aligned
    is_dry = pr < pr_thresh_aligned

    compound = is_hot & is_dry
    heat_only = is_hot & ~is_dry
    drought_only = ~is_hot & is_dry

    return compound, heat_only, drought_only


def compute_event_frequency(events: xr.DataArray) -> xr.DataArray:
    """Compute frequency (fraction of months) with events."""
    return events.mean(dim="time")


def analyze_model(source_id: str) -> Optional[dict]:
    """Run complete compound hazard analysis for one model."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {source_id}")
    print(f"{'='*60}")

    results = {"model": source_id}

    # Download historical data (for baseline)
    print("\nDownloading historical data...")
    hist_tas_files = download_dataset(source_id, "historical", "tas")
    hist_pr_files = download_dataset(source_id, "historical", "pr")

    if not hist_tas_files or not hist_pr_files:
        print(f"  Failed to get historical data for {source_id}")
        return None

    # Download SSP5-8.5 data
    print("\nDownloading SSP5-8.5 data...")
    ssp_tas_files = download_dataset(source_id, "ssp585", "tas")
    ssp_pr_files = download_dataset(source_id, "ssp585", "pr")

    if not ssp_tas_files or not ssp_pr_files:
        print(f"  Failed to get SSP5-8.5 data for {source_id}")
        return None

    # Load baseline period for thresholds
    print("\nProcessing baseline period (1985-2014)...")
    tas_baseline = load_and_process_data(
        hist_tas_files, "tas", *PERIODS["baseline"]
    )
    pr_baseline = load_and_process_data(
        hist_pr_files, "pr", *PERIODS["baseline"]
    )

    if tas_baseline is None or pr_baseline is None:
        print("  Failed to load baseline data")
        return None

    # Compute thresholds
    tas_thresh, pr_thresh = compute_percentile_thresholds(
        tas_baseline, pr_baseline, HEAT_PERCENTILE, DROUGHT_PERCENTILE
    )

    # Analyze each period
    period_results = {}

    for period_name, (start_year, end_year) in PERIODS.items():
        if period_name == "baseline":
            continue  # Skip baseline (used only for thresholds)

        print(f"\nAnalyzing {period_name} ({start_year}-{end_year})...")

        # Choose files based on period
        if start_year < 2015:
            tas_files = hist_tas_files
            pr_files = hist_pr_files
        else:
            tas_files = ssp_tas_files
            pr_files = ssp_pr_files

        # Load data
        tas_data = load_and_process_data(tas_files, "tas", start_year, end_year)
        pr_data = load_and_process_data(pr_files, "pr", start_year, end_year)

        if tas_data is None or pr_data is None:
            print(f"  Could not load data for {period_name}")
            continue

        # Align grids if necessary
        if tas_data.lat.size != pr_data.lat.size:
            print("  Regridding precipitation to match temperature...")
            pr_data = pr_data.interp(lat=tas_data.lat, lon=tas_data.lon)

        # Identify events
        compound, heat_only, drought_only = identify_compound_events(
            tas_data, pr_data, tas_thresh, pr_thresh
        )

        # Compute frequencies
        compound_freq = compute_event_frequency(compound).compute()
        heat_freq = compute_event_frequency(heat_only | compound).compute()  # Total heat events
        drought_freq = compute_event_frequency(drought_only | compound).compute()  # Total drought events

        # Store results
        period_results[period_name] = {
            "compound_freq": compound_freq,
            "heat_freq": heat_freq,
            "drought_freq": drought_freq,
            "years": (start_year, end_year),
        }

        # Global statistics (area-weighted)
        weights = np.cos(np.deg2rad(compound_freq.lat))
        global_compound = float(compound_freq.weighted(weights).mean().values)
        global_heat = float(heat_freq.weighted(weights).mean().values)
        global_drought = float(drought_freq.weighted(weights).mean().values)

        print(f"  Global compound event frequency: {global_compound*100:.2f}%")
        print(f"  Global heat event frequency: {global_heat*100:.2f}%")
        print(f"  Global drought event frequency: {global_drought*100:.2f}%")

        # Expected if independent
        expected_compound = global_heat * global_drought
        print(f"  Expected if independent: {expected_compound*100:.3f}%")
        print(f"  Amplification factor: {global_compound/expected_compound:.2f}x")

    results["periods"] = period_results
    return results


def plot_compound_hazard_results(all_results: list[dict]) -> Path:
    """Create comprehensive visualization of compound hazard analysis."""

    # Create 4-panel figure
    fig = plt.figure(figsize=(16, 14))

    # Get first model's data for maps (use GFDL-ESM4 if available)
    map_model = None
    for r in all_results:
        if r["model"] == "GFDL-ESM4":
            map_model = r
            break
    if map_model is None:
        map_model = all_results[0]

    # Panel 1: Compound event frequency change (map)
    ax1 = fig.add_subplot(2, 2, 1)

    hist_data = map_model["periods"]["historical"]["compound_freq"]
    end_data = map_model["periods"]["end_century"]["compound_freq"]
    change = end_data - hist_data

    # Convert to percentage points
    change_pct = change * 100

    im1 = ax1.pcolormesh(
        change_pct.lon, change_pct.lat, change_pct,
        cmap="YlOrRd", vmin=0, vmax=15,
        shading="auto"
    )
    ax1.set_title(
        f"Change in Compound Heat-Drought Frequency\n"
        f"{map_model['model']} SSP5-8.5 (2070-2099 minus 1995-2014)",
        fontsize=11
    )
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.coastlines() if hasattr(ax1, 'coastlines') else None
    plt.colorbar(im1, ax=ax1, label="Change (percentage points)", shrink=0.8)

    # Panel 2: Amplification factor map (compound vs expected)
    ax2 = fig.add_subplot(2, 2, 2)

    # Calculate amplification for end century
    heat_freq = map_model["periods"]["end_century"]["heat_freq"]
    drought_freq = map_model["periods"]["end_century"]["drought_freq"]
    expected = heat_freq * drought_freq

    # Avoid division by zero
    expected_safe = xr.where(expected > 0.001, expected, 0.001)
    amplification = end_data / expected_safe
    amplification = xr.where(expected > 0.001, amplification, np.nan)

    im2 = ax2.pcolormesh(
        amplification.lon, amplification.lat, amplification,
        cmap="RdYlBu_r", vmin=0.5, vmax=3,
        shading="auto"
    )
    ax2.set_title(
        f"Compound Event Amplification Factor\n"
        f"(Actual / Expected if Independent) - 2070-2099",
        fontsize=11
    )
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    plt.colorbar(im2, ax=ax2, label="Amplification factor", shrink=0.8)

    # Panel 3: Time evolution (all models)
    ax3 = fig.add_subplot(2, 2, 3)

    colors = {"GFDL-ESM4": "#1b9e77", "MIROC6": "#d95f02", "NorESM2-LM": "#7570b3"}
    period_order = ["historical", "mid_century", "end_century"]
    period_labels = ["Historical\n(1995-2014)", "Mid-Century\n(2040-2069)", "End-Century\n(2070-2099)"]
    x_positions = [0, 1, 2]

    for result in all_results:
        model = result["model"]
        compound_values = []

        for period in period_order:
            if period in result["periods"]:
                freq = result["periods"][period]["compound_freq"]
                weights = np.cos(np.deg2rad(freq.lat))
                global_val = float(freq.weighted(weights).mean().values) * 100
                compound_values.append(global_val)
            else:
                compound_values.append(np.nan)

        ax3.plot(
            x_positions, compound_values,
            marker='o', markersize=10, linewidth=2.5,
            label=model, color=colors.get(model, "gray")
        )

    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(period_labels)
    ax3.set_ylabel("Global Compound Event Frequency (%)", fontsize=11)
    ax3.set_title("Evolution of Compound Heat-Drought Events\nMulti-Model Comparison", fontsize=11)
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    # Panel 4: Non-linearity test
    ax4 = fig.add_subplot(2, 2, 4)

    # Filter models with complete data for panel 4
    complete_results = [r for r in all_results
                       if "historical" in r["periods"] and "end_century" in r["periods"]]

    # Compare actual compound increase vs expected from individual hazards
    bar_width = 0.25
    x = np.arange(len(complete_results))

    actual_increase = []
    expected_increase = []
    model_names = []

    for result in complete_results:
        model_names.append(result["model"])

        hist = result["periods"]["historical"]
        end = result["periods"]["end_century"]

        # Global means
        weights_hist = np.cos(np.deg2rad(hist["compound_freq"].lat))
        weights_end = np.cos(np.deg2rad(end["compound_freq"].lat))

        hist_compound = float(hist["compound_freq"].weighted(weights_hist).mean().values)
        end_compound = float(end["compound_freq"].weighted(weights_end).mean().values)

        hist_heat = float(hist["heat_freq"].weighted(weights_hist).mean().values)
        end_heat = float(end["heat_freq"].weighted(weights_end).mean().values)

        hist_drought = float(hist["drought_freq"].weighted(weights_hist).mean().values)
        end_drought = float(end["drought_freq"].weighted(weights_end).mean().values)

        # Actual increase in compound events
        actual_inc = (end_compound - hist_compound) * 100
        actual_increase.append(actual_inc)

        # Expected increase if risks multiplied independently
        # Change in compound = change_heat * drought + heat * change_drought + change_heat * change_drought
        # Simplified: just compute expected compound from individual frequencies
        expected_end = end_heat * end_drought
        expected_hist = hist_heat * hist_drought
        expected_inc = (expected_end - expected_hist) * 100
        expected_increase.append(expected_inc)

    bars1 = ax4.bar(x - bar_width/2, actual_increase, bar_width,
                    label='Actual Compound Increase', color='#d7191c', alpha=0.8)
    bars2 = ax4.bar(x + bar_width/2, expected_increase, bar_width,
                    label='Expected if Independent', color='#2c7bb6', alpha=0.8)

    ax4.set_ylabel('Increase in Event Frequency\n(percentage points)', fontsize=11)
    ax4.set_title('Non-Linearity Test: Actual vs Expected\nCompound Event Increase (Historical → 2070-2099)', fontsize=11)
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add ratio annotations
    for i, (act, exp) in enumerate(zip(actual_increase, expected_increase)):
        if exp > 0:
            ratio = act / exp
            ax4.annotate(
                f'{ratio:.1f}x',
                xy=(i, max(act, exp) + 0.3),
                ha='center', fontsize=10, fontweight='bold'
            )

    plt.tight_layout()

    output_path = ARTIFACTS_DIR / "compound_hazard_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def plot_regional_compound_risk(all_results: list[dict]) -> Path:
    """Create regional analysis of compound risk hotspots."""

    # Use first model with complete data for detailed regional analysis
    complete_results = [r for r in all_results
                       if "historical" in r["periods"] and "end_century" in r["periods"]]
    if not complete_results:
        print("  No complete model data for regional analysis")
        return None

    result = complete_results[0]
    model = result["model"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Define regions of interest
    regions = {
        "Mediterranean": {"lat": (30, 45), "lon": (-10, 40)},
        "US Southwest": {"lat": (25, 40), "lon": (-125, -100)},
        "Amazon": {"lat": (-20, 5), "lon": (-75, -45)},
        "Australia": {"lat": (-40, -15), "lon": (110, 155)},
        "Southern Africa": {"lat": (-35, -15), "lon": (15, 40)},
        "South Asia": {"lat": (5, 35), "lon": (65, 100)},
    }

    hist_data = result["periods"]["historical"]["compound_freq"]
    end_data = result["periods"]["end_century"]["compound_freq"]
    change = (end_data - hist_data) * 100

    for ax, (region_name, bounds) in zip(axes.flat, regions.items()):
        lat_slice = slice(bounds["lat"][0], bounds["lat"][1])
        lon_min, lon_max = bounds["lon"]

        # Handle longitude wrapping
        if lon_min < 0:
            lon_min = lon_min % 360
        if lon_max < 0:
            lon_max = lon_max % 360

        # Try to slice, handling different longitude conventions
        try:
            if lon_min > lon_max:  # Crosses 0/360
                region_change1 = change.sel(lat=lat_slice, lon=slice(lon_min, 360))
                region_change2 = change.sel(lat=lat_slice, lon=slice(0, lon_max))
                region_change = xr.concat([region_change1, region_change2], dim="lon")
            else:
                region_change = change.sel(lat=lat_slice, lon=slice(lon_min, lon_max))
        except Exception:
            # Try with negative longitudes
            region_change = change.sel(
                lat=lat_slice,
                lon=slice(bounds["lon"][0], bounds["lon"][1])
            )

        im = ax.pcolormesh(
            region_change.lon, region_change.lat, region_change,
            cmap="YlOrRd", vmin=0, vmax=20,
            shading="auto"
        )

        # Calculate regional statistics
        weights = np.cos(np.deg2rad(region_change.lat))
        mean_change = float(region_change.weighted(weights).mean().values)
        max_change = float(region_change.max().values)

        ax.set_title(f"{region_name}\nMean: +{mean_change:.1f}pp, Max: +{max_change:.1f}pp", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.colorbar(im, ax=ax, label="Δ Frequency (pp)", shrink=0.8)

    fig.suptitle(
        f"Regional Compound Heat-Drought Risk Hotspots\n{model} SSP5-8.5: Change from Historical to 2070-2099",
        fontsize=14, y=1.02
    )

    plt.tight_layout()

    output_path = ARTIFACTS_DIR / "compound_hazard_regional.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def generate_summary_statistics(all_results: list[dict]) -> dict:
    """Generate comprehensive summary statistics."""

    # Filter for models with complete data
    complete_results = [r for r in all_results
                       if "historical" in r["periods"] and "end_century" in r["periods"]]

    summary = {
        "hypothesis": "Climate change disproportionately increases compound heat-drought events",
        "models_analyzed": [r["model"] for r in all_results],
        "models_with_complete_data": [r["model"] for r in complete_results],
        "scenarios": ["historical", "ssp585"],
        "thresholds": {
            "heat": f">{HEAT_PERCENTILE}th percentile",
            "drought": f"<{DROUGHT_PERCENTILE}th percentile",
        },
        "findings": {}
    }

    if not complete_results:
        summary["findings"] = {"error": "No models with complete data"}
        summary["hypothesis_verdict"] = "INSUFFICIENT DATA"
        return summary

    # Calculate cross-model statistics
    global_increases = []
    amplification_factors = []
    nonlinearity_ratios = []

    for result in complete_results:
        hist = result["periods"]["historical"]
        end = result["periods"]["end_century"]

        weights = np.cos(np.deg2rad(hist["compound_freq"].lat))

        hist_compound = float(hist["compound_freq"].weighted(weights).mean().values)
        end_compound = float(end["compound_freq"].weighted(weights).mean().values)

        hist_heat = float(hist["heat_freq"].weighted(weights).mean().values)
        end_heat = float(end["heat_freq"].weighted(weights).mean().values)

        hist_drought = float(hist["drought_freq"].weighted(weights).mean().values)
        end_drought = float(end["drought_freq"].weighted(weights).mean().values)

        # Percent increase
        pct_increase = ((end_compound - hist_compound) / hist_compound) * 100
        global_increases.append(pct_increase)

        # Amplification factor
        expected = end_heat * end_drought
        if expected > 0:
            amp = end_compound / expected
            amplification_factors.append(amp)

        # Non-linearity ratio
        actual_increase = end_compound - hist_compound
        expected_increase = (end_heat * end_drought) - (hist_heat * hist_drought)
        if expected_increase > 0:
            nonlinearity_ratios.append(actual_increase / expected_increase)

    summary["findings"] = {
        "global_compound_increase": {
            "mean_percent": np.mean(global_increases),
            "range_percent": (min(global_increases), max(global_increases)),
            "interpretation": "Percent increase in compound events from historical to end-century"
        },
        "amplification_factor": {
            "mean": np.mean(amplification_factors),
            "range": (min(amplification_factors), max(amplification_factors)),
            "interpretation": "Ratio of actual to expected compound frequency (>1 means positive correlation)"
        },
        "nonlinearity_ratio": {
            "mean": np.mean(nonlinearity_ratios),
            "range": (min(nonlinearity_ratios), max(nonlinearity_ratios)),
            "interpretation": "Ratio of actual to expected increase (>1 supports nonlinearity hypothesis)"
        }
    }

    # Hypothesis verdict
    if np.mean(nonlinearity_ratios) > 1.2:
        verdict = "SUPPORTED - Compound risk increases faster than expected from individual hazards"
    elif np.mean(nonlinearity_ratios) > 1.0:
        verdict = "WEAKLY SUPPORTED - Some evidence of nonlinear amplification"
    else:
        verdict = "NOT SUPPORTED - Compound risk increases proportionally to individual hazards"

    summary["hypothesis_verdict"] = verdict

    return summary


def main():
    print("="*70)
    print("COMPOUND CLIMATE HAZARD ANALYSIS")
    print("Heat-Drought Concurrence Under Climate Change")
    print("="*70)
    print(f"\nResearch Question:")
    print("Does climate change disproportionately increase compound heat-drought")
    print("events, with amplification being non-linear and spatially heterogeneous?")
    print(f"\nConfiguration:")
    print(f"  Models: {', '.join(MODELS)}")
    print(f"  Heat threshold: >{HEAT_PERCENTILE}th percentile of baseline")
    print(f"  Drought threshold: <{DROUGHT_PERCENTILE}th percentile of baseline")
    print(f"  Baseline period: {PERIODS['baseline']}")
    print(f"  Comparison periods: historical, mid-century, end-century")

    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Analyze each model
    all_results = []
    for model in MODELS:
        result = analyze_model(model)
        if result:
            all_results.append(result)

    if not all_results:
        print("\nERROR: No models analyzed successfully")
        return

    # Generate visualizations
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)

    fig1_path = plot_compound_hazard_results(all_results)
    print(f"\nSaved: {fig1_path}")

    fig2_path = plot_regional_compound_risk(all_results)
    print(f"Saved: {fig2_path}")

    # Generate summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)

    summary = generate_summary_statistics(all_results)

    print(f"\nModels analyzed: {', '.join(summary['models_analyzed'])}")
    print(f"\nKey Findings:")

    findings = summary["findings"]
    print(f"\n  1. Global Compound Event Increase:")
    print(f"     Mean: {findings['global_compound_increase']['mean_percent']:.1f}%")
    print(f"     Range: {findings['global_compound_increase']['range_percent'][0]:.1f}% - {findings['global_compound_increase']['range_percent'][1]:.1f}%")

    print(f"\n  2. Amplification Factor (actual/expected):")
    print(f"     Mean: {findings['amplification_factor']['mean']:.2f}x")
    print(f"     (>1 indicates positive correlation between heat and drought)")

    print(f"\n  3. Non-linearity Ratio:")
    print(f"     Mean: {findings['nonlinearity_ratio']['mean']:.2f}x")
    print(f"     (>1 supports hypothesis that compound risk grows faster than individual)")

    print(f"\n{'='*70}")
    print(f"HYPOTHESIS VERDICT: {summary['hypothesis_verdict']}")
    print(f"{'='*70}")

    # Save summary to JSON
    summary_path = ARTIFACTS_DIR / "compound_hazard_summary.json"

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(i) for i in obj]
        return obj

    with open(summary_path, 'w') as f:
        json.dump(convert_for_json(summary), f, indent=2)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
