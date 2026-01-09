#!/usr/bin/env python3
"""
Mitigation Benefit Analysis: SSP1-2.6 vs SSP5-8.5

Quantify how much compound heat-drought risk can be avoided through
climate mitigation by comparing low and high emissions scenarios.

Key Question: How much compound risk is "locked in" vs "avoidable"?
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

# Models to analyze
MODELS = ["GFDL-ESM4", "MIROC6"]  # Use 2 models for speed (NorESM2-LM has file issues)

# Scenarios to compare
SCENARIOS = {
    "ssp126": "SSP1-2.6 (Low Emissions)",
    "ssp585": "SSP5-8.5 (High Emissions)",
}

PERIODS = {
    "baseline": (1985, 2014),
    "historical": (1995, 2014),
    "mid_century": (2040, 2069),
    "end_century": (2070, 2099),
}

HEAT_PERCENTILE = 90
DROUGHT_PERCENTILE = 10

DATA_DIR = Path("data/compound_hazard")
ARTIFACTS_DIR = Path("Artifacts")


def search_esgf_files(
    source_id: str,
    experiment_id: str,
    variable_id: str,
    data_node: str,
) -> tuple[list[str], Optional[str]]:
    """Search ESGF for files."""
    params = {
        "project": "CMIP6",
        "source_id": source_id,
        "experiment_id": experiment_id,
        "variable_id": variable_id,
        "table_id": "Amon",
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
        return [], None

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

    chosen = "r1i1p1f1" if "r1i1p1f1" in by_member else sorted(by_member.keys())[0]
    return sorted(set(by_member[chosen])), chosen


def download_file(url: str, dest_dir: Path) -> Optional[Path]:
    """Download a single file."""
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
        print(f"    Download error: {e}")
        return None


def download_dataset(source_id: str, experiment_id: str, variable_id: str) -> Optional[list[Path]]:
    """Download dataset files."""
    dest_dir = DATA_DIR / source_id / experiment_id / variable_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    existing = list(dest_dir.glob("*.nc"))
    if existing:
        print(f"    Using cached: {len(existing)} files")
        return sorted(existing)

    for node_name, node_host in DOE_NODES.items():
        urls, member = search_esgf_files(source_id, experiment_id, variable_id, node_host)
        if not urls:
            continue

        print(f"    {node_name}: {len(urls)} files ({member})")
        files = [download_file(url, dest_dir) for url in urls]
        files = [f for f in files if f is not None]

        if files:
            total_mb = sum(f.stat().st_size for f in files) / 1e6
            print(f"    Downloaded: {total_mb:.1f} MB")
            return sorted(files)

    return None


def load_data(files: list[Path], variable: str, start_year: int, end_year: int) -> Optional[xr.DataArray]:
    """Load and process NetCDF files."""
    try:
        datasets = []
        for f in sorted(files):
            try:
                datasets.append(xr.open_dataset(f))
            except:
                continue

        if not datasets:
            return None

        ds = xr.concat(datasets, dim="time").sortby("time")

        # Remove duplicates
        _, idx = np.unique(ds.time.values, return_index=True)
        ds = ds.isel(time=sorted(idx))

        da = ds[variable].sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
        return da if da.time.size > 0 else None
    except Exception as e:
        print(f"    Load error: {e}")
        return None


def compute_thresholds(tas: xr.DataArray, pr: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute percentile thresholds by month."""
    tas_thresh = tas.groupby("time.month").quantile(HEAT_PERCENTILE / 100.0, dim="time")
    pr_thresh = pr.groupby("time.month").quantile(DROUGHT_PERCENTILE / 100.0, dim="time")
    return tas_thresh, pr_thresh


def identify_compound(tas: xr.DataArray, pr: xr.DataArray,
                      tas_thresh: xr.DataArray, pr_thresh: xr.DataArray) -> xr.DataArray:
    """Identify compound events."""
    tas_th = tas_thresh.sel(month=tas.time.dt.month).drop_vars("month")
    pr_th = pr_thresh.sel(month=pr.time.dt.month).drop_vars("month")
    return (tas > tas_th) & (pr < pr_th)


def analyze_scenario(source_id: str, scenario: str, tas_thresh: xr.DataArray,
                     pr_thresh: xr.DataArray) -> Optional[dict]:
    """Analyze compound hazards for one scenario."""

    # Get scenario files
    tas_files = sorted((DATA_DIR / source_id / scenario / "tas").glob("*.nc"))
    pr_files = sorted((DATA_DIR / source_id / scenario / "pr").glob("*.nc"))

    if not tas_files or not pr_files:
        # Need to download
        print(f"  Downloading {scenario} data...")
        tas_files = download_dataset(source_id, scenario, "tas")
        pr_files = download_dataset(source_id, scenario, "pr")

        if not tas_files or not pr_files:
            return None

    results = {}

    for period_name in ["mid_century", "end_century"]:
        start, end = PERIODS[period_name]

        tas = load_data(tas_files, "tas", start, end)
        pr = load_data(pr_files, "pr", start, end)

        if tas is None or pr is None:
            continue

        if tas.lat.size != pr.lat.size:
            pr = pr.interp(lat=tas.lat, lon=tas.lon)

        compound = identify_compound(tas, pr, tas_thresh, pr_thresh)
        freq = compound.mean(dim="time").compute()

        weights = np.cos(np.deg2rad(freq.lat))
        global_freq = float(freq.weighted(weights).mean().values) * 100

        results[period_name] = {
            "global_frequency": global_freq,
            "spatial_data": freq,
        }

    return results


def analyze_model(source_id: str) -> Optional[dict]:
    """Run full mitigation comparison for one model."""
    print(f"\n{'='*60}")
    print(f"Mitigation Analysis: {source_id}")
    print(f"{'='*60}")

    # Load historical data for baseline thresholds
    hist_tas = sorted((DATA_DIR / source_id / "historical" / "tas").glob("*.nc"))
    hist_pr = sorted((DATA_DIR / source_id / "historical" / "pr").glob("*.nc"))

    if not hist_tas or not hist_pr:
        print("  Missing historical data - run compound hazard analysis first")
        return None

    print("  Loading baseline thresholds...")
    tas_base = load_data(hist_tas, "tas", *PERIODS["baseline"])
    pr_base = load_data(hist_pr, "pr", *PERIODS["baseline"])

    if tas_base is None or pr_base is None:
        return None

    tas_thresh, pr_thresh = compute_thresholds(tas_base, pr_base)

    # Get historical compound frequency
    print("  Analyzing historical period...")
    tas_hist = load_data(hist_tas, "tas", *PERIODS["historical"])
    pr_hist = load_data(hist_pr, "pr", *PERIODS["historical"])

    if tas_hist is not None and pr_hist is not None:
        compound_hist = identify_compound(tas_hist, pr_hist, tas_thresh, pr_thresh)
        freq_hist = compound_hist.mean(dim="time").compute()
        weights = np.cos(np.deg2rad(freq_hist.lat))
        hist_global = float(freq_hist.weighted(weights).mean().values) * 100
    else:
        hist_global = None
        freq_hist = None

    results = {
        "model": source_id,
        "historical": {"global_frequency": hist_global, "spatial_data": freq_hist},
        "scenarios": {}
    }

    # Analyze each scenario
    for scenario in SCENARIOS.keys():
        print(f"\n  Analyzing {SCENARIOS[scenario]}...")
        scenario_results = analyze_scenario(source_id, scenario, tas_thresh, pr_thresh)
        if scenario_results:
            results["scenarios"][scenario] = scenario_results

            for period, data in scenario_results.items():
                print(f"    {period}: {data['global_frequency']:.2f}%")

    return results


def calculate_mitigation_benefit(all_results: list[dict]) -> dict:
    """Calculate avoided compound risk from mitigation."""

    benefits = {
        "models": [],
        "mid_century": {"ssp126": [], "ssp585": [], "avoided": [], "avoided_pct": []},
        "end_century": {"ssp126": [], "ssp585": [], "avoided": [], "avoided_pct": []},
    }

    for result in all_results:
        benefits["models"].append(result["model"])

        for period in ["mid_century", "end_century"]:
            ssp126 = result["scenarios"].get("ssp126", {}).get(period, {}).get("global_frequency")
            ssp585 = result["scenarios"].get("ssp585", {}).get(period, {}).get("global_frequency")

            if ssp126 is not None and ssp585 is not None:
                benefits[period]["ssp126"].append(ssp126)
                benefits[period]["ssp585"].append(ssp585)

                avoided = ssp585 - ssp126
                avoided_pct = (avoided / ssp585) * 100 if ssp585 > 0 else 0

                benefits[period]["avoided"].append(avoided)
                benefits[period]["avoided_pct"].append(avoided_pct)

    return benefits


def plot_mitigation_comparison(all_results: list[dict], benefits: dict) -> Path:
    """Create comprehensive mitigation comparison visualization."""

    fig = plt.figure(figsize=(16, 12))

    # Panel 1: Scenario comparison over time
    ax1 = fig.add_subplot(2, 2, 1)

    colors = {"ssp126": "#2166ac", "ssp585": "#b2182b"}
    periods = ["historical", "mid_century", "end_century"]
    period_labels = ["Historical\n(1995-2014)", "Mid-Century\n(2040-2069)", "End-Century\n(2070-2099)"]

    # Multi-model mean
    for scenario, color in colors.items():
        values = []
        for period in periods:
            if period == "historical":
                vals = [r["historical"]["global_frequency"] for r in all_results
                       if r["historical"]["global_frequency"] is not None]
            else:
                vals = [r["scenarios"].get(scenario, {}).get(period, {}).get("global_frequency")
                       for r in all_results]
                vals = [v for v in vals if v is not None]

            values.append(np.mean(vals) if vals else np.nan)

        ax1.plot(range(3), values, marker='o', markersize=12, linewidth=3,
                color=color, label=SCENARIOS.get(scenario, scenario))

    # Shade avoided region
    ssp126_vals = [np.mean([r["historical"]["global_frequency"] for r in all_results if r["historical"]["global_frequency"]])]
    ssp585_vals = [ssp126_vals[0]]

    for period in ["mid_century", "end_century"]:
        ssp126_vals.append(np.mean(benefits[period]["ssp126"]) if benefits[period]["ssp126"] else np.nan)
        ssp585_vals.append(np.mean(benefits[period]["ssp585"]) if benefits[period]["ssp585"] else np.nan)

    ax1.fill_between(range(3), ssp126_vals, ssp585_vals, alpha=0.3, color='green',
                     label='Avoided Risk')

    ax1.set_xticks(range(3))
    ax1.set_xticklabels(period_labels)
    ax1.set_ylabel('Compound Event Frequency (%)', fontsize=11)
    ax1.set_title('Compound Heat-Drought Risk: Mitigation Benefit\n(Multi-Model Mean)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Panel 2: Avoided risk bar chart
    ax2 = fig.add_subplot(2, 2, 2)

    x = np.arange(2)
    width = 0.35

    mid_avoided = np.mean(benefits["mid_century"]["avoided"]) if benefits["mid_century"]["avoided"] else 0
    end_avoided = np.mean(benefits["end_century"]["avoided"]) if benefits["end_century"]["avoided"] else 0

    mid_pct = np.mean(benefits["mid_century"]["avoided_pct"]) if benefits["mid_century"]["avoided_pct"] else 0
    end_pct = np.mean(benefits["end_century"]["avoided_pct"]) if benefits["end_century"]["avoided_pct"] else 0

    bars = ax2.bar(x, [mid_avoided, end_avoided], width, color='#2ca02c', alpha=0.8)

    ax2.set_ylabel('Avoided Compound Event Frequency\n(percentage points)', fontsize=11)
    ax2.set_title('Mitigation Benefit: Avoided Compound Risk\nSSP5-8.5 → SSP1-2.6', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Mid-Century\n(2040-2069)', 'End-Century\n(2070-2099)'])
    ax2.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, [mid_pct, end_pct])):
        ax2.annotate(f'{pct:.0f}% less risk',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=11, fontweight='bold', color='darkgreen')

    # Panel 3: Spatial map of avoided risk (end century)
    ax3 = fig.add_subplot(2, 2, 3)

    # Use first model for spatial
    result = all_results[0]
    model = result["model"]

    ssp126_spatial = result["scenarios"].get("ssp126", {}).get("end_century", {}).get("spatial_data")
    ssp585_spatial = result["scenarios"].get("ssp585", {}).get("end_century", {}).get("spatial_data")

    if ssp126_spatial is not None and ssp585_spatial is not None:
        avoided_spatial = (ssp585_spatial - ssp126_spatial) * 100

        im3 = ax3.pcolormesh(avoided_spatial.lon, avoided_spatial.lat, avoided_spatial,
                            cmap="Greens", vmin=0, vmax=15, shading="auto")
        ax3.set_title(f'Avoided Compound Risk: End-Century\n{model} (SSP5-8.5 minus SSP1-2.6)', fontsize=11)
        ax3.set_xlabel("Longitude")
        ax3.set_ylabel("Latitude")
        plt.colorbar(im3, ax=ax3, label="Avoided frequency (pp)", shrink=0.8)

    # Panel 4: Model comparison
    ax4 = fig.add_subplot(2, 2, 4)

    model_names = benefits["models"]
    x = np.arange(len(model_names))
    width = 0.35

    ssp126_end = benefits["end_century"]["ssp126"]
    ssp585_end = benefits["end_century"]["ssp585"]

    bars1 = ax4.bar(x - width/2, ssp585_end, width, label='SSP5-8.5 (High)', color='#b2182b', alpha=0.8)
    bars2 = ax4.bar(x + width/2, ssp126_end, width, label='SSP1-2.6 (Low)', color='#2166ac', alpha=0.8)

    ax4.set_ylabel('Compound Event Frequency (%)', fontsize=11)
    ax4.set_title('End-Century Compound Risk by Model\nHigh vs Low Emissions', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add reduction annotations
    for i, (high, low) in enumerate(zip(ssp585_end, ssp126_end)):
        reduction = ((high - low) / high) * 100
        ax4.annotate(f'-{reduction:.0f}%',
                    xy=(i, max(high, low) + 0.5),
                    ha='center', fontsize=10, fontweight='bold', color='darkgreen')

    plt.tight_layout()

    output_path = ARTIFACTS_DIR / "mitigation_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def plot_regional_mitigation(all_results: list[dict]) -> Path:
    """Show regional mitigation benefits."""

    result = all_results[0]
    model = result["model"]

    ssp126_end = result["scenarios"].get("ssp126", {}).get("end_century", {}).get("spatial_data")
    ssp585_end = result["scenarios"].get("ssp585", {}).get("end_century", {}).get("spatial_data")

    if ssp126_end is None or ssp585_end is None:
        return None

    avoided = (ssp585_end - ssp126_end) * 100

    regions = {
        "Mediterranean": {"lat": (30, 45), "lon": (350, 40)},
        "Amazon": {"lat": (-20, 5), "lon": (285, 315)},
        "Southern Africa": {"lat": (-35, -15), "lon": (15, 40)},
        "Australia": {"lat": (-40, -15), "lon": (110, 155)},
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (region_name, bounds) in zip(axes.flat, regions.items()):
        lat_slice = slice(bounds["lat"][0], bounds["lat"][1])

        # Handle longitude
        lon_min, lon_max = bounds["lon"]

        try:
            if lon_min > lon_max:
                region1 = avoided.sel(lat=lat_slice, lon=slice(lon_min, 360))
                region2 = avoided.sel(lat=lat_slice, lon=slice(0, lon_max))
                region_data = xr.concat([region1, region2], dim="lon")
            else:
                region_data = avoided.sel(lat=lat_slice, lon=slice(lon_min, lon_max))
        except:
            region_data = avoided.sel(lat=lat_slice)

        im = ax.pcolormesh(region_data.lon, region_data.lat, region_data,
                          cmap="Greens", vmin=0, vmax=20, shading="auto")

        weights = np.cos(np.deg2rad(region_data.lat))
        mean_avoided = float(region_data.weighted(weights).mean().values)

        ax.set_title(f"{region_name}\nMean avoided risk: {mean_avoided:.1f} pp", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.colorbar(im, ax=ax, label="Avoided (pp)", shrink=0.8)

    fig.suptitle(f"Regional Mitigation Benefits: Avoided Compound Risk\n{model} End-Century (SSP5-8.5 → SSP1-2.6)",
                 fontsize=13, y=1.02)

    plt.tight_layout()

    output_path = ARTIFACTS_DIR / "mitigation_regional.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def generate_summary(all_results: list[dict], benefits: dict) -> dict:
    """Generate mitigation benefit summary."""

    summary = {
        "analysis": "Mitigation Benefit: SSP1-2.6 vs SSP5-8.5",
        "models": benefits["models"],
        "scenarios_compared": {
            "ssp126": "Low emissions - Paris Agreement aligned",
            "ssp585": "High emissions - Fossil-fueled development"
        },
        "findings": {}
    }

    for period in ["mid_century", "end_century"]:
        if benefits[period]["avoided"]:
            summary["findings"][period] = {
                "ssp585_frequency": round(np.mean(benefits[period]["ssp585"]), 2),
                "ssp126_frequency": round(np.mean(benefits[period]["ssp126"]), 2),
                "avoided_pp": round(np.mean(benefits[period]["avoided"]), 2),
                "avoided_percent": round(np.mean(benefits[period]["avoided_pct"]), 1),
            }

    # Key message
    end_avoided_pct = np.mean(benefits["end_century"]["avoided_pct"]) if benefits["end_century"]["avoided_pct"] else 0
    summary["key_message"] = f"Climate mitigation (SSP1-2.6 vs SSP5-8.5) avoids {end_avoided_pct:.0f}% of compound heat-drought risk by end of century"

    return summary


def main():
    print("="*70)
    print("MITIGATION BENEFIT ANALYSIS")
    print("SSP1-2.6 (Low Emissions) vs SSP5-8.5 (High Emissions)")
    print("="*70)
    print("\nQuantifying avoided compound heat-drought risk from climate mitigation")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Analyze each model
    all_results = []
    for model in MODELS:
        result = analyze_model(model)
        if result and result["scenarios"]:
            all_results.append(result)

    if not all_results:
        print("\nERROR: No models analyzed successfully")
        return

    # Calculate mitigation benefit
    print("\n" + "="*70)
    print("Calculating Mitigation Benefits")
    print("="*70)

    benefits = calculate_mitigation_benefit(all_results)

    # Generate visualizations
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)

    fig1_path = plot_mitigation_comparison(all_results, benefits)
    print(f"\nSaved: {fig1_path}")

    fig2_path = plot_regional_mitigation(all_results)
    if fig2_path:
        print(f"Saved: {fig2_path}")

    # Summary
    print("\n" + "="*70)
    print("MITIGATION BENEFIT SUMMARY")
    print("="*70)

    summary = generate_summary(all_results, benefits)

    print(f"\nModels: {', '.join(summary['models'])}")
    print(f"\nCompound Event Frequencies (Multi-Model Mean):")
    print(f"\n{'Period':<20} {'SSP5-8.5':>12} {'SSP1-2.6':>12} {'Avoided':>12} {'Reduction':>12}")
    print("-" * 70)

    for period in ["mid_century", "end_century"]:
        if period in summary["findings"]:
            f = summary["findings"][period]
            period_label = "Mid-Century" if period == "mid_century" else "End-Century"
            print(f"{period_label:<20} {f['ssp585_frequency']:>11.2f}% {f['ssp126_frequency']:>11.2f}% "
                  f"{f['avoided_pp']:>+11.2f}pp {f['avoided_percent']:>11.0f}%")

    print(f"\n{'='*70}")
    print(f"KEY FINDING: {summary['key_message']}")
    print(f"{'='*70}")

    # Save summary
    summary_path = ARTIFACTS_DIR / "mitigation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
