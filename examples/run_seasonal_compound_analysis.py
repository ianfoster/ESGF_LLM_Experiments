#!/usr/bin/env python3
"""
Seasonal Compound Hazard Analysis

Follow-up analysis examining WHEN compound heat-drought events occur:
- Which seasons see the most compound events?
- How does the seasonal distribution change under climate change?
- Are summer growing-season events increasing faster?
"""

import warnings
warnings.filterwarnings("ignore")

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import json

# Configuration - reuse data from compound hazard analysis
DATA_DIR = Path("data/compound_hazard")
ARTIFACTS_DIR = Path("Artifacts")

MODELS = ["GFDL-ESM4", "MIROC6", "NorESM2-LM"]

PERIODS = {
    "baseline": (1985, 2014),
    "historical": (1995, 2014),
    "mid_century": (2040, 2069),
    "end_century": (2070, 2099),
}

HEAT_PERCENTILE = 90
DROUGHT_PERCENTILE = 10

# Season definitions
SEASONS = {
    "DJF": [12, 1, 2],   # Winter (NH) / Summer (SH)
    "MAM": [3, 4, 5],    # Spring (NH) / Autumn (SH)
    "JJA": [6, 7, 8],    # Summer (NH) / Winter (SH)
    "SON": [9, 10, 11],  # Autumn (NH) / Spring (SH)
}

SEASON_NAMES = {
    "DJF": "Dec-Jan-Feb",
    "MAM": "Mar-Apr-May",
    "JJA": "Jun-Jul-Aug",
    "SON": "Sep-Oct-Nov",
}


def load_and_process_data(
    files: list[Path],
    variable: str,
    start_year: int,
    end_year: int
) -> Optional[xr.DataArray]:
    """Load NetCDF files and extract time period."""
    try:
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
        ds = ds.sortby("time")

        # Remove duplicate time values
        _, unique_indices = np.unique(ds.time.values, return_index=True)
        ds = ds.isel(time=sorted(unique_indices))

        da = ds[variable]
        da = da.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

        if da.time.size == 0:
            return None

        return da
    except Exception as e:
        print(f"    Load error: {e}")
        return None


def compute_seasonal_thresholds(
    tas_baseline: xr.DataArray,
    pr_baseline: xr.DataArray,
    heat_pct: int = 90,
    drought_pct: int = 10
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute percentile thresholds by month (captures seasonality)."""
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
) -> xr.DataArray:
    """Identify compound heat-drought events."""
    tas_thresh_aligned = tas_threshold.sel(month=tas.time.dt.month)
    pr_thresh_aligned = pr_threshold.sel(month=pr.time.dt.month)

    tas_thresh_aligned = tas_thresh_aligned.drop_vars("month")
    pr_thresh_aligned = pr_thresh_aligned.drop_vars("month")

    is_hot = tas > tas_thresh_aligned
    is_dry = pr < pr_thresh_aligned

    return is_hot & is_dry


def compute_seasonal_frequency(
    compound_events: xr.DataArray,
    season_months: list[int]
) -> xr.DataArray:
    """Compute compound event frequency for a specific season."""
    # Select only months in this season
    month_mask = compound_events.time.dt.month.isin(season_months)
    season_events = compound_events.where(month_mask, drop=True)

    if season_events.time.size == 0:
        return None

    return season_events.mean(dim="time")


def analyze_model_seasonal(source_id: str) -> Optional[dict]:
    """Analyze seasonal compound hazard patterns for one model."""
    print(f"\n{'='*60}")
    print(f"Seasonal Analysis: {source_id}")
    print(f"{'='*60}")

    # Load data files
    hist_tas_files = sorted((DATA_DIR / source_id / "historical" / "tas").glob("*.nc"))
    hist_pr_files = sorted((DATA_DIR / source_id / "historical" / "pr").glob("*.nc"))
    ssp_tas_files = sorted((DATA_DIR / source_id / "ssp585" / "tas").glob("*.nc"))
    ssp_pr_files = sorted((DATA_DIR / source_id / "ssp585" / "pr").glob("*.nc"))

    if not all([hist_tas_files, hist_pr_files, ssp_tas_files, ssp_pr_files]):
        print(f"  Missing data files for {source_id}")
        return None

    # Load baseline for thresholds
    print("  Loading baseline data...")
    tas_baseline = load_and_process_data(hist_tas_files, "tas", *PERIODS["baseline"])
    pr_baseline = load_and_process_data(hist_pr_files, "pr", *PERIODS["baseline"])

    if tas_baseline is None or pr_baseline is None:
        print("  Failed to load baseline")
        return None

    # Compute thresholds
    print("  Computing seasonal thresholds...")
    tas_thresh, pr_thresh = compute_seasonal_thresholds(
        tas_baseline, pr_baseline, HEAT_PERCENTILE, DROUGHT_PERCENTILE
    )

    results = {"model": source_id, "seasonal_data": {}}

    # Analyze each period
    for period_name in ["historical", "end_century"]:
        start_year, end_year = PERIODS[period_name]
        print(f"\n  Analyzing {period_name} ({start_year}-{end_year})...")

        if start_year < 2015:
            tas_files, pr_files = hist_tas_files, hist_pr_files
        else:
            tas_files, pr_files = ssp_tas_files, ssp_pr_files

        tas_data = load_and_process_data(tas_files, "tas", start_year, end_year)
        pr_data = load_and_process_data(pr_files, "pr", start_year, end_year)

        if tas_data is None or pr_data is None:
            continue

        # Align grids if needed
        if tas_data.lat.size != pr_data.lat.size:
            pr_data = pr_data.interp(lat=tas_data.lat, lon=tas_data.lon)

        # Identify compound events
        compound = identify_compound_events(tas_data, pr_data, tas_thresh, pr_thresh)

        # Compute frequency by season
        period_seasonal = {}
        for season_name, season_months in SEASONS.items():
            freq = compute_seasonal_frequency(compound, season_months)
            if freq is not None:
                freq_computed = freq.compute()

                # Global weighted mean
                weights = np.cos(np.deg2rad(freq_computed.lat))
                global_freq = float(freq_computed.weighted(weights).mean().values) * 100

                period_seasonal[season_name] = {
                    "global_frequency": global_freq,
                    "spatial_data": freq_computed,
                }
                print(f"    {season_name}: {global_freq:.2f}%")

        results["seasonal_data"][period_name] = period_seasonal

    return results


def plot_seasonal_analysis(all_results: list[dict]) -> Path:
    """Create comprehensive seasonal analysis visualization."""

    fig = plt.figure(figsize=(16, 12))

    # Panel 1: Seasonal frequency comparison (bar chart)
    ax1 = fig.add_subplot(2, 2, 1)

    seasons = list(SEASONS.keys())
    x = np.arange(len(seasons))
    width = 0.35

    # Average across models
    hist_means = []
    future_means = []

    for season in seasons:
        hist_vals = []
        future_vals = []
        for result in all_results:
            if "historical" in result["seasonal_data"] and season in result["seasonal_data"]["historical"]:
                hist_vals.append(result["seasonal_data"]["historical"][season]["global_frequency"])
            if "end_century" in result["seasonal_data"] and season in result["seasonal_data"]["end_century"]:
                future_vals.append(result["seasonal_data"]["end_century"][season]["global_frequency"])

        hist_means.append(np.mean(hist_vals) if hist_vals else 0)
        future_means.append(np.mean(future_vals) if future_vals else 0)

    bars1 = ax1.bar(x - width/2, hist_means, width, label='Historical (1995-2014)',
                    color='#2166ac', alpha=0.8)
    bars2 = ax1.bar(x + width/2, future_means, width, label='End-Century (2070-2099)',
                    color='#b2182b', alpha=0.8)

    ax1.set_ylabel('Compound Event Frequency (%)', fontsize=11)
    ax1.set_title('Seasonal Distribution of Compound Heat-Drought Events\n(Multi-Model Mean)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([SEASON_NAMES[s] for s in seasons])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add increase labels
    for i, (h, f) in enumerate(zip(hist_means, future_means)):
        if h > 0:
            increase = ((f - h) / h) * 100
            ax1.annotate(f'+{increase:.0f}%', xy=(i + width/2, f + 0.5),
                        ha='center', fontsize=9, fontweight='bold', color='#b2182b')

    # Panel 2: Seasonal increase by model
    ax2 = fig.add_subplot(2, 2, 2)

    colors = {"GFDL-ESM4": "#1b9e77", "MIROC6": "#d95f02", "NorESM2-LM": "#7570b3"}
    bar_width = 0.25

    for i, result in enumerate(all_results):
        model = result["model"]
        increases = []
        for season in seasons:
            hist_val = result["seasonal_data"].get("historical", {}).get(season, {}).get("global_frequency", 0)
            future_val = result["seasonal_data"].get("end_century", {}).get(season, {}).get("global_frequency", 0)
            increases.append(future_val - hist_val)

        offset = (i - 1) * bar_width
        ax2.bar(x + offset, increases, bar_width, label=model,
                color=colors.get(model, "gray"), alpha=0.8)

    ax2.set_ylabel('Increase in Frequency (percentage points)', fontsize=11)
    ax2.set_title('Seasonal Compound Event Increase by Model\n(Historical → 2070-2099)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([SEASON_NAMES[s] for s in seasons])
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Northern Hemisphere summer (JJA) spatial change
    ax3 = fig.add_subplot(2, 2, 3)

    # Use first model for spatial map
    result = all_results[0]
    model = result["model"]

    if "JJA" in result["seasonal_data"].get("historical", {}) and "JJA" in result["seasonal_data"].get("end_century", {}):
        hist_jja = result["seasonal_data"]["historical"]["JJA"]["spatial_data"]
        future_jja = result["seasonal_data"]["end_century"]["JJA"]["spatial_data"]
        change_jja = (future_jja - hist_jja) * 100

        im3 = ax3.pcolormesh(change_jja.lon, change_jja.lat, change_jja,
                            cmap="YlOrRd", vmin=0, vmax=25, shading="auto")
        ax3.set_title(f'Summer (JJA) Compound Event Change\n{model} SSP5-8.5', fontsize=11)
        ax3.set_xlabel("Longitude")
        ax3.set_ylabel("Latitude")
        plt.colorbar(im3, ax=ax3, label="Change (pp)", shrink=0.8)

    # Panel 4: Southern Hemisphere summer (DJF) spatial change
    ax4 = fig.add_subplot(2, 2, 4)

    if "DJF" in result["seasonal_data"].get("historical", {}) and "DJF" in result["seasonal_data"].get("end_century", {}):
        hist_djf = result["seasonal_data"]["historical"]["DJF"]["spatial_data"]
        future_djf = result["seasonal_data"]["end_century"]["DJF"]["spatial_data"]
        change_djf = (future_djf - hist_djf) * 100

        im4 = ax4.pcolormesh(change_djf.lon, change_djf.lat, change_djf,
                            cmap="YlOrRd", vmin=0, vmax=25, shading="auto")
        ax4.set_title(f'Winter/SH Summer (DJF) Compound Event Change\n{model} SSP5-8.5', fontsize=11)
        ax4.set_xlabel("Longitude")
        ax4.set_ylabel("Latitude")
        plt.colorbar(im4, ax=ax4, label="Change (pp)", shrink=0.8)

    plt.tight_layout()

    output_path = ARTIFACTS_DIR / "compound_hazard_seasonal.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def plot_hemispheric_seasonal(all_results: list[dict]) -> Path:
    """Create hemispheric seasonal analysis - NH vs SH patterns."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Use first model with complete data
    result = all_results[0]
    model = result["model"]

    seasons = list(SEASONS.keys())

    # Calculate NH and SH frequencies
    nh_hist = []
    nh_future = []
    sh_hist = []
    sh_future = []

    for season in seasons:
        hist_data = result["seasonal_data"].get("historical", {}).get(season, {}).get("spatial_data")
        future_data = result["seasonal_data"].get("end_century", {}).get(season, {}).get("spatial_data")

        if hist_data is not None and future_data is not None:
            # Northern Hemisphere (lat > 0)
            nh_hist_data = hist_data.sel(lat=slice(0, 90))
            nh_future_data = future_data.sel(lat=slice(0, 90))

            weights_nh = np.cos(np.deg2rad(nh_hist_data.lat))
            nh_hist.append(float(nh_hist_data.weighted(weights_nh).mean().values) * 100)
            nh_future.append(float(nh_future_data.weighted(weights_nh).mean().values) * 100)

            # Southern Hemisphere (lat < 0)
            sh_hist_data = hist_data.sel(lat=slice(-90, 0))
            sh_future_data = future_data.sel(lat=slice(-90, 0))

            weights_sh = np.cos(np.deg2rad(sh_hist_data.lat))
            sh_hist.append(float(sh_hist_data.weighted(weights_sh).mean().values) * 100)
            sh_future.append(float(sh_future_data.weighted(weights_sh).mean().values) * 100)
        else:
            nh_hist.append(0)
            nh_future.append(0)
            sh_hist.append(0)
            sh_future.append(0)

    x = np.arange(len(seasons))
    width = 0.35

    # Northern Hemisphere
    ax1 = axes[0]
    ax1.bar(x - width/2, nh_hist, width, label='Historical', color='#2166ac', alpha=0.8)
    ax1.bar(x + width/2, nh_future, width, label='End-Century', color='#b2182b', alpha=0.8)
    ax1.set_ylabel('Compound Event Frequency (%)', fontsize=11)
    ax1.set_title(f'Northern Hemisphere Seasonal Pattern\n{model}', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([SEASON_NAMES[s] for s in seasons])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(nh_future + sh_future) * 1.2)

    # Highlight NH summer
    ax1.axvspan(1.5, 2.5, alpha=0.1, color='red', label='_nolegend_')
    ax1.text(2, ax1.get_ylim()[1]*0.95, 'NH\nSummer', ha='center', fontsize=9, color='darkred')

    # Southern Hemisphere
    ax2 = axes[1]
    ax2.bar(x - width/2, sh_hist, width, label='Historical', color='#2166ac', alpha=0.8)
    ax2.bar(x + width/2, sh_future, width, label='End-Century', color='#b2182b', alpha=0.8)
    ax2.set_ylabel('Compound Event Frequency (%)', fontsize=11)
    ax2.set_title(f'Southern Hemisphere Seasonal Pattern\n{model}', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([SEASON_NAMES[s] for s in seasons])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(nh_future + sh_future) * 1.2)

    # Highlight SH summer
    ax2.axvspan(-0.5, 0.5, alpha=0.1, color='red', label='_nolegend_')
    ax2.text(0, ax2.get_ylim()[1]*0.95, 'SH\nSummer', ha='center', fontsize=9, color='darkred')

    plt.tight_layout()

    output_path = ARTIFACTS_DIR / "compound_hazard_hemispheric.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def generate_seasonal_summary(all_results: list[dict]) -> dict:
    """Generate summary statistics for seasonal analysis."""

    seasons = list(SEASONS.keys())

    summary = {
        "analysis": "Seasonal Compound Hazard Breakdown",
        "models": [r["model"] for r in all_results],
        "seasons": {s: SEASON_NAMES[s] for s in seasons},
        "findings": {}
    }

    # Calculate statistics per season
    for season in seasons:
        hist_vals = []
        future_vals = []

        for result in all_results:
            h = result["seasonal_data"].get("historical", {}).get(season, {}).get("global_frequency")
            f = result["seasonal_data"].get("end_century", {}).get(season, {}).get("global_frequency")
            if h is not None:
                hist_vals.append(h)
            if f is not None:
                future_vals.append(f)

        if hist_vals and future_vals:
            hist_mean = np.mean(hist_vals)
            future_mean = np.mean(future_vals)
            abs_increase = future_mean - hist_mean
            pct_increase = ((future_mean - hist_mean) / hist_mean) * 100 if hist_mean > 0 else 0

            summary["findings"][season] = {
                "historical_frequency": round(hist_mean, 2),
                "future_frequency": round(future_mean, 2),
                "absolute_increase_pp": round(abs_increase, 2),
                "percent_increase": round(pct_increase, 1),
            }

    # Find most affected season
    max_increase = 0
    max_season = None
    for season, data in summary["findings"].items():
        if data["absolute_increase_pp"] > max_increase:
            max_increase = data["absolute_increase_pp"]
            max_season = season

    summary["key_finding"] = {
        "most_affected_season": max_season,
        "season_name": SEASON_NAMES.get(max_season, ""),
        "increase": max_increase,
        "interpretation": f"The {SEASON_NAMES.get(max_season, '')} season sees the largest increase in compound events"
    }

    return summary


def main():
    print("="*70)
    print("SEASONAL COMPOUND HAZARD ANALYSIS")
    print("When Do Compound Heat-Drought Events Occur?")
    print("="*70)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Analyze each model
    all_results = []
    for model in MODELS:
        result = analyze_model_seasonal(model)
        if result and result["seasonal_data"]:
            all_results.append(result)

    if not all_results:
        print("\nERROR: No models analyzed successfully")
        return

    # Generate visualizations
    print("\n" + "="*70)
    print("Generating Seasonal Visualizations")
    print("="*70)

    fig1_path = plot_seasonal_analysis(all_results)
    print(f"\nSaved: {fig1_path}")

    fig2_path = plot_hemispheric_seasonal(all_results)
    print(f"Saved: {fig2_path}")

    # Generate summary
    print("\n" + "="*70)
    print("SEASONAL ANALYSIS SUMMARY")
    print("="*70)

    summary = generate_seasonal_summary(all_results)

    print(f"\nSeasonal Compound Event Frequencies (Multi-Model Mean):\n")
    print(f"{'Season':<15} {'Historical':>12} {'End-Century':>12} {'Change':>10} {'% Increase':>12}")
    print("-" * 65)

    for season in SEASONS.keys():
        if season in summary["findings"]:
            data = summary["findings"][season]
            print(f"{SEASON_NAMES[season]:<15} {data['historical_frequency']:>11.2f}% "
                  f"{data['future_frequency']:>11.2f}% {data['absolute_increase_pp']:>+9.2f}pp "
                  f"{data['percent_increase']:>+11.1f}%")

    print(f"\n{'='*70}")
    print(f"KEY FINDING: {summary['key_finding']['interpretation']}")
    print(f"             (+{summary['key_finding']['increase']:.1f} percentage points)")
    print(f"{'='*70}")

    # Save summary
    summary_path = ARTIFACTS_DIR / "compound_hazard_seasonal_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
