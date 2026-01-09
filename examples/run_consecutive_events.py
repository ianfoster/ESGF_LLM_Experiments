#!/usr/bin/env python3
"""
Consecutive Compound Event Analysis

Analyze multi-month persistent compound heat-drought events.
Single months of compound stress are bad; consecutive months are catastrophic.

Key Questions:
- How does the frequency of 2+, 3+, 6+ month compound events change?
- Does compound event persistence increase under climate change?
- Which regions see the largest increase in multi-month droughts?
"""

import warnings
warnings.filterwarnings("ignore")

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import json
from scipy import ndimage

# Configuration
DATA_DIR = Path("data/compound_hazard")
ARTIFACTS_DIR = Path("Artifacts")

MODELS = ["GFDL-ESM4", "MIROC6"]

PERIODS = {
    "baseline": (1985, 2014),
    "historical": (1995, 2014),
    "end_century": (2070, 2099),
}

HEAT_PERCENTILE = 90
DROUGHT_PERCENTILE = 10

# Consecutive event thresholds (months)
CONSECUTIVE_THRESHOLDS = [2, 3, 6]


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


def identify_compound_events(tas: xr.DataArray, pr: xr.DataArray,
                             tas_thresh: xr.DataArray, pr_thresh: xr.DataArray) -> xr.DataArray:
    """Identify compound events (boolean array)."""
    tas_th = tas_thresh.sel(month=tas.time.dt.month).drop_vars("month")
    pr_th = pr_thresh.sel(month=pr.time.dt.month).drop_vars("month")
    return (tas > tas_th) & (pr < pr_th)


def count_consecutive_events(compound_bool: np.ndarray, min_length: int) -> int:
    """
    Count the number of events with at least min_length consecutive True values.

    Args:
        compound_bool: 1D boolean array of compound events over time
        min_length: Minimum consecutive months to count as an event

    Returns:
        Number of distinct multi-month events
    """
    if not np.any(compound_bool):
        return 0

    # Label connected regions
    labeled, num_features = ndimage.label(compound_bool.astype(int))

    # Count events meeting minimum length
    count = 0
    for i in range(1, num_features + 1):
        event_length = np.sum(labeled == i)
        if event_length >= min_length:
            count += 1

    return count


def compute_consecutive_frequency(compound: xr.DataArray, min_months: int) -> xr.DataArray:
    """
    Compute frequency of consecutive compound events at each grid cell.

    Returns: Events per decade at each location
    """
    compound_computed = compound.compute()

    # Get dimensions
    n_time = compound_computed.time.size
    n_years = n_time / 12

    # Initialize result array
    result = np.zeros((compound_computed.lat.size, compound_computed.lon.size))

    # Process each grid cell
    for i in range(compound_computed.lat.size):
        for j in range(compound_computed.lon.size):
            time_series = compound_computed.values[:, i, j]
            n_events = count_consecutive_events(time_series, min_months)
            # Convert to events per decade
            result[i, j] = n_events / n_years * 10

    # Create DataArray
    result_da = xr.DataArray(
        result,
        dims=["lat", "lon"],
        coords={"lat": compound_computed.lat, "lon": compound_computed.lon}
    )

    return result_da


def compute_mean_event_length(compound: xr.DataArray) -> xr.DataArray:
    """
    Compute mean length of compound events at each grid cell.
    """
    compound_computed = compound.compute()

    result = np.zeros((compound_computed.lat.size, compound_computed.lon.size))

    for i in range(compound_computed.lat.size):
        for j in range(compound_computed.lon.size):
            time_series = compound_computed.values[:, i, j]

            if not np.any(time_series):
                result[i, j] = 0
                continue

            # Label events
            labeled, num_features = ndimage.label(time_series.astype(int))

            if num_features == 0:
                result[i, j] = 0
                continue

            # Calculate mean length
            lengths = []
            for k in range(1, num_features + 1):
                lengths.append(np.sum(labeled == k))

            result[i, j] = np.mean(lengths)

    result_da = xr.DataArray(
        result,
        dims=["lat", "lon"],
        coords={"lat": compound_computed.lat, "lon": compound_computed.lon}
    )

    return result_da


def analyze_model(source_id: str) -> Optional[dict]:
    """Analyze consecutive compound events for one model."""
    print(f"\n{'='*60}")
    print(f"Consecutive Event Analysis: {source_id}")
    print(f"{'='*60}")

    # Load data
    hist_tas = sorted((DATA_DIR / source_id / "historical" / "tas").glob("*.nc"))
    hist_pr = sorted((DATA_DIR / source_id / "historical" / "pr").glob("*.nc"))
    ssp_tas = sorted((DATA_DIR / source_id / "ssp585" / "tas").glob("*.nc"))
    ssp_pr = sorted((DATA_DIR / source_id / "ssp585" / "pr").glob("*.nc"))

    if not all([hist_tas, hist_pr, ssp_tas, ssp_pr]):
        print("  Missing data files")
        return None

    # Load baseline for thresholds
    print("  Loading baseline thresholds...")
    tas_base = load_data(hist_tas, "tas", *PERIODS["baseline"])
    pr_base = load_data(hist_pr, "pr", *PERIODS["baseline"])

    if tas_base is None or pr_base is None:
        return None

    tas_thresh, pr_thresh = compute_thresholds(tas_base, pr_base)

    results = {"model": source_id, "periods": {}}

    # Analyze each period
    for period_name in ["historical", "end_century"]:
        start, end = PERIODS[period_name]
        print(f"\n  Analyzing {period_name} ({start}-{end})...")

        if start < 2015:
            tas_files, pr_files = hist_tas, hist_pr
        else:
            tas_files, pr_files = ssp_tas, ssp_pr

        tas = load_data(tas_files, "tas", start, end)
        pr = load_data(pr_files, "pr", start, end)

        if tas is None or pr is None:
            continue

        if tas.lat.size != pr.lat.size:
            pr = pr.interp(lat=tas.lat, lon=tas.lon)

        # Identify compound events
        compound = identify_compound_events(tas, pr, tas_thresh, pr_thresh)

        period_results = {"consecutive": {}}

        # Compute consecutive event frequencies
        for min_months in CONSECUTIVE_THRESHOLDS:
            print(f"    Computing {min_months}+ month events...", end=" ", flush=True)
            freq = compute_consecutive_frequency(compound, min_months)

            # Global mean
            weights = np.cos(np.deg2rad(freq.lat))
            global_freq = float(freq.weighted(weights).mean().values)

            period_results["consecutive"][min_months] = {
                "global_events_per_decade": global_freq,
                "spatial_data": freq,
            }
            print(f"{global_freq:.2f} events/decade")

        # Compute mean event length
        print("    Computing mean event length...", end=" ", flush=True)
        mean_length = compute_mean_event_length(compound)
        weights = np.cos(np.deg2rad(mean_length.lat))
        global_mean_length = float(mean_length.weighted(weights).mean().values)
        period_results["mean_event_length"] = {
            "global_months": global_mean_length,
            "spatial_data": mean_length,
        }
        print(f"{global_mean_length:.2f} months")

        results["periods"][period_name] = period_results

    return results


def plot_consecutive_analysis(all_results: list[dict]) -> Path:
    """Create comprehensive consecutive event visualization."""

    fig = plt.figure(figsize=(16, 12))

    # Panel 1: Consecutive event frequency by threshold
    ax1 = fig.add_subplot(2, 2, 1)

    x = np.arange(len(CONSECUTIVE_THRESHOLDS))
    width = 0.35

    hist_means = []
    future_means = []

    for min_months in CONSECUTIVE_THRESHOLDS:
        hist_vals = []
        future_vals = []
        for result in all_results:
            h = result["periods"].get("historical", {}).get("consecutive", {}).get(min_months, {}).get("global_events_per_decade")
            f = result["periods"].get("end_century", {}).get("consecutive", {}).get(min_months, {}).get("global_events_per_decade")
            if h is not None:
                hist_vals.append(h)
            if f is not None:
                future_vals.append(f)

        hist_means.append(np.mean(hist_vals) if hist_vals else 0)
        future_means.append(np.mean(future_vals) if future_vals else 0)

    bars1 = ax1.bar(x - width/2, hist_means, width, label='Historical (1995-2014)',
                    color='#2166ac', alpha=0.8)
    bars2 = ax1.bar(x + width/2, future_means, width, label='End-Century (2070-2099)',
                    color='#b2182b', alpha=0.8)

    ax1.set_ylabel('Events per Decade (global mean)', fontsize=11)
    ax1.set_title('Multi-Month Compound Event Frequency\nBy Minimum Duration', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{m}+ months' for m in CONSECUTIVE_THRESHOLDS])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add increase labels
    for i, (h, f) in enumerate(zip(hist_means, future_means)):
        if h > 0:
            increase = ((f - h) / h) * 100
            ax1.annotate(f'+{increase:.0f}%', xy=(i + width/2, f + 0.1),
                        ha='center', fontsize=10, fontweight='bold', color='#b2182b')

    # Panel 2: Mean event length
    ax2 = fig.add_subplot(2, 2, 2)

    model_names = []
    hist_lengths = []
    future_lengths = []

    for result in all_results:
        model_names.append(result["model"])
        h = result["periods"].get("historical", {}).get("mean_event_length", {}).get("global_months", 0)
        f = result["periods"].get("end_century", {}).get("mean_event_length", {}).get("global_months", 0)
        hist_lengths.append(h)
        future_lengths.append(f)

    x = np.arange(len(model_names))
    bars1 = ax2.bar(x - width/2, hist_lengths, width, label='Historical',
                    color='#2166ac', alpha=0.8)
    bars2 = ax2.bar(x + width/2, future_lengths, width, label='End-Century',
                    color='#b2182b', alpha=0.8)

    ax2.set_ylabel('Mean Event Length (months)', fontsize=11)
    ax2.set_title('Average Compound Event Duration\nHistorical vs End-Century', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add change annotations
    for i, (h, f) in enumerate(zip(hist_lengths, future_lengths)):
        change = f - h
        ax2.annotate(f'+{change:.1f}mo', xy=(i + width/2, f + 0.05),
                    ha='center', fontsize=10, fontweight='bold', color='#b2182b')

    # Panel 3: Spatial map of 3+ month event change
    ax3 = fig.add_subplot(2, 2, 3)

    result = all_results[0]
    model = result["model"]

    hist_3mo = result["periods"]["historical"]["consecutive"][3]["spatial_data"]
    future_3mo = result["periods"]["end_century"]["consecutive"][3]["spatial_data"]
    change_3mo = future_3mo - hist_3mo

    im3 = ax3.pcolormesh(change_3mo.lon, change_3mo.lat, change_3mo,
                        cmap="YlOrRd", vmin=0, vmax=5, shading="auto")
    ax3.set_title(f'Change in 3+ Month Compound Events\n{model} (events/decade)', fontsize=11)
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    plt.colorbar(im3, ax=ax3, label="Change (events/decade)", shrink=0.8)

    # Panel 4: Spatial map of mean event length change
    ax4 = fig.add_subplot(2, 2, 4)

    hist_length = result["periods"]["historical"]["mean_event_length"]["spatial_data"]
    future_length = result["periods"]["end_century"]["mean_event_length"]["spatial_data"]
    length_change = future_length - hist_length

    im4 = ax4.pcolormesh(length_change.lon, length_change.lat, length_change,
                        cmap="YlOrRd", vmin=0, vmax=2, shading="auto")
    ax4.set_title(f'Change in Mean Event Duration\n{model} (months)', fontsize=11)
    ax4.set_xlabel("Longitude")
    ax4.set_ylabel("Latitude")
    plt.colorbar(im4, ax=ax4, label="Change (months)", shrink=0.8)

    plt.tight_layout()

    output_path = ARTIFACTS_DIR / "consecutive_events.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def plot_regional_persistence(all_results: list[dict]) -> Path:
    """Show regional patterns of increasing event persistence."""

    result = all_results[0]
    model = result["model"]

    # 6+ month events - the really severe ones
    hist_6mo = result["periods"]["historical"]["consecutive"][6]["spatial_data"]
    future_6mo = result["periods"]["end_century"]["consecutive"][6]["spatial_data"]
    change_6mo = future_6mo - hist_6mo

    regions = {
        "Mediterranean": {"lat": (30, 45), "lon": (350, 40)},
        "Amazon": {"lat": (-20, 5), "lon": (285, 315)},
        "Southern Africa": {"lat": (-35, -15), "lon": (15, 40)},
        "Australia": {"lat": (-40, -15), "lon": (110, 155)},
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (region_name, bounds) in zip(axes.flat, regions.items()):
        lat_slice = slice(bounds["lat"][0], bounds["lat"][1])
        lon_min, lon_max = bounds["lon"]

        try:
            if lon_min > lon_max:
                region1 = change_6mo.sel(lat=lat_slice, lon=slice(lon_min, 360))
                region2 = change_6mo.sel(lat=lat_slice, lon=slice(0, lon_max))
                region_data = xr.concat([region1, region2], dim="lon")
            else:
                region_data = change_6mo.sel(lat=lat_slice, lon=slice(lon_min, lon_max))
        except:
            region_data = change_6mo.sel(lat=lat_slice)

        im = ax.pcolormesh(region_data.lon, region_data.lat, region_data,
                          cmap="YlOrRd", vmin=0, vmax=3, shading="auto")

        weights = np.cos(np.deg2rad(region_data.lat))
        mean_change = float(region_data.weighted(weights).mean().values)

        ax.set_title(f"{region_name}\nMean change: +{mean_change:.2f} events/decade", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.colorbar(im, ax=ax, label="Change", shrink=0.8)

    fig.suptitle(f"6+ Month Compound Events: Regional Change\n{model} (events per decade)",
                 fontsize=13, y=1.02)

    plt.tight_layout()

    output_path = ARTIFACTS_DIR / "consecutive_regional.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def generate_summary(all_results: list[dict]) -> dict:
    """Generate consecutive event summary."""

    summary = {
        "analysis": "Consecutive Compound Event Analysis",
        "models": [r["model"] for r in all_results],
        "thresholds_months": CONSECUTIVE_THRESHOLDS,
        "findings": {}
    }

    # Statistics for each threshold
    for min_months in CONSECUTIVE_THRESHOLDS:
        hist_vals = []
        future_vals = []

        for result in all_results:
            h = result["periods"].get("historical", {}).get("consecutive", {}).get(min_months, {}).get("global_events_per_decade")
            f = result["periods"].get("end_century", {}).get("consecutive", {}).get(min_months, {}).get("global_events_per_decade")
            if h is not None:
                hist_vals.append(h)
            if f is not None:
                future_vals.append(f)

        if hist_vals and future_vals:
            hist_mean = np.mean(hist_vals)
            future_mean = np.mean(future_vals)
            pct_change = ((future_mean - hist_mean) / hist_mean) * 100 if hist_mean > 0 else 0

            summary["findings"][f"{min_months}_month_events"] = {
                "historical_per_decade": round(hist_mean, 2),
                "future_per_decade": round(future_mean, 2),
                "percent_increase": round(pct_change, 1),
            }

    # Mean event length
    hist_lengths = []
    future_lengths = []
    for result in all_results:
        h = result["periods"].get("historical", {}).get("mean_event_length", {}).get("global_months")
        f = result["periods"].get("end_century", {}).get("mean_event_length", {}).get("global_months")
        if h is not None:
            hist_lengths.append(h)
        if f is not None:
            future_lengths.append(f)

    if hist_lengths and future_lengths:
        summary["findings"]["mean_event_length"] = {
            "historical_months": round(np.mean(hist_lengths), 2),
            "future_months": round(np.mean(future_lengths), 2),
            "increase_months": round(np.mean(future_lengths) - np.mean(hist_lengths), 2),
        }

    return summary


def main():
    print("="*70)
    print("CONSECUTIVE COMPOUND EVENT ANALYSIS")
    print("Multi-Month Persistent Heat-Drought Events")
    print("="*70)
    print("\nSingle months of compound stress are bad.")
    print("Consecutive months are catastrophic.")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Analyze each model
    all_results = []
    for model in MODELS:
        result = analyze_model(model)
        if result and result["periods"]:
            all_results.append(result)

    if not all_results:
        print("\nERROR: No models analyzed successfully")
        return

    # Generate visualizations
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)

    fig1_path = plot_consecutive_analysis(all_results)
    print(f"\nSaved: {fig1_path}")

    fig2_path = plot_regional_persistence(all_results)
    print(f"Saved: {fig2_path}")

    # Summary
    print("\n" + "="*70)
    print("CONSECUTIVE EVENT SUMMARY")
    print("="*70)

    summary = generate_summary(all_results)

    print(f"\nMulti-Month Compound Events (Global Mean, Events per Decade):")
    print(f"\n{'Duration':<15} {'Historical':>12} {'End-Century':>12} {'Change':>12}")
    print("-" * 55)

    for min_months in CONSECUTIVE_THRESHOLDS:
        key = f"{min_months}_month_events"
        if key in summary["findings"]:
            f = summary["findings"][key]
            print(f"{min_months}+ months{'':<7} {f['historical_per_decade']:>11.2f} "
                  f"{f['future_per_decade']:>11.2f} {f['percent_increase']:>+11.0f}%")

    if "mean_event_length" in summary["findings"]:
        mel = summary["findings"]["mean_event_length"]
        print(f"\nMean Event Length:")
        print(f"  Historical: {mel['historical_months']:.2f} months")
        print(f"  End-Century: {mel['future_months']:.2f} months")
        print(f"  Increase: +{mel['increase_months']:.2f} months")

    # Key finding
    if "6_month_events" in summary["findings"]:
        f6 = summary["findings"]["6_month_events"]
        print(f"\n{'='*70}")
        print(f"KEY FINDING: 6+ month compound events increase by {f6['percent_increase']:.0f}%")
        print(f"             ({f6['historical_per_decade']:.1f} → {f6['future_per_decade']:.1f} per decade)")
        print(f"{'='*70}")

    # Save summary
    summary_path = ARTIFACTS_DIR / "consecutive_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
