#!/usr/bin/env python3
"""
Teleconnection Analysis: Synchronized Compound Events

Research Questions:
1. Are compound heat-drought events correlated across distant regions?
2. Does El Niño/La Niña drive synchronized compound events?
3. How often do multiple breadbasket regions experience simultaneous compound stress?
4. Do teleconnection patterns change under climate change?

This matters because synchronized events in multiple regions simultaneously
can cause global food crises, correlated economic losses, and overwhelm
humanitarian response capacity.
"""

import warnings
warnings.filterwarnings("ignore")

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from typing import Optional
import json
from scipy import stats
from itertools import combinations

DATA_DIR = Path("data/compound_hazard")
ARTIFACTS_DIR = Path("Artifacts")

MODEL = "GFDL-ESM4"  # Use single model for cleaner teleconnection signal

PERIODS = {
    "baseline": (1985, 2014),
    "historical": (1950, 2014),  # Longer period for teleconnection analysis
    "future": (2050, 2099),       # Future period
}

HEAT_PERCENTILE = 90
DROUGHT_PERCENTILE = 10

# Key agricultural/populated regions for teleconnection analysis
REGIONS = {
    "Amazon": {"lat": (-15, 5), "lon": (-70, -50)},
    "Sahel": {"lat": (10, 20), "lon": (-10, 30)},
    "Southern_Africa": {"lat": (-30, -15), "lon": (20, 35)},
    "Mediterranean": {"lat": (35, 45), "lon": (-5, 25)},
    "South_Asia": {"lat": (15, 30), "lon": (70, 90)},
    "East_Asia": {"lat": (25, 40), "lon": (100, 120)},
    "Australia": {"lat": (-35, -20), "lon": (135, 150)},
    "US_Midwest": {"lat": (35, 45), "lon": (-100, -85)},
    "US_Southwest": {"lat": (30, 40), "lon": (-120, -105)},
    "Southern_Europe": {"lat": (36, 44), "lon": (0, 20)},
}

# Niño 3.4 region for ENSO index
NINO34 = {"lat": (-5, 5), "lon": (-170, -120)}


def load_data(files, variable, start_year, end_year):
    """Load NetCDF data."""
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


def convert_lon_to_180(da):
    """Convert longitude from 0-360 to -180 to 180."""
    lon = da.lon.values
    lon_180 = np.where(lon > 180, lon - 360, lon)
    da = da.assign_coords(lon=lon_180)
    da = da.sortby("lon")
    return da


def extract_region(da, lat_bounds, lon_bounds):
    """Extract regional average time series."""
    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds

    region = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    # Area-weighted mean
    weights = np.cos(np.deg2rad(region.lat))
    regional_mean = region.weighted(weights).mean(dim=["lat", "lon"])

    return regional_mean


def compute_compound_index(tas_region, pr_region, tas_thresh, pr_thresh):
    """
    Compute compound event index for a region.
    Returns fraction of grid cells in compound state each month.
    """
    # For regional time series, just return binary compound indicator
    is_hot = tas_region > tas_thresh
    is_dry = pr_region < pr_thresh
    compound = is_hot & is_dry
    return compound.astype(float)


def compute_regional_compound_timeseries(tas, pr, tas_base, pr_base, regions):
    """
    Compute monthly compound event indicator for each region.
    """
    # Convert to -180 to 180 longitude
    tas = convert_lon_to_180(tas)
    pr = convert_lon_to_180(pr)
    tas_base = convert_lon_to_180(tas_base)
    pr_base = convert_lon_to_180(pr_base)

    regional_series = {}

    for region_name, bounds in regions.items():
        # Extract regional time series
        tas_region = extract_region(tas, bounds["lat"], bounds["lon"])
        pr_region = extract_region(pr, bounds["lat"], bounds["lon"])

        # Compute thresholds from baseline
        tas_base_region = extract_region(tas_base, bounds["lat"], bounds["lon"])
        pr_base_region = extract_region(pr_base, bounds["lat"], bounds["lon"])

        # Monthly thresholds
        tas_thresh = tas_base_region.groupby("time.month").quantile(
            HEAT_PERCENTILE / 100.0, dim="time"
        )
        pr_thresh = pr_base_region.groupby("time.month").quantile(
            DROUGHT_PERCENTILE / 100.0, dim="time"
        )

        # Align thresholds
        tas_th = tas_thresh.sel(month=tas_region.time.dt.month).drop_vars("month")
        pr_th = pr_thresh.sel(month=pr_region.time.dt.month).drop_vars("month")

        # Compute compound indicator
        compound = ((tas_region > tas_th) & (pr_region < pr_th)).astype(float)

        regional_series[region_name] = compound.compute()

    return regional_series


def compute_cross_correlations(regional_series):
    """
    Compute cross-correlation matrix between all region pairs.
    """
    regions = list(regional_series.keys())
    n_regions = len(regions)

    corr_matrix = np.zeros((n_regions, n_regions))
    pval_matrix = np.zeros((n_regions, n_regions))

    for i, reg1 in enumerate(regions):
        for j, reg2 in enumerate(regions):
            series1 = regional_series[reg1].values
            series2 = regional_series[reg2].values

            # Remove NaN
            mask = ~(np.isnan(series1) | np.isnan(series2))
            if mask.sum() > 10:
                corr, pval = stats.pearsonr(series1[mask], series2[mask])
                corr_matrix[i, j] = corr
                pval_matrix[i, j] = pval
            else:
                corr_matrix[i, j] = np.nan
                pval_matrix[i, j] = np.nan

    return corr_matrix, pval_matrix, regions


def compute_enso_correlation(regional_series, sst_data, nino_bounds):
    """
    Compute correlation between regional compound events and ENSO (Niño 3.4 SST).
    """
    # Extract Niño 3.4 index
    sst_180 = convert_lon_to_180(sst_data)
    nino34_sst = extract_region(sst_180, nino_bounds["lat"], nino_bounds["lon"])

    # Compute anomaly (simple detrended anomaly)
    nino34_anom = nino34_sst.groupby("time.month") - nino34_sst.groupby("time.month").mean()

    enso_correlations = {}

    for region_name, compound_series in regional_series.items():
        # Align time periods
        common_times = np.intersect1d(compound_series.time.values, nino34_anom.time.values)

        if len(common_times) > 12:
            comp = compound_series.sel(time=common_times).values
            enso = nino34_anom.sel(time=common_times).values

            mask = ~(np.isnan(comp) | np.isnan(enso))
            if mask.sum() > 10:
                corr, pval = stats.pearsonr(comp[mask], enso[mask])
                enso_correlations[region_name] = {"correlation": corr, "pvalue": pval}

    return enso_correlations


def count_synchronized_events(regional_series, min_regions=3):
    """
    Count months where multiple regions simultaneously experience compound events.
    """
    regions = list(regional_series.keys())

    # Align all series to common time
    common_times = regional_series[regions[0]].time.values
    for reg in regions[1:]:
        common_times = np.intersect1d(common_times, regional_series[reg].time.values)

    # Stack all series
    n_times = len(common_times)
    compound_matrix = np.zeros((n_times, len(regions)))

    for i, reg in enumerate(regions):
        compound_matrix[:, i] = regional_series[reg].sel(time=common_times).values

    # Count simultaneous events
    regions_affected = compound_matrix.sum(axis=1)

    sync_counts = {}
    for n in range(min_regions, len(regions) + 1):
        sync_counts[n] = (regions_affected >= n).sum()

    # Identify which regions co-occur most often
    cooccurrence = np.zeros((len(regions), len(regions)))
    for i in range(len(regions)):
        for j in range(len(regions)):
            cooccurrence[i, j] = ((compound_matrix[:, i] == 1) &
                                   (compound_matrix[:, j] == 1)).sum()

    return sync_counts, regions_affected, cooccurrence, regions, common_times


def analyze_teleconnections(model=MODEL):
    """Main teleconnection analysis."""
    print(f"\n{'='*70}")
    print(f"Teleconnection Analysis: {model}")
    print(f"{'='*70}")

    # Load data
    hist_tas = sorted((DATA_DIR / model / "historical" / "tas").glob("*.nc"))
    hist_pr = sorted((DATA_DIR / model / "historical" / "pr").glob("*.nc"))
    ssp_tas = sorted((DATA_DIR / model / "ssp585" / "tas").glob("*.nc"))
    ssp_pr = sorted((DATA_DIR / model / "ssp585" / "pr").glob("*.nc"))

    if not all([hist_tas, hist_pr, ssp_tas, ssp_pr]):
        print("Missing data files")
        return None

    results = {"model": model}

    # Load baseline for thresholds
    print("\nLoading baseline data...")
    tas_base = load_data(hist_tas, "tas", *PERIODS["baseline"])
    pr_base = load_data(hist_pr, "pr", *PERIODS["baseline"])

    if tas_base is None or pr_base is None:
        print("Could not load baseline")
        return None

    # Analyze historical period
    print("\nAnalyzing historical period (1950-2014)...")
    tas_hist = load_data(hist_tas, "tas", *PERIODS["historical"])
    pr_hist = load_data(hist_pr, "pr", *PERIODS["historical"])

    if tas_hist is not None and pr_hist is not None:
        print("  Computing regional compound time series...")
        hist_series = compute_regional_compound_timeseries(
            tas_hist, pr_hist, tas_base, pr_base, REGIONS
        )

        print("  Computing cross-correlations...")
        hist_corr, hist_pval, region_names = compute_cross_correlations(hist_series)

        print("  Counting synchronized events...")
        hist_sync, hist_affected, hist_cooccur, _, hist_times = count_synchronized_events(hist_series)

        results["historical"] = {
            "correlation_matrix": hist_corr,
            "pvalue_matrix": hist_pval,
            "synchronized_counts": hist_sync,
            "regions_affected_series": hist_affected,
            "cooccurrence_matrix": hist_cooccur,
            "region_names": region_names,
            "n_months": len(hist_times),
            "series": hist_series,
        }

        # Print summary
        print(f"\n  Historical synchronized events:")
        for n, count in hist_sync.items():
            pct = count / len(hist_times) * 100
            print(f"    {n}+ regions simultaneously: {count} months ({pct:.1f}%)")

    # Analyze future period
    print("\nAnalyzing future period (2050-2099)...")
    tas_future = load_data(ssp_tas, "tas", *PERIODS["future"])
    pr_future = load_data(ssp_pr, "pr", *PERIODS["future"])

    if tas_future is not None and pr_future is not None:
        print("  Computing regional compound time series...")
        future_series = compute_regional_compound_timeseries(
            tas_future, pr_future, tas_base, pr_base, REGIONS
        )

        print("  Computing cross-correlations...")
        future_corr, future_pval, _ = compute_cross_correlations(future_series)

        print("  Counting synchronized events...")
        future_sync, future_affected, future_cooccur, _, future_times = count_synchronized_events(future_series)

        results["future"] = {
            "correlation_matrix": future_corr,
            "pvalue_matrix": future_pval,
            "synchronized_counts": future_sync,
            "regions_affected_series": future_affected,
            "cooccurrence_matrix": future_cooccur,
            "n_months": len(future_times),
            "series": future_series,
        }

        print(f"\n  Future synchronized events:")
        for n, count in future_sync.items():
            pct = count / len(future_times) * 100
            print(f"    {n}+ regions simultaneously: {count} months ({pct:.1f}%)")

    return results


def plot_teleconnection_analysis(results):
    """Create comprehensive teleconnection visualizations."""

    fig = plt.figure(figsize=(18, 14))

    region_names = results["historical"]["region_names"]
    n_regions = len(region_names)

    # Clean up region names for display
    display_names = [r.replace("_", " ") for r in region_names]

    # Panel 1: Historical correlation matrix
    ax1 = fig.add_subplot(2, 3, 1)

    hist_corr = results["historical"]["correlation_matrix"]
    im1 = ax1.imshow(hist_corr, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    ax1.set_xticks(range(n_regions))
    ax1.set_yticks(range(n_regions))
    ax1.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
    ax1.set_yticklabels(display_names, fontsize=8)
    ax1.set_title("Historical (1950-2014)\nCompound Event Correlations", fontsize=11)
    plt.colorbar(im1, ax=ax1, label="Correlation", shrink=0.8)

    # Panel 2: Future correlation matrix
    ax2 = fig.add_subplot(2, 3, 2)

    future_corr = results["future"]["correlation_matrix"]
    im2 = ax2.imshow(future_corr, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    ax2.set_xticks(range(n_regions))
    ax2.set_yticks(range(n_regions))
    ax2.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(display_names, fontsize=8)
    ax2.set_title("Future (2050-2099)\nCompound Event Correlations", fontsize=11)
    plt.colorbar(im2, ax=ax2, label="Correlation", shrink=0.8)

    # Panel 3: Correlation change
    ax3 = fig.add_subplot(2, 3, 3)

    corr_change = future_corr - hist_corr
    im3 = ax3.imshow(corr_change, cmap="RdBu_r", vmin=-0.2, vmax=0.2)
    ax3.set_xticks(range(n_regions))
    ax3.set_yticks(range(n_regions))
    ax3.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
    ax3.set_yticklabels(display_names, fontsize=8)
    ax3.set_title("Change in Correlations\n(Future - Historical)", fontsize=11)
    plt.colorbar(im3, ax=ax3, label="Δ Correlation", shrink=0.8)

    # Panel 4: Synchronized event frequency
    ax4 = fig.add_subplot(2, 3, 4)

    hist_sync = results["historical"]["synchronized_counts"]
    future_sync = results["future"]["synchronized_counts"]
    hist_n = results["historical"]["n_months"]
    future_n = results["future"]["n_months"]

    x = list(hist_sync.keys())
    hist_pct = [hist_sync[n] / hist_n * 100 for n in x]
    future_pct = [future_sync[n] / future_n * 100 for n in x]

    bar_width = 0.35
    x_pos = np.arange(len(x))

    bars1 = ax4.bar(x_pos - bar_width/2, hist_pct, bar_width, label='Historical', color='#2166ac', alpha=0.8)
    bars2 = ax4.bar(x_pos + bar_width/2, future_pct, bar_width, label='Future', color='#b2182b', alpha=0.8)

    ax4.set_xlabel("Minimum Regions Affected Simultaneously")
    ax4.set_ylabel("Frequency (%)")
    ax4.set_title("Synchronized Compound Events\nMultiple Regions Simultaneously", fontsize=11)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"{n}+" for n in x])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add increase annotations
    for i, (h, f) in enumerate(zip(hist_pct, future_pct)):
        if h > 0:
            increase = ((f - h) / h) * 100
            ax4.annotate(f'+{increase:.0f}%', xy=(i + bar_width/2, f + 0.3),
                        ha='center', fontsize=9, fontweight='bold', color='#b2182b')

    # Panel 5: Co-occurrence heatmap (future)
    ax5 = fig.add_subplot(2, 3, 5)

    future_cooccur = results["future"]["cooccurrence_matrix"]
    # Normalize by diagonal (self-occurrence)
    diag = np.diag(future_cooccur)
    cooccur_pct = future_cooccur / diag[:, np.newaxis] * 100
    np.fill_diagonal(cooccur_pct, 0)  # Zero out diagonal for clarity

    im5 = ax5.imshow(cooccur_pct, cmap="YlOrRd", vmin=0, vmax=50)
    ax5.set_xticks(range(n_regions))
    ax5.set_yticks(range(n_regions))
    ax5.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
    ax5.set_yticklabels(display_names, fontsize=8)
    ax5.set_title("Co-occurrence Rate (Future)\n% of Region A events when Region B also affected", fontsize=10)
    plt.colorbar(im5, ax=ax5, label="Co-occurrence %", shrink=0.8)

    # Panel 6: Map showing regions
    ax6 = fig.add_subplot(2, 3, 6, projection=ccrs.Robinson())
    ax6.set_global()

    ax6.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
    ax6.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax6.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='--')

    # Color regions by their average compound frequency increase
    colors = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(REGIONS)))

    for i, (region_name, bounds) in enumerate(REGIONS.items()):
        lat_min, lat_max = bounds["lat"]
        lon_min, lon_max = bounds["lon"]

        # Draw rectangle
        ax6.plot([lon_min, lon_max, lon_max, lon_min, lon_min],
                [lat_min, lat_min, lat_max, lat_max, lat_min],
                color=colors[i], linewidth=2, transform=ccrs.PlateCarree())

        # Add label
        ax6.text((lon_min + lon_max)/2, (lat_min + lat_max)/2,
                region_name.replace("_", "\n"), fontsize=7,
                ha='center', va='center', transform=ccrs.PlateCarree(),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax6.set_title("Regions Analyzed for Teleconnections", fontsize=11)

    plt.tight_layout()

    output_path = ARTIFACTS_DIR / "teleconnection_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\nSaved: {output_path}")
    return output_path


def plot_synchronized_timeseries(results):
    """Plot time series of synchronized events."""

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Historical
    ax1 = axes[0]
    hist_affected = results["historical"]["regions_affected_series"]
    hist_n = results["historical"]["n_months"]

    # Create year-month index
    years_hist = np.arange(PERIODS["historical"][0], PERIODS["historical"][1] + 1)

    ax1.fill_between(range(len(hist_affected)), hist_affected, alpha=0.7, color='#2166ac')
    ax1.axhline(y=3, color='orange', linestyle='--', linewidth=2, label='3+ regions threshold')
    ax1.axhline(y=5, color='red', linestyle='--', linewidth=2, label='5+ regions threshold')
    ax1.set_ylabel("Regions Affected")
    ax1.set_title("Historical Period (1950-2014): Synchronized Compound Events", fontsize=12)
    ax1.set_ylim(0, 10)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Future
    ax2 = axes[1]
    future_affected = results["future"]["regions_affected_series"]

    ax2.fill_between(range(len(future_affected)), future_affected, alpha=0.7, color='#b2182b')
    ax2.axhline(y=3, color='orange', linestyle='--', linewidth=2, label='3+ regions threshold')
    ax2.axhline(y=5, color='red', linestyle='--', linewidth=2, label='5+ regions threshold')
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Regions Affected")
    ax2.set_title("Future Period (2050-2099): Synchronized Compound Events", fontsize=12)
    ax2.set_ylim(0, 10)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = ARTIFACTS_DIR / "teleconnection_timeseries.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def generate_summary(results):
    """Generate summary statistics."""

    hist_sync = results["historical"]["synchronized_counts"]
    future_sync = results["future"]["synchronized_counts"]
    hist_n = results["historical"]["n_months"]
    future_n = results["future"]["n_months"]

    hist_corr = results["historical"]["correlation_matrix"]
    future_corr = results["future"]["correlation_matrix"]

    # Get significant correlations (excluding diagonal)
    mask = ~np.eye(hist_corr.shape[0], dtype=bool)
    hist_mean_corr = np.nanmean(hist_corr[mask])
    future_mean_corr = np.nanmean(future_corr[mask])

    # Top correlated pairs
    region_names = results["historical"]["region_names"]
    n = len(region_names)

    pairs_hist = []
    pairs_future = []
    for i in range(n):
        for j in range(i+1, n):
            pairs_hist.append((region_names[i], region_names[j], hist_corr[i,j]))
            pairs_future.append((region_names[i], region_names[j], future_corr[i,j]))

    pairs_hist.sort(key=lambda x: abs(x[2]), reverse=True)
    pairs_future.sort(key=lambda x: abs(x[2]), reverse=True)

    summary = {
        "analysis": "Teleconnection Analysis - Synchronized Compound Events",
        "model": results["model"],
        "regions_analyzed": region_names,
        "findings": {
            "mean_correlation": {
                "historical": round(hist_mean_corr, 3),
                "future": round(future_mean_corr, 3),
                "change": round(future_mean_corr - hist_mean_corr, 3),
            },
            "synchronized_events": {
                "3_plus_regions": {
                    "historical_pct": round(hist_sync[3] / hist_n * 100, 2),
                    "future_pct": round(future_sync[3] / future_n * 100, 2),
                    "increase_pct": round((future_sync[3]/future_n - hist_sync[3]/hist_n) / (hist_sync[3]/hist_n) * 100, 1) if hist_sync[3] > 0 else None,
                },
                "5_plus_regions": {
                    "historical_pct": round(hist_sync.get(5, 0) / hist_n * 100, 2),
                    "future_pct": round(future_sync.get(5, 0) / future_n * 100, 2),
                },
            },
            "top_correlated_pairs_historical": [(p[0], p[1], round(p[2], 3)) for p in pairs_hist[:5]],
            "top_correlated_pairs_future": [(p[0], p[1], round(p[2], 3)) for p in pairs_future[:5]],
        }
    }

    return summary


def main():
    print("="*70)
    print("TELECONNECTION ANALYSIS")
    print("Synchronized Compound Events Across Regions")
    print("="*70)
    print("\nResearch Question: Do compound events in distant regions co-occur?")
    print("This matters for global food security and humanitarian response.")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run analysis
    results = analyze_teleconnections()

    if results is None:
        print("\nERROR: Analysis failed")
        return

    # Generate visualizations
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)

    plot_teleconnection_analysis(results)
    plot_synchronized_timeseries(results)

    # Generate summary
    print("\n" + "="*70)
    print("TELECONNECTION SUMMARY")
    print("="*70)

    summary = generate_summary(results)

    print(f"\nRegions analyzed: {len(summary['regions_analyzed'])}")
    print(f"\nMean inter-regional correlation:")
    print(f"  Historical: {summary['findings']['mean_correlation']['historical']:.3f}")
    print(f"  Future: {summary['findings']['mean_correlation']['future']:.3f}")
    print(f"  Change: {summary['findings']['mean_correlation']['change']:+.3f}")

    print(f"\nSynchronized events (3+ regions simultaneously):")
    sync_3 = summary['findings']['synchronized_events']['3_plus_regions']
    print(f"  Historical: {sync_3['historical_pct']:.1f}%")
    print(f"  Future: {sync_3['future_pct']:.1f}%")
    if sync_3['increase_pct']:
        print(f"  Increase: +{sync_3['increase_pct']:.0f}%")

    print(f"\nSynchronized events (5+ regions simultaneously):")
    sync_5 = summary['findings']['synchronized_events']['5_plus_regions']
    print(f"  Historical: {sync_5['historical_pct']:.1f}%")
    print(f"  Future: {sync_5['future_pct']:.1f}%")

    print(f"\nTop correlated region pairs (future):")
    for r1, r2, corr in summary['findings']['top_correlated_pairs_future'][:5]:
        print(f"  {r1} <-> {r2}: {corr:+.3f}")

    print(f"\n{'='*70}")
    key_msg = f"Synchronized 3+ region events increase from {sync_3['historical_pct']:.1f}% to {sync_3['future_pct']:.1f}%"
    print(f"KEY FINDING: {key_msg}")
    print(f"{'='*70}")

    # Save summary
    summary_path = ARTIFACTS_DIR / "teleconnection_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_path}")

    return results, summary


if __name__ == "__main__":
    main()
