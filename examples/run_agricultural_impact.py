#!/usr/bin/env python3
"""
Agricultural Impact Analysis: Climate Change Effects on Crop Production

Analyzes how climate change affects key agricultural metrics:
1. Growing Degree Days (GDD) - accumulated heat for crop development
2. Growing Season Length - frost-free period
3. Heat Stress - damaging high temperatures
4. Drought Stress - precipitation deficits during growing season
5. Compound Stress - simultaneous heat and drought

Uses CMIP6 temperature (tas) and precipitation (pr) data from ESGF.
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

# Configuration
DATA_DIR = Path("data/compound_hazard")  # Reuse existing data
ARTIFACTS_DIR = Path("Artifacts")

MODEL = "GFDL-ESM4"

PERIODS = {
    "historical": (1995, 2014),
    "mid_century": (2040, 2069),
    "end_century": (2070, 2099),
}

# Agricultural thresholds (for monthly mean data)
GDD_BASE = 10  # Base temperature for GDD calculation (°C)
HEAT_STRESS_THRESHOLD = 30  # Monthly mean > 30°C indicates frequent heat stress
SEVERE_HEAT_THRESHOLD = 35  # Monthly mean > 35°C indicates severe/lethal heat
FROST_THRESHOLD = 5  # Monthly mean < 5°C indicates frost risk (conservative for monthly data)
DROUGHT_PERCENTILE = 20  # Below 20th percentile = drought

# Key agricultural regions
AG_REGIONS = {
    "US_Midwest": {
        "lat": (35, 50), "lon": (-105, -80),
        "title": "US Corn Belt",
        "crops": "Corn, Soybeans",
        "growing_season": (4, 10),  # April-October
    },
    "Europe": {
        "lat": (43, 55), "lon": (-5, 30),
        "title": "European Breadbasket",
        "crops": "Wheat, Barley",
        "growing_season": (3, 9),  # March-September
    },
    "South_Asia": {
        "lat": (20, 35), "lon": (70, 90),
        "title": "Indo-Gangetic Plain",
        "crops": "Rice, Wheat",
        "growing_season": (6, 11),  # June-November (monsoon crops)
    },
    "Sahel": {
        "lat": (10, 18), "lon": (-15, 35),
        "title": "African Sahel",
        "crops": "Millet, Sorghum",
        "growing_season": (6, 10),  # June-October (wet season)
    },
    "Brazil": {
        "lat": (-30, -15), "lon": (-60, -45),
        "title": "Brazilian Cerrado",
        "crops": "Soybeans, Corn",
        "growing_season": (10, 3),  # October-March (Southern Hemisphere summer)
    },
    "Australia": {
        "lat": (-38, -28), "lon": (135, 150),
        "title": "Australian Wheat Belt",
        "crops": "Wheat",
        "growing_season": (5, 11),  # May-November
    },
}


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


def convert_lon_to_180(da):
    """Convert longitude from 0-360 to -180 to 180."""
    lon = da.lon.values
    lon_180 = np.where(lon > 180, lon - 360, lon)
    da = da.assign_coords(lon=lon_180)
    da = da.sortby("lon")
    return da


def calculate_gdd(tas_celsius, base_temp=GDD_BASE):
    """
    Calculate Growing Degree Days from monthly temperature.

    GDD = sum of (T - base_temp) for months where T > base_temp
    For monthly data, we multiply by ~30 days per month.
    """
    # Only count positive contributions
    monthly_gdd = xr.where(tas_celsius > base_temp, (tas_celsius - base_temp) * 30, 0)
    annual_gdd = monthly_gdd.groupby("time.year").sum(dim="time")
    return annual_gdd


def calculate_growing_season_length(tas_celsius, frost_threshold=FROST_THRESHOLD):
    """
    Calculate growing season length (months with mean T > frost threshold).

    For monthly data, count months above threshold.
    """
    frost_free = tas_celsius > frost_threshold
    annual_months = frost_free.groupby("time.year").sum(dim="time")
    return annual_months


def calculate_heat_stress_months(tas_celsius, threshold=HEAT_STRESS_THRESHOLD):
    """
    Calculate number of months with heat stress (mean T > threshold).
    """
    heat_stress = tas_celsius > threshold
    annual_months = heat_stress.groupby("time.year").sum(dim="time")
    return annual_months


def calculate_drought_months(pr, baseline_pr, percentile=DROUGHT_PERCENTILE):
    """
    Calculate drought months (precipitation below percentile threshold).
    """
    # Calculate threshold from baseline
    threshold = baseline_pr.groupby("time.month").quantile(percentile / 100, dim="time")

    # Compare to threshold by month
    pr_months = pr.groupby("time.month")
    drought = pr.groupby("time.month") < threshold

    # Count annual drought months
    annual_drought = drought.groupby("time.year").sum(dim="time")
    return annual_drought


def extract_growing_season(da, start_month, end_month):
    """Extract data for growing season months."""
    if start_month <= end_month:
        # Normal case (e.g., April-October)
        mask = (da.time.dt.month >= start_month) & (da.time.dt.month <= end_month)
    else:
        # Crosses year boundary (e.g., October-March for Southern Hemisphere)
        mask = (da.time.dt.month >= start_month) | (da.time.dt.month <= end_month)
    return da.where(mask, drop=True)


def analyze_agricultural_impacts(model=MODEL):
    """Run agricultural impact analysis."""
    print(f"\n{'='*70}")
    print(f"AGRICULTURAL IMPACT ANALYSIS: {model}")
    print(f"{'='*70}")

    # Load data paths
    hist_tas = sorted((DATA_DIR / model / "historical" / "tas").glob("*.nc"))
    hist_pr = sorted((DATA_DIR / model / "historical" / "pr").glob("*.nc"))
    ssp585_tas = sorted((DATA_DIR / model / "ssp585" / "tas").glob("*.nc"))
    ssp585_pr = sorted((DATA_DIR / model / "ssp585" / "pr").glob("*.nc"))
    ssp126_tas = sorted((DATA_DIR / model / "ssp126" / "tas").glob("*.nc"))
    ssp126_pr = sorted((DATA_DIR / model / "ssp126" / "pr").glob("*.nc"))

    if not all([hist_tas, hist_pr, ssp585_tas, ssp585_pr]):
        print("Missing required data files")
        return None

    results = {}

    # Load baseline precipitation for drought calculation
    print("\nLoading baseline data for thresholds...")
    pr_baseline = load_data(hist_pr, "pr", 1985, 2014)

    # Process each scenario
    scenarios = {
        "historical": (hist_tas, hist_pr, PERIODS["historical"]),
        "end_century_ssp585": (ssp585_tas, ssp585_pr, PERIODS["end_century"]),
    }

    if ssp126_tas and ssp126_pr:
        scenarios["end_century_ssp126"] = (ssp126_tas, ssp126_pr, PERIODS["end_century"])

    for scenario_name, (tas_files, pr_files, (start, end)) in scenarios.items():
        print(f"\nProcessing {scenario_name} ({start}-{end})...")

        tas = load_data(tas_files, "tas", start, end)
        pr = load_data(pr_files, "pr", start, end)

        if tas is None or pr is None:
            print(f"  Could not load data for {scenario_name}")
            continue

        # Convert temperature to Celsius
        tas_c = tas - 273.15

        # Align grids if needed
        if tas.lat.size != pr.lat.size:
            pr = pr.interp(lat=tas.lat, lon=tas.lon)
        if pr_baseline is not None and tas.lat.size != pr_baseline.lat.size:
            pr_baseline_aligned = pr_baseline.interp(lat=tas.lat, lon=tas.lon)
        else:
            pr_baseline_aligned = pr_baseline

        print("  Calculating Growing Degree Days...")
        gdd = calculate_gdd(tas_c)
        mean_gdd = gdd.mean(dim="year")

        print("  Calculating growing season length...")
        gsl = calculate_growing_season_length(tas_c)
        mean_gsl = gsl.mean(dim="year")

        print("  Calculating heat stress months...")
        heat_stress = calculate_heat_stress_months(tas_c)
        mean_heat_stress = heat_stress.mean(dim="year")

        severe_heat = calculate_heat_stress_months(tas_c, SEVERE_HEAT_THRESHOLD)
        mean_severe_heat = severe_heat.mean(dim="year")

        print("  Calculating drought months...")
        if pr_baseline_aligned is not None:
            drought = calculate_drought_months(pr, pr_baseline_aligned)
            mean_drought = drought.mean(dim="year")
        else:
            mean_drought = None

        # Calculate regional statistics
        regional_stats = {}
        for region_name, bounds in AG_REGIONS.items():
            lat_min, lat_max = bounds["lat"]
            lon_min, lon_max = bounds["lon"]

            # Handle longitude conversion for selection (data is in 0-360)
            lon_min_sel = lon_min if lon_min >= 0 else lon_min + 360
            lon_max_sel = lon_max if lon_max >= 0 else lon_max + 360

            try:
                # Select region - handle cases where selection crosses 0 meridian
                if lon_min_sel > lon_max_sel:
                    # Region crosses prime meridian (e.g., Europe: -5 to 30 -> 355 to 30)
                    region_gdd = xr.concat([
                        mean_gdd.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_sel, 360)),
                        mean_gdd.sel(lat=slice(lat_min, lat_max), lon=slice(0, lon_max_sel))
                    ], dim="lon")
                    region_gsl = xr.concat([
                        mean_gsl.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_sel, 360)),
                        mean_gsl.sel(lat=slice(lat_min, lat_max), lon=slice(0, lon_max_sel))
                    ], dim="lon")
                    region_heat = xr.concat([
                        mean_heat_stress.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_sel, 360)),
                        mean_heat_stress.sel(lat=slice(lat_min, lat_max), lon=slice(0, lon_max_sel))
                    ], dim="lon")
                    region_severe = xr.concat([
                        mean_severe_heat.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_sel, 360)),
                        mean_severe_heat.sel(lat=slice(lat_min, lat_max), lon=slice(0, lon_max_sel))
                    ], dim="lon")
                else:
                    region_gdd = mean_gdd.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_sel, lon_max_sel))
                    region_gsl = mean_gsl.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_sel, lon_max_sel))
                    region_heat = mean_heat_stress.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_sel, lon_max_sel))
                    region_severe = mean_severe_heat.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_sel, lon_max_sel))

                weights = np.cos(np.deg2rad(region_gdd.lat))

                regional_stats[region_name] = {
                    "gdd": float(region_gdd.weighted(weights).mean().values),
                    "growing_season_months": float(region_gsl.weighted(weights).mean().values),
                    "heat_stress_months": float(region_heat.weighted(weights).mean().values),
                    "severe_heat_months": float(region_severe.weighted(weights).mean().values),
                }

                if mean_drought is not None:
                    region_drought = mean_drought.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_sel, lon_max_sel))
                    regional_stats[region_name]["drought_months"] = float(region_drought.weighted(weights).mean().values)

            except Exception as e:
                print(f"  Error with region {region_name}: {e}")
                continue

        # Global statistics
        weights = np.cos(np.deg2rad(mean_gdd.lat))

        results[scenario_name] = {
            "gdd": mean_gdd,
            "growing_season": mean_gsl,
            "heat_stress": mean_heat_stress,
            "severe_heat": mean_severe_heat,
            "drought": mean_drought,
            "global": {
                "gdd": float(mean_gdd.weighted(weights).mean().values),
                "growing_season_months": float(mean_gsl.weighted(weights).mean().values),
                "heat_stress_months": float(mean_heat_stress.weighted(weights).mean().values),
                "severe_heat_months": float(mean_severe_heat.weighted(weights).mean().values),
            },
            "regional": regional_stats
        }

        print(f"  Global mean GDD: {results[scenario_name]['global']['gdd']:.0f}")
        print(f"  Global mean growing season: {results[scenario_name]['global']['growing_season_months']:.1f} months")
        print(f"  Global mean heat stress months: {results[scenario_name]['global']['heat_stress_months']:.2f}")

    return results


def plot_agricultural_results(results):
    """Create agricultural impact visualizations."""
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*70}")

    if "historical" not in results or "end_century_ssp585" not in results:
        print("Missing required scenarios for plotting")
        return

    # Figure 1: Global maps
    fig = plt.figure(figsize=(16, 12))

    # GDD change
    ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.Robinson())
    ax1.set_global()

    hist_gdd = convert_lon_to_180(results["historical"]["gdd"])
    future_gdd = convert_lon_to_180(results["end_century_ssp585"]["gdd"])
    gdd_change = future_gdd - hist_gdd

    im1 = ax1.pcolormesh(
        gdd_change.lon, gdd_change.lat, gdd_change,
        cmap="RdYlGn", vmin=-500, vmax=1500,
        transform=ccrs.PlateCarree(),
        shading="auto"
    )
    ax1.add_feature(cfeature.OCEAN, facecolor='white', zorder=2)
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--", zorder=3)
    ax1.set_title("Change in Growing Degree Days\n(SSP5-8.5 minus Historical)", fontsize=11)
    plt.colorbar(im1, ax=ax1, orientation="horizontal", pad=0.05, shrink=0.6, label="Δ GDD (°C·days)")

    # Heat stress change
    ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.Robinson())
    ax2.set_global()

    hist_heat = convert_lon_to_180(results["historical"]["heat_stress"])
    future_heat = convert_lon_to_180(results["end_century_ssp585"]["heat_stress"])
    heat_change = future_heat - hist_heat

    im2 = ax2.pcolormesh(
        heat_change.lon, heat_change.lat, heat_change,
        cmap="YlOrRd", vmin=0, vmax=4,
        transform=ccrs.PlateCarree(),
        shading="auto"
    )
    ax2.add_feature(cfeature.OCEAN, facecolor='white', zorder=2)
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--", zorder=3)
    ax2.set_title(f"Change in Heat Stress Months\n(Monthly mean > {HEAT_STRESS_THRESHOLD}°C)", fontsize=11)
    plt.colorbar(im2, ax=ax2, orientation="horizontal", pad=0.05, shrink=0.6, label="Δ months/year")

    # Growing season change
    ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.Robinson())
    ax3.set_global()

    hist_gsl = convert_lon_to_180(results["historical"]["growing_season"])
    future_gsl = convert_lon_to_180(results["end_century_ssp585"]["growing_season"])
    gsl_change = future_gsl - hist_gsl

    im3 = ax3.pcolormesh(
        gsl_change.lon, gsl_change.lat, gsl_change,
        cmap="RdYlGn", vmin=-2, vmax=4,
        transform=ccrs.PlateCarree(),
        shading="auto"
    )
    ax3.add_feature(cfeature.OCEAN, facecolor='white', zorder=2)
    ax3.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax3.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--", zorder=3)
    ax3.set_title(f"Change in Growing Season Length\n(Months > {FROST_THRESHOLD}°C)", fontsize=11)
    plt.colorbar(im3, ax=ax3, orientation="horizontal", pad=0.05, shrink=0.6, label="Δ months")

    # Severe heat (future absolute)
    ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.Robinson())
    ax4.set_global()

    future_severe = convert_lon_to_180(results["end_century_ssp585"]["severe_heat"])

    im4 = ax4.pcolormesh(
        future_severe.lon, future_severe.lat, future_severe,
        cmap="hot_r", vmin=0, vmax=6,
        transform=ccrs.PlateCarree(),
        shading="auto"
    )
    ax4.add_feature(cfeature.OCEAN, facecolor='white', zorder=2)
    ax4.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax4.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--", zorder=3)
    ax4.set_title(f"End-Century Severe Heat Months\n(Monthly mean > {SEVERE_HEAT_THRESHOLD}°C)", fontsize=11)
    plt.colorbar(im4, ax=ax4, orientation="horizontal", pad=0.05, shrink=0.6, label="months/year")

    plt.suptitle("Agricultural Climate Impacts\nGFDL-ESM4 CMIP6 (Land Only)", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = ARTIFACTS_DIR / "agricultural_global.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")

    # Figure 2: Regional bar charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    regions = list(AG_REGIONS.keys())
    x = np.arange(len(regions))
    width = 0.35

    # Panel 1: GDD change
    ax = axes[0, 0]
    hist_vals = [results["historical"]["regional"].get(r, {}).get("gdd", 0) for r in regions]
    future_vals = [results["end_century_ssp585"]["regional"].get(r, {}).get("gdd", 0) for r in regions]

    ax.bar(x - width/2, hist_vals, width, label="Historical", color="#2166ac", alpha=0.8)
    ax.bar(x + width/2, future_vals, width, label="End-Century SSP5-8.5", color="#b2182b", alpha=0.8)

    ax.set_ylabel("Growing Degree Days (°C·days)")
    ax.set_title("Annual Growing Degree Days by Region", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([AG_REGIONS[r]["title"] for r in regions], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Heat stress months
    ax = axes[0, 1]
    hist_vals = [results["historical"]["regional"].get(r, {}).get("heat_stress_months", 0) for r in regions]
    future_vals = [results["end_century_ssp585"]["regional"].get(r, {}).get("heat_stress_months", 0) for r in regions]

    ax.bar(x - width/2, hist_vals, width, label="Historical", color="#2166ac", alpha=0.8)
    ax.bar(x + width/2, future_vals, width, label="End-Century SSP5-8.5", color="#b2182b", alpha=0.8)

    ax.set_ylabel(f"Months with mean T > {HEAT_STRESS_THRESHOLD}°C")
    ax.set_title("Heat Stress Months by Region", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([AG_REGIONS[r]["title"] for r in regions], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Growing season length
    ax = axes[1, 0]
    hist_vals = [results["historical"]["regional"].get(r, {}).get("growing_season_months", 0) for r in regions]
    future_vals = [results["end_century_ssp585"]["regional"].get(r, {}).get("growing_season_months", 0) for r in regions]

    ax.bar(x - width/2, hist_vals, width, label="Historical", color="#2166ac", alpha=0.8)
    ax.bar(x + width/2, future_vals, width, label="End-Century SSP5-8.5", color="#b2182b", alpha=0.8)

    ax.set_ylabel("Frost-free months")
    ax.set_title("Growing Season Length by Region", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([AG_REGIONS[r]["title"] for r in regions], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 12)

    # Panel 4: Severe heat months
    ax = axes[1, 1]
    hist_vals = [results["historical"]["regional"].get(r, {}).get("severe_heat_months", 0) for r in regions]
    future_vals = [results["end_century_ssp585"]["regional"].get(r, {}).get("severe_heat_months", 0) for r in regions]

    ax.bar(x - width/2, hist_vals, width, label="Historical", color="#2166ac", alpha=0.8)
    ax.bar(x + width/2, future_vals, width, label="End-Century SSP5-8.5", color="#b2182b", alpha=0.8)

    ax.set_ylabel(f"Months with mean T > {SEVERE_HEAT_THRESHOLD}°C")
    ax.set_title("Severe/Lethal Heat Months by Region", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([AG_REGIONS[r]["title"] for r in regions], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    output_path = ARTIFACTS_DIR / "agricultural_regional.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")

    # Figure 3: Winners and Losers summary
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate net impact score for each region
    # Positive = benefits (more GDD, longer season)
    # Negative = costs (more heat stress)

    impact_data = []
    for r in regions:
        hist = results["historical"]["regional"].get(r, {})
        future = results["end_century_ssp585"]["regional"].get(r, {})

        gdd_change = future.get("gdd", 0) - hist.get("gdd", 0)
        gsl_change = future.get("growing_season_months", 0) - hist.get("growing_season_months", 0)
        heat_change = future.get("heat_stress_months", 0) - hist.get("heat_stress_months", 0)
        severe_change = future.get("severe_heat_months", 0) - hist.get("severe_heat_months", 0)

        impact_data.append({
            "region": AG_REGIONS[r]["title"],
            "gdd_change": gdd_change,
            "gsl_change": gsl_change,
            "heat_change": heat_change,
            "severe_change": severe_change,
        })

    # Create stacked bar showing changes
    x = np.arange(len(regions))

    gdd_changes = [d["gdd_change"] / 100 for d in impact_data]  # Scale for visibility
    gsl_changes = [d["gsl_change"] * 50 for d in impact_data]  # Scale up
    heat_changes = [-d["heat_change"] * 100 for d in impact_data]  # Negative = bad
    severe_changes = [-d["severe_change"] * 200 for d in impact_data]  # More negative = worse

    ax.bar(x, gdd_changes, label="GDD increase (÷100)", color="#4daf4a", alpha=0.8)
    ax.bar(x, gsl_changes, bottom=gdd_changes, label="Season length (×50)", color="#377eb8", alpha=0.8)
    ax.bar(x, heat_changes, label="Heat stress (×-100)", color="#ff7f00", alpha=0.8)
    ax.bar(x, severe_changes, bottom=heat_changes, label="Severe heat (×-200)", color="#e41a1c", alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel("Impact Score (scaled)")
    ax.set_title("Agricultural Climate Winners and Losers\n(Green = benefits, Red = damages)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([AG_REGIONS[r]["title"] for r in regions], rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    output_path = ARTIFACTS_DIR / "agricultural_winners_losers.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def save_summary(results):
    """Save results summary to JSON."""
    summary = {
        "thresholds": {
            "gdd_base_temp": GDD_BASE,
            "heat_stress_threshold": HEAT_STRESS_THRESHOLD,
            "severe_heat_threshold": SEVERE_HEAT_THRESHOLD,
            "frost_threshold": FROST_THRESHOLD,
        },
        "regions": {k: {"title": v["title"], "crops": v["crops"]} for k, v in AG_REGIONS.items()},
        "global_results": {},
        "regional_results": {}
    }

    for scenario, data in results.items():
        summary["global_results"][scenario] = data["global"]
        summary["regional_results"][scenario] = data["regional"]

    output_path = ARTIFACTS_DIR / "agricultural_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {output_path}")

    return summary


def main():
    print("=" * 70)
    print("AGRICULTURAL IMPACT ANALYSIS")
    print("Climate Change Effects on Crop Production")
    print("=" * 70)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run analysis
    results = analyze_agricultural_impacts()

    if results:
        # Create visualizations
        plot_agricultural_results(results)

        # Save summary
        summary = save_summary(results)

        # Print summary
        print("\n" + "=" * 70)
        print("AGRICULTURAL IMPACT SUMMARY")
        print("=" * 70)

        if "historical" in results and "end_century_ssp585" in results:
            hist = results["historical"]
            future = results["end_century_ssp585"]

            print("\nGlobal Changes:")
            print(f"  GDD: {hist['global']['gdd']:.0f} → {future['global']['gdd']:.0f} "
                  f"(+{future['global']['gdd'] - hist['global']['gdd']:.0f})")
            print(f"  Growing season: {hist['global']['growing_season_months']:.1f} → "
                  f"{future['global']['growing_season_months']:.1f} months")
            print(f"  Heat stress months: {hist['global']['heat_stress_months']:.2f} → "
                  f"{future['global']['heat_stress_months']:.2f}")

            print("\nRegional Results:")
            for region in AG_REGIONS:
                if region in hist["regional"] and region in future["regional"]:
                    h = hist["regional"][region]
                    f = future["regional"][region]
                    print(f"\n  {AG_REGIONS[region]['title']} ({AG_REGIONS[region]['crops']}):")
                    print(f"    GDD: {h['gdd']:.0f} → {f['gdd']:.0f} ({f['gdd']-h['gdd']:+.0f})")
                    print(f"    Heat stress: {h['heat_stress_months']:.1f} → {f['heat_stress_months']:.1f} months")
                    print(f"    Severe heat: {h['severe_heat_months']:.1f} → {f['severe_heat_months']:.1f} months")

    print("\n" + "=" * 70)
    print("Agricultural impact analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
