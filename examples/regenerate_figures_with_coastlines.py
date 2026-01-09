#!/usr/bin/env python3
"""
Regenerate all regional figures with proper coastlines and correct extent.
Fixes longitude wrapping issues and adds geographic context.
"""

import warnings
warnings.filterwarnings("ignore")

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

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


def get_compound_frequency_change(model):
    """Get compound frequency change data for a model."""
    hist_tas = sorted((DATA_DIR / model / "historical" / "tas").glob("*.nc"))
    hist_pr = sorted((DATA_DIR / model / "historical" / "pr").glob("*.nc"))
    ssp_tas = sorted((DATA_DIR / model / "ssp585" / "tas").glob("*.nc"))
    ssp_pr = sorted((DATA_DIR / model / "ssp585" / "pr").glob("*.nc"))

    if not all([hist_tas, hist_pr, ssp_tas, ssp_pr]):
        return None, None

    # Load baseline
    tas_base = load_data(hist_tas, "tas", *PERIODS["baseline"])
    pr_base = load_data(hist_pr, "pr", *PERIODS["baseline"])

    if tas_base is None or pr_base is None:
        return None, None

    # Compute thresholds
    tas_thresh = tas_base.groupby("time.month").quantile(HEAT_PERCENTILE / 100.0, dim="time")
    pr_thresh = pr_base.groupby("time.month").quantile(DROUGHT_PERCENTILE / 100.0, dim="time")

    results = {}
    for period_name in ["historical", "end_century"]:
        start, end = PERIODS[period_name]
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

        tas_th = tas_thresh.sel(month=tas.time.dt.month).drop_vars("month")
        pr_th = pr_thresh.sel(month=pr.time.dt.month).drop_vars("month")

        compound = (tas > tas_th) & (pr < pr_th)
        freq = compound.mean(dim="time").compute()
        results[period_name] = freq

    if "historical" in results and "end_century" in results:
        change = (results["end_century"] - results["historical"]) * 100
        return change, model

    return None, None


def convert_lon_to_180(da):
    """Convert longitude from 0-360 to -180 to 180."""
    lon = da.lon.values
    lon_180 = np.where(lon > 180, lon - 360, lon)
    da = da.assign_coords(lon=lon_180)
    da = da.sortby("lon")
    return da


def plot_regional_compound_risk_fixed():
    """Create regional analysis with proper coastlines and extents."""

    print("Loading compound frequency data...")
    change, model = get_compound_frequency_change("GFDL-ESM4")

    if change is None:
        print("Could not load data")
        return

    # Convert to -180 to 180 longitude
    change = convert_lon_to_180(change)

    # Define regions with proper lon/lat bounds (using -180 to 180)
    regions = {
        "Mediterranean": {"lat": (30, 46), "lon": (-10, 36), "title": "Mediterranean Basin"},
        "US Southwest": {"lat": (25, 42), "lon": (-125, -100), "title": "US Southwest"},
        "Amazon": {"lat": (-20, 10), "lon": (-80, -40), "title": "Amazon Basin"},
        "Australia": {"lat": (-45, -10), "lon": (110, 155), "title": "Australia"},
        "Southern Africa": {"lat": (-35, -10), "lon": (10, 45), "title": "Southern Africa"},
        "South Asia": {"lat": (5, 35), "lon": (65, 100), "title": "South Asia"},
    }

    fig = plt.figure(figsize=(18, 11))

    for idx, (region_name, bounds) in enumerate(regions.items()):
        ax = fig.add_subplot(2, 3, idx + 1, projection=ccrs.PlateCarree())

        lat_min, lat_max = bounds["lat"]
        lon_min, lon_max = bounds["lon"]

        # Extract region
        try:
            region_data = change.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max)
            )
        except Exception as e:
            print(f"  Error extracting {region_name}: {e}")
            continue

        # Plot data
        im = ax.pcolormesh(
            region_data.lon, region_data.lat, region_data,
            cmap="YlOrRd", vmin=0, vmax=25,
            transform=ccrs.PlateCarree(),
            shading="auto"
        )

        # Set extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        # Add geographic features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', edgecolor='gray')
        ax.add_feature(cfeature.LAKES, alpha=0.3, facecolor='lightblue')
        ax.add_feature(cfeature.RIVERS, linewidth=0.3, edgecolor='lightblue')

        # Calculate statistics
        weights = np.cos(np.deg2rad(region_data.lat))
        mean_change = float(region_data.weighted(weights).mean().values)
        max_change = float(region_data.max().values)

        ax.set_title(f"{bounds['title']}\nMean: +{mean_change:.1f}pp, Max: +{max_change:.1f}pp", fontsize=11)

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        plt.colorbar(im, ax=ax, label="Δ Frequency (pp)", shrink=0.7, pad=0.02)

    fig.suptitle(
        f"Regional Compound Heat-Drought Risk Hotspots\n{model} SSP5-8.5: Change from Historical (1995-2014) to End-Century (2070-2099)",
        fontsize=14, y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = ARTIFACTS_DIR / "compound_hazard_regional.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def plot_mitigation_regional_fixed():
    """Create regional mitigation benefit maps with coastlines."""

    print("\nLoading mitigation data...")

    model = "GFDL-ESM4"

    # Load SSP1-2.6 and SSP5-8.5 end-century data
    hist_tas = sorted((DATA_DIR / model / "historical" / "tas").glob("*.nc"))
    hist_pr = sorted((DATA_DIR / model / "historical" / "pr").glob("*.nc"))
    ssp126_tas = sorted((DATA_DIR / model / "ssp126" / "tas").glob("*.nc"))
    ssp126_pr = sorted((DATA_DIR / model / "ssp126" / "pr").glob("*.nc"))
    ssp585_tas = sorted((DATA_DIR / model / "ssp585" / "tas").glob("*.nc"))
    ssp585_pr = sorted((DATA_DIR / model / "ssp585" / "pr").glob("*.nc"))

    if not all([hist_tas, hist_pr, ssp126_tas, ssp126_pr, ssp585_tas, ssp585_pr]):
        print("Missing data files")
        return

    # Compute thresholds from baseline
    tas_base = load_data(hist_tas, "tas", *PERIODS["baseline"])
    pr_base = load_data(hist_pr, "pr", *PERIODS["baseline"])

    tas_thresh = tas_base.groupby("time.month").quantile(HEAT_PERCENTILE / 100.0, dim="time")
    pr_thresh = pr_base.groupby("time.month").quantile(DROUGHT_PERCENTILE / 100.0, dim="time")

    def get_compound_freq(tas_files, pr_files, start, end):
        tas = load_data(tas_files, "tas", start, end)
        pr = load_data(pr_files, "pr", start, end)
        if tas is None or pr is None:
            return None
        if tas.lat.size != pr.lat.size:
            pr = pr.interp(lat=tas.lat, lon=tas.lon)
        tas_th = tas_thresh.sel(month=tas.time.dt.month).drop_vars("month")
        pr_th = pr_thresh.sel(month=pr.time.dt.month).drop_vars("month")
        compound = (tas > tas_th) & (pr < pr_th)
        return compound.mean(dim="time").compute()

    ssp126_freq = get_compound_freq(ssp126_tas, ssp126_pr, *PERIODS["end_century"])
    ssp585_freq = get_compound_freq(ssp585_tas, ssp585_pr, *PERIODS["end_century"])

    if ssp126_freq is None or ssp585_freq is None:
        print("Could not compute frequencies")
        return

    avoided = (ssp585_freq - ssp126_freq) * 100

    # Convert longitude
    avoided = convert_lon_to_180(avoided)

    regions = {
        "Mediterranean": {"lat": (30, 46), "lon": (-10, 36), "title": "Mediterranean Basin"},
        "Amazon": {"lat": (-20, 10), "lon": (-80, -40), "title": "Amazon Basin"},
        "Southern Africa": {"lat": (-35, -10), "lon": (10, 45), "title": "Southern Africa"},
        "Australia": {"lat": (-45, -10), "lon": (110, 155), "title": "Australia"},
    }

    fig = plt.figure(figsize=(14, 10))

    for idx, (region_name, bounds) in enumerate(regions.items()):
        ax = fig.add_subplot(2, 2, idx + 1, projection=ccrs.PlateCarree())

        lat_min, lat_max = bounds["lat"]
        lon_min, lon_max = bounds["lon"]

        try:
            region_data = avoided.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max)
            )
        except Exception as e:
            print(f"  Error extracting {region_name}: {e}")
            continue

        im = ax.pcolormesh(
            region_data.lon, region_data.lat, region_data,
            cmap="Greens", vmin=0, vmax=25,
            transform=ccrs.PlateCarree(),
            shading="auto"
        )

        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', edgecolor='gray')
        ax.add_feature(cfeature.LAKES, alpha=0.3, facecolor='lightblue')

        weights = np.cos(np.deg2rad(region_data.lat))
        mean_avoided = float(region_data.weighted(weights).mean().values)

        ax.set_title(f"{bounds['title']}\nMean avoided risk: {mean_avoided:.1f} pp", fontsize=11)

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        plt.colorbar(im, ax=ax, label="Avoided (pp)", shrink=0.7, pad=0.02)

    fig.suptitle(
        f"Regional Mitigation Benefits: Avoided Compound Risk\n{model} End-Century (SSP5-8.5 → SSP1-2.6)",
        fontsize=13, y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = ARTIFACTS_DIR / "mitigation_regional.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def plot_global_map_with_coastlines():
    """Create improved global compound risk change map."""

    print("\nCreating global map...")

    change, model = get_compound_frequency_change("GFDL-ESM4")
    if change is None:
        return

    change = convert_lon_to_180(change)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    # Global extent
    ax.set_global()

    im = ax.pcolormesh(
        change.lon, change.lat, change,
        cmap="YlOrRd", vmin=0, vmax=20,
        transform=ccrs.PlateCarree(),
        shading="auto"
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='--', edgecolor='gray')

    ax.set_title(
        f"Global Change in Compound Heat-Drought Frequency\n{model} SSP5-8.5 (2070-2099 minus 1995-2014)",
        fontsize=13
    )

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar.set_label("Change in Compound Event Frequency (percentage points)", fontsize=11)

    output_path = ARTIFACTS_DIR / "compound_global_map.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def plot_seasonal_maps_with_coastlines():
    """Recreate seasonal maps with coastlines."""

    print("\nCreating seasonal maps...")

    model = "GFDL-ESM4"

    hist_tas = sorted((DATA_DIR / model / "historical" / "tas").glob("*.nc"))
    hist_pr = sorted((DATA_DIR / model / "historical" / "pr").glob("*.nc"))
    ssp_tas = sorted((DATA_DIR / model / "ssp585" / "tas").glob("*.nc"))
    ssp_pr = sorted((DATA_DIR / model / "ssp585" / "pr").glob("*.nc"))

    # Load baseline
    tas_base = load_data(hist_tas, "tas", *PERIODS["baseline"])
    pr_base = load_data(hist_pr, "pr", *PERIODS["baseline"])

    tas_thresh = tas_base.groupby("time.month").quantile(HEAT_PERCENTILE / 100.0, dim="time")
    pr_thresh = pr_base.groupby("time.month").quantile(DROUGHT_PERCENTILE / 100.0, dim="time")

    seasons = {
        "JJA": ([6, 7, 8], "Jun-Jul-Aug (NH Summer)"),
        "DJF": ([12, 1, 2], "Dec-Jan-Feb (SH Summer)"),
    }

    def get_seasonal_compound(tas_files, pr_files, start, end, months):
        tas = load_data(tas_files, "tas", start, end)
        pr = load_data(pr_files, "pr", start, end)
        if tas is None or pr is None:
            return None
        if tas.lat.size != pr.lat.size:
            pr = pr.interp(lat=tas.lat, lon=tas.lon)

        tas_th = tas_thresh.sel(month=tas.time.dt.month).drop_vars("month")
        pr_th = pr_thresh.sel(month=pr.time.dt.month).drop_vars("month")

        compound = (tas > tas_th) & (pr < pr_th)

        # Filter to season
        month_mask = compound.time.dt.month.isin(months)
        seasonal = compound.where(month_mask, drop=True)

        return seasonal.mean(dim="time").compute()

    fig = plt.figure(figsize=(14, 6))

    for idx, (season, (months, title)) in enumerate(seasons.items()):
        ax = fig.add_subplot(1, 2, idx + 1, projection=ccrs.Robinson())
        ax.set_global()

        hist_freq = get_seasonal_compound(hist_tas, hist_pr, *PERIODS["historical"], months)
        future_freq = get_seasonal_compound(ssp_tas, ssp_pr, *PERIODS["end_century"], months)

        if hist_freq is None or future_freq is None:
            continue

        change = (future_freq - hist_freq) * 100
        change = convert_lon_to_180(change)

        im = ax.pcolormesh(
            change.lon, change.lat, change,
            cmap="YlOrRd", vmin=0, vmax=30,
            transform=ccrs.PlateCarree(),
            shading="auto"
        )

        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, linestyle='--', edgecolor='gray')

        ax.set_title(f"{title}\nCompound Event Change", fontsize=11)

        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6,
                    label="Change (pp)")

    fig.suptitle(f"Seasonal Compound Event Changes - {model} SSP5-8.5", fontsize=13, y=1.02)

    plt.tight_layout()

    output_path = ARTIFACTS_DIR / "seasonal_maps_coastlines.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def main():
    print("="*70)
    print("REGENERATING FIGURES WITH COASTLINES")
    print("="*70)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Regenerate all regional figures
    plot_regional_compound_risk_fixed()
    plot_mitigation_regional_fixed()
    plot_global_map_with_coastlines()
    plot_seasonal_maps_with_coastlines()

    print("\n" + "="*70)
    print("All figures regenerated with coastlines!")
    print("="*70)


if __name__ == "__main__":
    main()
