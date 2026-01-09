#!/usr/bin/env python3
"""
Fix consecutive event plots with proper coastlines.
"""

import warnings
warnings.filterwarnings("ignore")

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from scipy import ndimage

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
CONSECUTIVE_THRESHOLDS = [2, 3, 6]


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


def count_consecutive_events(compound_bool, min_length):
    """Count events with at least min_length consecutive True values."""
    if not np.any(compound_bool):
        return 0
    labeled, num_features = ndimage.label(compound_bool.astype(int))
    count = 0
    for i in range(1, num_features + 1):
        if np.sum(labeled == i) >= min_length:
            count += 1
    return count


def compute_consecutive_frequency(compound, min_months):
    """Compute frequency of consecutive compound events."""
    compound_computed = compound.compute()
    n_time = compound_computed.time.size
    n_years = n_time / 12

    result = np.zeros((compound_computed.lat.size, compound_computed.lon.size))

    for i in range(compound_computed.lat.size):
        for j in range(compound_computed.lon.size):
            time_series = compound_computed.values[:, i, j]
            n_events = count_consecutive_events(time_series, min_months)
            result[i, j] = n_events / n_years * 10

    return xr.DataArray(
        result,
        dims=["lat", "lon"],
        coords={"lat": compound_computed.lat, "lon": compound_computed.lon}
    )


def compute_mean_event_length(compound):
    """Compute mean length of compound events."""
    compound_computed = compound.compute()
    result = np.zeros((compound_computed.lat.size, compound_computed.lon.size))

    for i in range(compound_computed.lat.size):
        for j in range(compound_computed.lon.size):
            time_series = compound_computed.values[:, i, j]
            if not np.any(time_series):
                result[i, j] = 0
                continue
            labeled, num_features = ndimage.label(time_series.astype(int))
            if num_features == 0:
                result[i, j] = 0
                continue
            lengths = [np.sum(labeled == k) for k in range(1, num_features + 1)]
            result[i, j] = np.mean(lengths)

    return xr.DataArray(
        result,
        dims=["lat", "lon"],
        coords={"lat": compound_computed.lat, "lon": compound_computed.lon}
    )


def get_consecutive_data(model):
    """Get consecutive event data for a model."""
    hist_tas = sorted((DATA_DIR / model / "historical" / "tas").glob("*.nc"))
    hist_pr = sorted((DATA_DIR / model / "historical" / "pr").glob("*.nc"))
    ssp_tas = sorted((DATA_DIR / model / "ssp585" / "tas").glob("*.nc"))
    ssp_pr = sorted((DATA_DIR / model / "ssp585" / "pr").glob("*.nc"))

    if not all([hist_tas, hist_pr, ssp_tas, ssp_pr]):
        return None

    # Load baseline
    tas_base = load_data(hist_tas, "tas", *PERIODS["baseline"])
    pr_base = load_data(hist_pr, "pr", *PERIODS["baseline"])

    if tas_base is None or pr_base is None:
        return None

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

        period_data = {"consecutive": {}}

        for min_months in CONSECUTIVE_THRESHOLDS:
            print(f"    Computing {min_months}+ month events for {period_name}...")
            freq = compute_consecutive_frequency(compound, min_months)
            period_data["consecutive"][min_months] = freq

        print(f"    Computing mean event length for {period_name}...")
        period_data["mean_length"] = compute_mean_event_length(compound)

        results[period_name] = period_data

    return results


def plot_consecutive_with_coastlines():
    """Create consecutive event plots with coastlines."""

    print("Loading consecutive event data for GFDL-ESM4...")
    data = get_consecutive_data("GFDL-ESM4")

    if data is None:
        print("Could not load data")
        return

    # Also load MIROC6 for comparison
    print("Loading consecutive event data for MIROC6...")
    data_miroc = get_consecutive_data("MIROC6")

    fig = plt.figure(figsize=(16, 12))

    # Panel 1: Bar chart of consecutive event frequency
    ax1 = fig.add_subplot(2, 2, 1)

    x = np.arange(len(CONSECUTIVE_THRESHOLDS))
    width = 0.35

    hist_means = []
    future_means = []

    for min_months in CONSECUTIVE_THRESHOLDS:
        hist_gfdl = data["historical"]["consecutive"][min_months]
        future_gfdl = data["end_century"]["consecutive"][min_months]

        weights = np.cos(np.deg2rad(hist_gfdl.lat))
        h_val = float(hist_gfdl.weighted(weights).mean().values)
        f_val = float(future_gfdl.weighted(weights).mean().values)

        if data_miroc:
            hist_miroc = data_miroc["historical"]["consecutive"][min_months]
            future_miroc = data_miroc["end_century"]["consecutive"][min_months]
            h_val = (h_val + float(hist_miroc.weighted(weights).mean().values)) / 2
            f_val = (f_val + float(future_miroc.weighted(weights).mean().values)) / 2

        hist_means.append(h_val)
        future_means.append(f_val)

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

    for i, (h, f) in enumerate(zip(hist_means, future_means)):
        if h > 0.001:
            increase = ((f - h) / h) * 100
            ax1.annotate(f'+{increase:.0f}%', xy=(i + width/2, f + 0.1),
                        ha='center', fontsize=10, fontweight='bold', color='#b2182b')

    # Panel 2: Mean event length comparison
    ax2 = fig.add_subplot(2, 2, 2)

    models = ["GFDL-ESM4"]
    hist_lengths = []
    future_lengths = []

    weights = np.cos(np.deg2rad(data["historical"]["mean_length"].lat))
    hist_lengths.append(float(data["historical"]["mean_length"].weighted(weights).mean().values))
    future_lengths.append(float(data["end_century"]["mean_length"].weighted(weights).mean().values))

    if data_miroc:
        models.append("MIROC6")
        hist_lengths.append(float(data_miroc["historical"]["mean_length"].weighted(weights).mean().values))
        future_lengths.append(float(data_miroc["end_century"]["mean_length"].weighted(weights).mean().values))

    x = np.arange(len(models))
    bars1 = ax2.bar(x - width/2, hist_lengths, width, label='Historical', color='#2166ac', alpha=0.8)
    bars2 = ax2.bar(x + width/2, future_lengths, width, label='End-Century', color='#b2182b', alpha=0.8)

    ax2.set_ylabel('Mean Event Length (months)', fontsize=11)
    ax2.set_title('Average Compound Event Duration', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    for i, (h, f) in enumerate(zip(hist_lengths, future_lengths)):
        change = f - h
        ax2.annotate(f'+{change:.2f}mo', xy=(i + width/2, f + 0.03),
                    ha='center', fontsize=10, fontweight='bold', color='#b2182b')

    # Panel 3: Global map of 3+ month event change
    ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.Robinson())
    ax3.set_global()

    hist_3mo = data["historical"]["consecutive"][3]
    future_3mo = data["end_century"]["consecutive"][3]
    change_3mo = future_3mo - hist_3mo
    change_3mo = convert_lon_to_180(change_3mo)

    im3 = ax3.pcolormesh(
        change_3mo.lon, change_3mo.lat, change_3mo,
        cmap="YlOrRd", vmin=0, vmax=5,
        transform=ccrs.PlateCarree(),
        shading="auto"
    )

    ax3.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
    ax3.add_feature(cfeature.BORDERS, linewidth=0.2, linestyle='--', edgecolor='gray')

    ax3.set_title('Change in 3+ Month Compound Events\nGFDL-ESM4 (events/decade)', fontsize=11)
    cbar3 = plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar3.set_label("Change (events/decade)")

    # Panel 4: Global map of mean event length change
    ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.Robinson())
    ax4.set_global()

    hist_length = data["historical"]["mean_length"]
    future_length = data["end_century"]["mean_length"]
    length_change = future_length - hist_length
    length_change = convert_lon_to_180(length_change)

    im4 = ax4.pcolormesh(
        length_change.lon, length_change.lat, length_change,
        cmap="YlOrRd", vmin=0, vmax=1.5,
        transform=ccrs.PlateCarree(),
        shading="auto"
    )

    ax4.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
    ax4.add_feature(cfeature.BORDERS, linewidth=0.2, linestyle='--', edgecolor='gray')

    ax4.set_title('Change in Mean Event Duration\nGFDL-ESM4 (months)', fontsize=11)
    cbar4 = plt.colorbar(im4, ax=ax4, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar4.set_label("Change (months)")

    plt.tight_layout()

    output_path = ARTIFACTS_DIR / "consecutive_events.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {output_path}")


def plot_consecutive_regional():
    """Create regional consecutive event maps with coastlines."""

    print("\nCreating regional consecutive event maps...")

    data = get_consecutive_data("GFDL-ESM4")
    if data is None:
        return

    hist_6mo = data["historical"]["consecutive"][6]
    future_6mo = data["end_century"]["consecutive"][6]
    change_6mo = future_6mo - hist_6mo
    change_6mo = convert_lon_to_180(change_6mo)

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
            region_data = change_6mo.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max)
            )
        except Exception as e:
            print(f"  Error extracting {region_name}: {e}")
            continue

        im = ax.pcolormesh(
            region_data.lon, region_data.lat, region_data,
            cmap="YlOrRd", vmin=0, vmax=1,
            transform=ccrs.PlateCarree(),
            shading="auto"
        )

        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', edgecolor='gray')
        ax.add_feature(cfeature.LAKES, alpha=0.3, facecolor='lightblue')

        weights = np.cos(np.deg2rad(region_data.lat))
        mean_change = float(region_data.weighted(weights).mean().values)

        ax.set_title(f"{bounds['title']}\nMean change: +{mean_change:.2f} events/decade", fontsize=11)

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        plt.colorbar(im, ax=ax, label="Change (events/decade)", shrink=0.7, pad=0.02)

    fig.suptitle(
        "6+ Month Compound Events: Regional Change\nGFDL-ESM4 SSP5-8.5 (events per decade)",
        fontsize=13, y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = ARTIFACTS_DIR / "consecutive_regional.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    print("="*70)
    print("FIXING CONSECUTIVE EVENT PLOTS")
    print("="*70)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_consecutive_with_coastlines()
    plot_consecutive_regional()

    print("\n" + "="*70)
    print("Consecutive event plots fixed!")
    print("="*70)


if __name__ == "__main__":
    main()
