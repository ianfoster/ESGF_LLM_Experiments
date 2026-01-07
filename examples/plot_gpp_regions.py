#!/usr/bin/env python3
"""
Regional GPP change maps showing each climate zone.
"""

import warnings
warnings.filterwarnings("ignore")

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


VARIABLE = "gpp"
DATA_DIR = Path("data/gpp")

# Time periods
PRESENT = (1995, 2014)
FUTURE = (2081, 2100)

# Regional definitions
REGIONS = {
    "Arctic (>60°N)": {"lat_min": 60, "lat_max": 90, "color": "#4575b4"},
    "Northern Mid-Lat (30°N-60°N)": {"lat_min": 30, "lat_max": 60, "color": "#91bfdb"},
    "Tropics (23°S-23°N)": {"lat_min": -23, "lat_max": 23, "color": "#fc8d59"},
    "Southern Mid-Lat (60°S-30°S)": {"lat_min": -60, "lat_max": -30, "color": "#fee090"},
}


def compute_period_mean(files: list[Path], start_year: int, end_year: int) -> xr.DataArray:
    """Compute mean GPP for a time period."""
    ds = xr.open_mfdataset(files, combine="by_coords")
    gpp = ds[VARIABLE]
    gpp_period = gpp.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    annual = gpp_period.groupby("time.year").mean("time")
    period_mean = annual.mean("year")
    return period_mean.compute()


def main():
    print("Loading GPP data...")

    hist_files = sorted((DATA_DIR / "historical").glob("*.nc"))
    ssp_files = sorted((DATA_DIR / "ssp585").glob("*.nc"))

    if not hist_files or not ssp_files:
        print("Error: Run run_gpp_analysis.py first to download data")
        return

    # Compute period means
    print("Computing present-day GPP...", end=" ", flush=True)
    gpp_present = compute_period_mean(hist_files, PRESENT[0], PRESENT[1])
    print("done")

    print("Computing future GPP...", end=" ", flush=True)
    gpp_future = compute_period_mean(ssp_files, FUTURE[0], FUTURE[1])
    print("done")

    # Convert to g C m-2 day-1
    scale = 86400 * 1000
    gpp_present_gday = gpp_present * scale
    gpp_future_gday = gpp_future * scale
    gpp_pct_change = xr.where(
        gpp_present > 1e-12,
        ((gpp_future - gpp_present) / gpp_present) * 100,
        np.nan
    )

    # Get coordinates
    lon = gpp_present.lon.values
    lat = gpp_present.lat.values

    # Land mask
    land_mask = gpp_present > 1e-12
    weights = np.cos(np.deg2rad(gpp_present.lat))

    # Compute regional statistics
    print("\nComputing regional statistics...")
    regional_stats = {}
    for region_name, region_def in REGIONS.items():
        lat_mask = (gpp_present.lat >= region_def["lat_min"]) & (gpp_present.lat <= region_def["lat_max"])

        present_reg = gpp_present_gday.where(lat_mask & land_mask)
        future_reg = gpp_future_gday.where(lat_mask & land_mask)

        weights_reg = weights.where(lat_mask).fillna(0)

        p = float(present_reg.weighted(weights_reg).mean().values)
        f = float(future_reg.weighted(weights_reg).mean().values)
        c = f - p
        pct = (c / p) * 100 if p > 0 else 0

        regional_stats[region_name] = {
            "present": p,
            "future": f,
            "change": c,
            "pct_change": pct,
            "color": region_def["color"],
            "lat_min": region_def["lat_min"],
            "lat_max": region_def["lat_max"],
        }
        print(f"  {region_name}: {p:.2f} → {f:.2f} ({pct:+.0f}%)")

    # Create figure with 4 regional maps + bar chart
    print("\nGenerating regional maps...")

    fig = plt.figure(figsize=(18, 14))

    # Create grid: 2x2 for regional maps, bottom for bar chart
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.15)

    # Plot each region
    region_list = list(REGIONS.keys())
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, (region_name, region_def) in enumerate(REGIONS.items()):
        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col])

        stats = regional_stats[region_name]
        lat_min, lat_max = stats["lat_min"], stats["lat_max"]

        # Create regional mask for highlighting
        lat_idx = (lat >= lat_min) & (lat <= lat_max)

        # Plot percent change for this region
        pct_data = gpp_pct_change.values.copy()

        # Mask out areas outside region (show as gray)
        mask_2d = np.zeros_like(pct_data, dtype=bool)
        for i, la in enumerate(lat):
            if lat_min <= la <= lat_max:
                mask_2d[i, :] = True

        # Create masked array for background (non-region)
        background = np.ma.masked_where(mask_2d | np.isnan(pct_data), pct_data)

        # Plot background in gray
        ax.pcolormesh(lon, lat, np.where(~mask_2d & ~np.isnan(pct_data), 0, np.nan),
                      cmap='Greys', vmin=-1, vmax=1, shading='auto', alpha=0.3)

        # Plot the region with color
        region_data = np.where(mask_2d, pct_data, np.nan)
        im = ax.pcolormesh(lon, lat, region_data,
                          cmap='RdBu', vmin=-50, vmax=50, shading='auto')

        # Add region boundary lines
        ax.axhline(y=lat_min, color=stats["color"], linewidth=2, linestyle='--')
        ax.axhline(y=lat_max, color=stats["color"], linewidth=2, linestyle='--')

        # Highlight region with colored border
        rect = mpatches.Rectangle((lon.min(), lat_min), lon.max() - lon.min(), lat_max - lat_min,
                                   linewidth=3, edgecolor=stats["color"], facecolor='none')
        ax.add_patch(rect)

        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'{region_name}\n'
                     f'GPP: {stats["present"]:.1f} → {stats["future"]:.1f} g C m⁻² day⁻¹ '
                     f'({stats["pct_change"]:+.0f}%)',
                     fontsize=11, color=stats["color"], fontweight='bold')

        plt.colorbar(im, ax=ax, label='% change', shrink=0.8)

    # Bar chart at bottom
    ax_bar = fig.add_subplot(gs[2, :])

    regions = list(regional_stats.keys())
    x = np.arange(len(regions))
    width = 0.35

    present_vals = [regional_stats[r]["present"] for r in regions]
    future_vals = [regional_stats[r]["future"] for r in regions]
    colors = [regional_stats[r]["color"] for r in regions]
    pct_changes = [regional_stats[r]["pct_change"] for r in regions]

    bars1 = ax_bar.bar(x - width/2, present_vals, width, label=f'Present ({PRESENT[0]}-{PRESENT[1]})',
                       color=[c + '80' for c in colors], edgecolor=colors, linewidth=2)
    bars2 = ax_bar.bar(x + width/2, future_vals, width, label=f'Future ({FUTURE[0]}-{FUTURE[1]})',
                       color=colors, edgecolor='black', linewidth=1)

    # Add percent change labels
    for i, (bar, pct) in enumerate(zip(bars2, pct_changes)):
        ax_bar.annotate(f'+{pct:.0f}%',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', va='bottom', fontsize=11, fontweight='bold',
                       color=colors[i])

    ax_bar.set_ylabel('GPP (g C m⁻² day⁻¹)', fontsize=12)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([r.replace(' (', '\n(') for r in regions], fontsize=10)
    ax_bar.legend(loc='upper right', fontsize=10)
    ax_bar.set_ylim(0, max(future_vals) * 1.25)
    ax_bar.grid(True, axis='y', alpha=0.3)
    ax_bar.set_title('Regional GPP Comparison: Present vs Future', fontsize=13)

    plt.suptitle('CESM2 SSP5-8.5: Regional GPP Changes by Climate Zone\n'
                 f'Present ({PRESENT[0]}-{PRESENT[1]}) vs Future ({FUTURE[0]}-{FUTURE[1]})',
                 fontsize=15, y=0.98)

    output_path = Path("gpp_regional_maps.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {output_path.absolute()}")


if __name__ == "__main__":
    main()
