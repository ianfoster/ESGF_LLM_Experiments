# ESGF + LLM Experiments: Session Summary

## Overview

This project explores using Large Language Models (LLMs) to interact with climate data from the **Earth System Grid Federation (ESGF)**, specifically CMIP6 model output hosted at U.S. Department of Energy data centers.

**Date:** January 7, 2026
**Data Sources:** Argonne National Lab (ALCF) + Oak Ridge National Lab (ORNL)
**GitHub:** https://github.com/ianfoster/ESGF_LLM_Experiments
**LLM:** Claude Opus 4.5 via [Claude Code](https://claude.com/claude-code)

### Estimated Session Cost

| Token Type | Usage (est.) | Rate | Cost |
|------------|--------------|------|------|
| Input | ~250,000 | $15/1M | ~$3.75 |
| Output | ~65,000 | $75/1M | ~$4.88 |
| **Total** | | | **~$8-10** |

Output: ~2,500 lines of code, ~1.8 GB climate data analyzed, 8 publication-ready figures.

---

## What We Built

### 1. ESGF Search Client (`src/esgf_llm/esgf_client.py`)

A Python client for the ESGF Search API that:
- Searches across federated ESGF nodes
- Lists available models, experiments, and variables
- Extracts OPeNDAP and HTTP download URLs
- Includes metadata for common CMIP6 variables and experiments

```python
from esgf_llm import ESGFClient

client = ESGFClient(node="llnl")
results = client.search(
    variable_id="tas",
    experiment_id="ssp585",
    source_id="GFDL-ESM4",
    table_id="Amon",
)
```

### 2. LLM Interface (`src/esgf_llm/llm_interface.py`)

Natural language interface using Claude to translate queries into ESGF API calls:

```python
from esgf_llm import create_assistant

assistant = create_assistant()
response = assistant.search("Find monthly precipitation data from the high emissions scenario")
# Automatically translates to: variable_id="pr", experiment_id="ssp585", table_id="Amon"
```

### 3. Analysis Scripts

| Script | Description |
|--------|-------------|
| `run_temperature_doe.py` | Downloads from Argonne, analyzes 3 models |
| `run_temperature_ornl.py` | Downloads from Oak Ridge, analyzes 2 models |
| `plot_combined.py` | Combined visualization of all 5 models |
| `run_regional_chicago_nz.py` | Regional analysis: Chicago vs New Zealand |
| `run_regional_analysis.py` | Regional analysis: Oak Ridge vs Livermore |
| `run_gpp_analysis.py` | GPP change analysis (CESM2 SSP5-8.5) |
| `plot_gpp_regions.py` | Regional GPP change maps |
| `test_client.py` | Basic ESGF client test |

---

## Data Access Findings

### DOE Data Center Capabilities

| Center | Node | Access Methods | Status |
|--------|------|----------------|--------|
| **Argonne (ALCF)** | eagle.alcf.anl.gov | HTTPServer, Globus | Public HTTPS works |
| **Oak Ridge (ORNL)** | esgf-node.ornl.gov | HTTPServer, Globus, OPeNDAP | Public HTTPS works |

**Key Discovery:** Both DOE centers expose Globus HTTPS endpoints that are publicly accessible without authentication, enabling direct downloads via `requests`.

### CMIP6 Holdings at DOE Centers

| Metric | Argonne | Oak Ridge | Combined |
|--------|---------|-----------|----------|
| Total Datasets | 5.7M | 6.0M | ~11.6M |
| Models | 126 | 128 | 132 unique |
| Experiments | 258 | 255 | ~260 |
| Variables | 1,184 | 1,180 | ~1,200 |

Full inventory saved to: `doe_cmip6_holdings.json`

---

## Analysis Results

### 1. Global Temperature Projections

**Models Analyzed:**

| Model | Institution | Source | Data Size |
|-------|-------------|--------|-----------|
| GFDL-ESM4 | NOAA/GFDL (USA) | Argonne | 240 MB |
| MIROC6 | JAMSTEC (Japan) | Argonne | 149 MB |
| NorESM2-LM | NCC (Norway) | Argonne | 67 MB |
| GISS-E2-1-H | NASA GISS (USA) | Oak Ridge | 605 MB |
| UKESM1-0-LL | UK Met Office | Oak Ridge | 533 MB |

**Warming by 2100 (relative to 2015-2025):**

| Model | SSP1-2.6 (Low) | SSP5-8.5 (High) | Climate Sensitivity |
|-------|----------------|-----------------|---------------------|
| GFDL-ESM4 | +0.4°C | +3.0°C | Low |
| NorESM2-LM | +0.4°C | +3.0°C | Low |
| MIROC6 | +0.6°C | +3.2°C | Low-Medium |
| GISS-E2-1-H | +0.9°C | +3.6°C | Medium |
| UKESM1-0-LL | +1.4°C | +5.7°C | High |

**Key Finding:** Model spread of +3.0°C to +5.7°C under SSP5-8.5 represents structural uncertainty in climate sensitivity.

### 2. Regional Temperature Analysis

Two regional comparisons with consistent y-axis scaling (6-24°C):

**Chicago vs New Zealand:**
- Chicago shows stronger warming (+4-8°C under SSP5-8.5)
- New Zealand more moderate (+2-4°C), buffered by Southern Ocean

**Oak Ridge vs Livermore (DOE Lab locations):**
- Both show similar warming patterns
- Livermore slightly warmer baseline due to California climate

### 3. GPP (Gross Primary Production) Change

**Query:** How will vegetation productivity change by end of century under SSP5-8.5?

**Configuration:**
- Model: CESM2 (r11i1p1f1)
- Variable: GPP (Lmon table)
- Present: 1995-2014
- Future: 2081-2100

**Results:**

| Region | Present | Future | Change |
|--------|---------|--------|--------|
| Global land mean | 2.71 | 3.91 g C m⁻² day⁻¹ | +44% |
| Arctic (>60°N) | 1.60 | 3.18 | +99% |
| Northern mid-lat (30-60°N) | 1.85 | 3.01 | +62% |
| Tropics (23°S-23°N) | 4.12 | 5.39 | +31% |
| Southern mid-lat (30-60°S) | 2.53 | 4.10 | +62% |

**Key Findings:**
- CO₂ fertilization drives global GPP increase (+44%)
- Arctic shows largest relative increase (+99%) due to longer growing seasons
- Some tropical regions (Amazon, Congo) show GPP *decline* due to heat/drought stress

---

## Generated Outputs

All plots are in the `Artifacts/` directory:

| File | Description |
|------|-------------|
| `temperature_combined_doe.png` | All 5 models, global mean temperature |
| `temperature_chicago_nz.png` | Regional: Chicago vs New Zealand |
| `temperature_regional.png` | Regional: Oak Ridge vs Livermore |
| `gpp_change_cesm2.png` | GPP change maps (4-panel) |
| `gpp_regional_maps.png` | GPP by climate zone |
| `temperature_doe.png` | Argonne models only |
| `temperature_ornl.png` | Oak Ridge models only |

---

## Key Learnings

### What Worked

1. **Globus HTTPS is publicly accessible** - No authentication required for downloads
2. **ESGF Search API is reliable** - Consistent across all index nodes
3. **Single-file datasets are efficient** - GFDL-ESM4 and MIROC6 provide full 2015-2100 in one file
4. **xarray + dask** - Handles multi-file datasets seamlessly

### Challenges Encountered

1. **OPeNDAP reliability varies** - Many nodes return errors or timeouts
2. **Member ID inconsistency** - Different models use different ensemble naming
3. **Partial data at some nodes** - Not all nodes have complete time coverage
4. **File encoding differences** - Some models have incompatible time coordinates

### Recommendations

| Use Case | Recommended Approach |
|----------|---------------------|
| Specific DOE data | Direct HTTPS from Argonne/ORNL |
| Bulk downloads | Globus transfer |
| Data discovery | ESGF Search API |

---

## How to Run

### Setup

```bash
cd ESGF_LLM_Experiments
pip install -e .
pip install xarray netCDF4 matplotlib requests
```

For the LLM interface (optional):
```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key
```

### Run Analyses

```bash
# Test ESGF client
python examples/test_client.py

# Global temperature analysis
python examples/run_temperature_doe.py
python examples/run_temperature_ornl.py
python examples/plot_combined.py

# Regional temperature analysis
python examples/run_regional_chicago_nz.py
python examples/run_regional_analysis.py

# GPP change analysis
python examples/run_gpp_analysis.py
python examples/plot_gpp_regions.py
```

---

## Project Structure

```
ESGF_LLM_Experiments/
├── README.md
├── SESSION_SUMMARY.md          # This file
├── pyproject.toml
├── doe_cmip6_holdings.json     # Full DOE CMIP6 inventory
├── src/esgf_llm/
│   ├── __init__.py
│   ├── esgf_client.py          # ESGF Search API client
│   └── llm_interface.py        # Claude-powered interface
├── examples/
│   ├── test_client.py          # Basic API test
│   ├── run_temperature_doe.py  # Temperature (Argonne)
│   ├── run_temperature_ornl.py # Temperature (Oak Ridge)
│   ├── plot_combined.py        # Combined 5-model plot
│   ├── run_regional_chicago_nz.py   # Chicago vs NZ
│   ├── run_regional_analysis.py     # Oak Ridge vs Livermore
│   ├── run_gpp_analysis.py     # GPP change analysis
│   ├── plot_gpp_regions.py     # Regional GPP maps
│   └── llm_search.py           # LLM demo
├── Artifacts/                  # Generated plots
│   ├── temperature_combined_doe.png
│   ├── temperature_chicago_nz.png
│   ├── temperature_regional.png
│   ├── gpp_change_cesm2.png
│   └── gpp_regional_maps.png
└── data/                       # Downloaded NetCDF files (not in repo)
    ├── doe_nodes/
    ├── ornl/
    └── gpp/
```

---

## References

- [ESGF Search API](https://esgf.github.io/esgf-user-support/search_api.html)
- [CMIP6 Data Access](https://wcrp-cmip.org/cmip-data-access/)
- [Argonne ALCF](https://www.alcf.anl.gov/)
- [Oak Ridge ORNL](https://www.ornl.gov/)
- [GitHub Repository](https://github.com/ianfoster/ESGF_LLM_Experiments)
