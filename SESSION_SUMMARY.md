# ESGF + LLM Experiments: Session Summary

## Overview

This project explores using Large Language Models (LLMs) to interact with climate data from the **Earth System Grid Federation (ESGF)**, specifically CMIP6 model output hosted at U.S. Department of Energy data centers.

**Date:** January 7, 2026
**Data Sources:** Argonne National Lab (ALCF) + Oak Ridge National Lab (ORNL)

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
| `examples/run_temperature_doe.py` | Downloads from Argonne, analyzes 3 models |
| `examples/run_temperature_ornl.py` | Downloads from Oak Ridge, analyzes 2 models |
| `examples/plot_combined.py` | Combined visualization of all 5 models |
| `examples/test_client.py` | Basic ESGF client test |

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

### Models Analyzed

| Model | Institution | Source | Data Size |
|-------|-------------|--------|-----------|
| GFDL-ESM4 | NOAA/GFDL (USA) | Argonne | 240 MB |
| MIROC6 | JAMSTEC (Japan) | Argonne | 149 MB |
| NorESM2-LM | NCC (Norway) | Argonne | 67 MB |
| GISS-E2-1-H | NASA GISS (USA) | Oak Ridge | 605 MB |
| UKESM1-0-LL | UK Met Office | Oak Ridge | 533 MB |

**Total Downloaded:** ~1.6 GB

### Global Temperature Projections (2015-2100)

| Model | SSP1-2.6 (Low) | SSP5-8.5 (High) | Climate Sensitivity |
|-------|----------------|-----------------|---------------------|
| GFDL-ESM4 | +0.4°C | +3.0°C | Low |
| NorESM2-LM | +0.4°C | +3.0°C | Low |
| MIROC6 | +0.6°C | +3.2°C | Low-Medium |
| GISS-E2-1-H | +0.9°C | +3.6°C | Medium |
| UKESM1-0-LL | +1.4°C | +5.7°C | High |

**Key Finding:** Model spread of +3.0°C to +5.7°C under SSP5-8.5 represents structural uncertainty in climate sensitivity across modeling centers.

---

## Generated Outputs

### Plots

1. **`temperature_combined_doe.png`** - All 5 models from both DOE centers
2. **`temperature_doe.png`** - Argonne models (GFDL-ESM4, MIROC6, NorESM2-LM)
3. **`temperature_ornl.png`** - Oak Ridge models (GISS-E2-1-H, UKESM1-0-LL)

### Data Files

- `doe_cmip6_holdings.json` - Complete inventory of CMIP6 at DOE centers
- `data/doe_nodes/` - Downloaded NetCDF files from Argonne
- `data/ornl/` - Downloaded NetCDF files from Oak Ridge

---

## Key Learnings

### What Worked

1. **Globus HTTPS is publicly accessible** - No authentication required for downloads
2. **ESGF Search API is reliable** - Consistent across all index nodes
3. **Single-file datasets are efficient** - GFDL-ESM4 and MIROC6 provide full 2015-2100 in one file
4. **Pangeo Cloud catalog** - Most reliable for analysis (Zarr on GCS)

### Challenges Encountered

1. **OPeNDAP reliability varies** - Many nodes return errors or timeouts
2. **Member ID inconsistency** - Different models use different ensemble naming
3. **Partial data at some nodes** - Not all nodes have complete time coverage
4. **File encoding differences** - Some models have incompatible time coordinates

### Recommendations

| Use Case | Recommended Approach |
|----------|---------------------|
| Quick analysis | Pangeo Cloud catalog (intake-esm + Zarr) |
| Specific DOE data | Direct HTTPS from Argonne/ORNL |
| Bulk downloads | Globus transfer |
| Data discovery | ESGF Search API |

---

## How to Run

### Setup

```bash
cd ESGF_LLM_Experiments
pip install -e .
pip install xarray netCDF4 matplotlib intake-esm gcsfs
```

### Run Analysis

```bash
# Test ESGF client
python examples/test_client.py

# Download from Argonne and analyze
python examples/run_temperature_doe.py

# Download from Oak Ridge and analyze
python examples/run_temperature_ornl.py

# Generate combined plot
python examples/plot_combined.py
```

### Use LLM Interface (requires Anthropic API key)

```bash
export ANTHROPIC_API_KEY=your_key
python examples/llm_search.py
```

---

## Project Structure

```
ESGF_LLM_Experiments/
├── pyproject.toml              # Dependencies
├── src/esgf_llm/
│   ├── __init__.py
│   ├── esgf_client.py          # ESGF Search API client
│   └── llm_interface.py        # Claude-powered natural language interface
├── examples/
│   ├── test_client.py          # Basic API test
│   ├── run_temperature_doe.py  # Argonne analysis
│   ├── run_temperature_ornl.py # Oak Ridge analysis
│   ├── plot_combined.py        # Combined visualization
│   └── llm_search.py           # LLM demo
├── data/
│   ├── doe_nodes/              # Argonne downloads
│   ├── ornl/                   # Oak Ridge downloads
│   └── argonne/                # Additional Argonne data
├── doe_cmip6_holdings.json     # Full DOE inventory
├── temperature_combined_doe.png
├── temperature_doe.png
├── temperature_ornl.png
└── SESSION_SUMMARY.md          # This file
```

---

## Next Steps

Potential extensions:
1. **Other variables** - Precipitation (pr), sea ice (siconc), sea level (zos)
2. **Regional analysis** - Subset by lat/lon for specific regions
3. **Multi-model ensemble** - Compute ensemble mean and spread
4. **Tool use** - Give Claude tools to search, download, and analyze autonomously
5. **Visualization dashboard** - Interactive plots with Panel or Streamlit

---

## References

- [ESGF Search API](https://esgf.github.io/esgf-user-support/search_api.html)
- [CMIP6 Data Access](https://wcrp-cmip.org/cmip-data-access/)
- [Pangeo CMIP6 Catalog](https://pangeo-data.github.io/pangeo-cmip6-cloud/)
- [Argonne ALCF](https://www.alcf.anl.gov/)
- [Oak Ridge ORNL](https://www.ornl.gov/)
