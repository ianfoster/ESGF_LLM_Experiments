# ESGF + LLM Experiments

Exploring the use of Large Language Models (LLMs) to discover and analyze climate data from the **Earth System Grid Federation (ESGF)**, specifically CMIP6 model output hosted at U.S. Department of Energy data centers.

## Overview

This project demonstrates:
1. **ESGF Search API client** - Programmatic access to CMIP6 climate model data
2. **LLM-powered natural language interface** - Translate plain English queries into ESGF API calls using Claude
3. **Climate analysis examples** - Temperature projections and GPP (vegetation productivity) change analysis

**Data Sources:** Argonne National Lab (ALCF) and Oak Ridge National Lab (ORNL)

## Installation

```bash
pip install -e .
pip install xarray netCDF4 matplotlib requests
```

For the LLM interface (optional):
```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key
```

## Project Structure

```
ESGF_LLM_Experiments/
├── src/esgf_llm/
│   ├── esgf_client.py      # ESGF Search API client
│   └── llm_interface.py    # Claude-powered natural language interface
├── examples/
│   ├── test_client.py      # Basic ESGF API test
│   ├── run_temperature_doe.py      # Temperature analysis (Argonne)
│   ├── run_temperature_ornl.py     # Temperature analysis (Oak Ridge)
│   ├── plot_combined.py            # Combined 5-model visualization
│   ├── run_regional_analysis.py    # Regional temperature comparison
│   ├── run_gpp_analysis.py         # GPP change analysis (CESM2)
│   └── plot_gpp_regions.py         # Regional GPP maps
└── doe_cmip6_holdings.json  # Inventory of CMIP6 data at DOE centers
```

## Analyses Performed

### 1. Global Temperature Projections

**Query:** Compare global mean temperature under low (SSP1-2.6) vs high (SSP5-8.5) emissions scenarios using 5 CMIP6 models from DOE data centers.

**Models analyzed:**
| Model | Institution | Data Source |
|-------|-------------|-------------|
| GFDL-ESM4 | NOAA/GFDL (USA) | Argonne |
| MIROC6 | JAMSTEC (Japan) | Argonne |
| NorESM2-LM | NCC (Norway) | Argonne |
| GISS-E2-1-H | NASA GISS (USA) | Oak Ridge |
| UKESM1-0-LL | UK Met Office | Oak Ridge |

**Scripts:** `run_temperature_doe.py`, `run_temperature_ornl.py`, `plot_combined.py`

**Results:** Warming by 2100 ranges from +3.0°C (GFDL-ESM4) to +5.7°C (UKESM1-0-LL) under SSP5-8.5.

![Temperature Projections](Artifacts/temperature_combined_doe.png)

### 2. Regional Temperature Analysis

**Query:** Compare temperature projections for Oak Ridge, TN vs Livermore, CA (DOE national lab locations).

**Scripts:** `run_regional_analysis.py`

![Regional Temperature](Artifacts/temperature_regional.png)

### 3. GPP (Gross Primary Production) Change

**Query:** How will vegetation productivity (GPP) change by end of century under SSP5-8.5 using CESM2?

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
| Northern mid-lat | 1.85 | 3.01 | +62% |
| Tropics | 4.12 | 5.39 | +31% |
| Southern mid-lat | 2.53 | 4.10 | +62% |

**Scripts:** `run_gpp_analysis.py`, `plot_gpp_regions.py`

![GPP Change](Artifacts/gpp_change_cesm2.png)

![GPP Regional Maps](Artifacts/gpp_regional_maps.png)

## ESGF Client Usage

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

## LLM Interface Usage

```python
from esgf_llm import create_assistant

assistant = create_assistant()
response = assistant.search("Find monthly precipitation data from the high emissions scenario")
# Automatically translates to: variable_id="pr", experiment_id="ssp585", table_id="Amon"
```

## Key Findings

1. **DOE Globus HTTPS endpoints are publicly accessible** - No authentication required for direct downloads from Argonne and Oak Ridge

2. **Model spread represents structural uncertainty** - The 3-6°C range in warming projections reflects different climate sensitivities across modeling centers

3. **Arctic greening is dramatic** - GPP nearly doubles at high latitudes due to longer growing seasons

4. **Tropical forests at risk** - Despite global CO₂ fertilization, some tropical regions show GPP decline due to heat/drought stress

## References

- [ESGF Search API](https://esgf.github.io/esgf-user-support/search_api.html)
- [CMIP6 Data Access](https://wcrp-cmip.org/cmip-data-access/)
- [Argonne ALCF](https://www.alcf.anl.gov/)
- [Oak Ridge ORNL](https://www.ornl.gov/)
