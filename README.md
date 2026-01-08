# ESGF + LLM Experiments

Exploring the use of Large Language Models (LLMs) to discover and analyze climate data from the **Earth System Grid Federation (ESGF)**, specifically CMIP6 model output hosted at U.S. Department of Energy data centers.

## Overview

This project demonstrates:
1. **ESGF Search API client** - Programmatic access to CMIP6 climate model data
2. **LLM-powered natural language interface** - Translate plain English queries into ESGF API calls using Claude
3. **Climate analysis examples** - Temperature projections and GPP (vegetation productivity) change analysis

**Data Sources:** Argonne National Lab (ALCF) and Oak Ridge National Lab (ORNL)

## LLM Model Used

This project was developed interactively using **Claude Opus 4.5** via [Claude Code](https://claude.com/claude-code), Anthropic's CLI tool for software engineering tasks. The entire workflow—from initial exploration of ESGF data access patterns, through code generation, to analysis and visualization—was driven by natural language conversation with the model.

### Estimated Cost

| Token Type | Usage (est.) | Rate | Cost |
|------------|--------------|------|------|
| Input | ~250,000 | $15/1M | ~$3.75 |
| Output | ~65,000 | $75/1M | ~$4.88 |
| **Total** | | | **~$8-10** |

The session produced ~2,500 lines of working code, analyzed ~1.8 GB of climate data, and generated publication-ready figures.

## Session Queries

The following natural language queries drove the analyses in this project.

I started with *I want to explore the use of LLMs to analyze data maintained by the Earth System Grid Federation. That will 
require you to know what ESGF is; how to access its holdings; and what interesting questions can be used. What do 
you think?* which led to the generation of some initial code. I followed up with the following. (Not listed are a few "*plot this*" or similar requests plus one to "*consider only datasets at Argonne or Oak Ridge*" [because it had problems with some other ESGF sites] and a final "*I want to commit the code that we have written to GitHub with a README that describes it briefly and details the queries performed and the pictures generated.*). 

| Query | Result |
|-------|--------|
| *"Suggest an example analysis that will access multiple datasets but not be too expensive"* | Global temperature projections comparing 5 CMIP6 models under SSP1-2.6 vs SSP5-8.5 |
| *"I want to repeat the analysis but focusing just on Chicago and New Zealand"* | Regional temperature comparison between Northern and Southern Hemisphere locations |
| *"Repeat for Oak Ridge and Livermore"* | Temperature projections at DOE national lab locations |
| *"Same y-axis range for both graphs"* | Consistent 6-24°C scale across all 4 regional plots |
| *"How will GPP change in the future compared to now, using CESM2 and ssp585?"* | GPP change analysis showing +44% global increase, +99% in Arctic |
| *"Can you create global maps for them?"* | Regional GPP maps by climate zone |
| *"Create a table of all CMIP6 run datasets that are at Argonne and ORNL"* | `doe_cmip6_holdings.json` inventory (~11.6M datasets) |

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
│   ├── test_client.py              # Basic ESGF API test
│   ├── run_temperature_doe.py      # Temperature analysis (Argonne)
│   ├── run_temperature_ornl.py     # Temperature analysis (Oak Ridge)
│   ├── plot_combined.py            # Combined 5-model visualization
│   ├── run_regional_analysis.py    # Regional: Oak Ridge vs Livermore
│   ├── run_regional_chicago_nz.py  # Regional: Chicago vs New Zealand
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

#### Chicago vs New Zealand

**Query:** Compare temperature projections for Chicago (Northern Hemisphere) vs New Zealand (Southern Hemisphere).

**Scripts:** `run_regional_chicago_nz.py`

![Chicago vs NZ](Artifacts/temperature_chicago_nz.png)

#### Oak Ridge vs Livermore

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
