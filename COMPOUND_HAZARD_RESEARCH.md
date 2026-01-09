# Compound Climate Hazard Research: Heat-Drought Concurrence Analysis

## Executive Summary

This analysis investigated whether climate change disproportionately increases compound heat-drought events using CMIP6 model projections from the Earth System Grid Federation. While our specific non-linearity hypothesis was not supported, we discovered critically important findings about the dramatic increase in compound hazards and the changing correlation structure between heat and drought under climate change.

## Research Question

**Does climate change disproportionately increase compound heat-drought events, with amplification being non-linear and spatially heterogeneous?**

## Methodology

### Data Sources
- **Models**: GFDL-ESM4, MIROC6, NorESM2-LM (3 CMIP6 Earth System Models)
- **Variables**: Near-surface air temperature (tas), Precipitation (pr)
- **Scenarios**: Historical (1850-2014), SSP5-8.5 (2015-2100)
- **Data Volume**: ~1.5 GB from DOE data centers (Argonne, Oak Ridge)

### Compound Event Definition
- **Heat event**: Monthly temperature > 90th percentile of 1985-2014 baseline
- **Drought event**: Monthly precipitation < 10th percentile of 1985-2014 baseline
- **Compound event**: Heat AND drought occurring simultaneously

### Analysis Periods
- Baseline for thresholds: 1985-2014
- Historical comparison: 1995-2014
- Mid-century projection: 2040-2069
- End-century projection: 2070-2099

## Key Findings

### 1. Dramatic Increase in Compound Events

| Model | Historical | End-Century | Increase |
|-------|------------|-------------|----------|
| GFDL-ESM4 | 1.58% | 13.98% | +785% |
| MIROC6 | 1.69% | 12.81% | +658% |
| NorESM2-LM | 1.17% | 13.55% | +1059% |
| **Mean** | **1.48%** | **13.45%** | **+833%** |

**Key Finding**: Compound heat-drought events increase by **657-1055%** (mean 833%) from historical to end-of-century under SSP5-8.5.

### 2. Regional Hotspots of Compound Risk

| Region | Mean Change | Max Change | Risk Level |
|--------|-------------|------------|------------|
| Amazon | +41.4 pp | +66.1 pp | EXTREME |
| Southern Africa | +23.3 pp | +37.5 pp | Very High |
| Mediterranean | +21.3 pp | +33.3 pp | Very High |
| Australia | +14.2 pp | +28.5 pp | High |
| US Southwest | +11.0 pp | +21.9 pp | High |
| South Asia | +10.9 pp | +32.6 pp | High |

*pp = percentage points*

**Key Finding**: The Amazon and Southern Africa emerge as extreme compound hazard hotspots, with some areas experiencing compound events >60% more often than historical.

### 3. Hypothesis Testing: Non-Linearity

**Original Hypothesis**: Compound risk grows faster than expected from individual hazards.

**Result**: NOT SUPPORTED (Non-linearity ratio = 0.98x)

The increase in compound events is almost exactly what would be expected if heat and drought increased independently. This means:
- Compound risk ≈ Heat risk × Drought risk
- No evidence of synergistic amplification

### 4. Unexpected Finding: Changing Correlation Structure

| Period | GFDL-ESM4 | MIROC6 | NorESM2-LM |
|--------|-----------|--------|------------|
| Historical | 1.28x | 1.36x | 0.91x |
| Mid-century | 1.04x | 1.04x | 0.90x |
| End-century | 1.02x | 1.01x | 0.96x |

**Key Finding**: In the historical climate, heat and drought are positively correlated (amplification factor >1), meaning they tend to co-occur. Under climate change, this correlation weakens toward independence (~1.0x).

**Interpretation**: This correlation breakdown occurs because:
1. Heat events become so frequent (~90% of months by end-century) that they lose selectivity
2. Drought patterns shift regionally in complex ways
3. The historical land-atmosphere feedbacks that linked heat and drought may change

## Scientific Implications

### What We Learned

1. **Absolute risk matters more than correlation**: Even though compound events don't grow "synergistically," the 8-10x increase in frequency is catastrophic for:
   - Agriculture (crop failures)
   - Wildfire risk
   - Ecosystem stress
   - Water resources
   - Human health

2. **Regional hotspots require targeted adaptation**: The Amazon, Mediterranean, and Southern Africa face fundamentally different climate futures.

3. **Correlation changes under forcing**: Climate statistics that hold in the historical period may not persist under strong forcing - this has implications for statistical downscaling and impact modeling.

### Comparison to Literature

Our findings align with:
- Zscheischler et al. (2018): Compound extremes increase across most regions
- Ridder et al. (2020): Substantial increases in compound hot-dry events globally
- AghaKouchak et al. (2020): Compound hazards are increasing in frequency and intensity

Our novel contribution:
- Quantified the weakening of heat-drought correlation under climate change
- Identified the Amazon as the most extreme compound hazard hotspot

## Visualizations Generated

1. **compound_hazard_analysis.png**: 4-panel figure showing:
   - Global map of compound event frequency change
   - Amplification factor spatial patterns
   - Time evolution across models
   - Non-linearity test results

2. **compound_hazard_regional.png**: 6-panel regional analysis showing compound risk hotspots:
   - Mediterranean, US Southwest, Amazon
   - Australia, Southern Africa, South Asia

## Data and Code

- **Analysis script**: `examples/run_compound_hazard_analysis.py`
- **Downloaded data**: `data/compound_hazard/` (~1.5 GB)
- **Output artifacts**: `Artifacts/compound_hazard_*.png`, `Artifacts/compound_hazard_summary.json`

## Conclusions

While our specific non-linearity hypothesis was not supported, this analysis reveals a critical climate risk: **compound heat-drought events will increase by nearly an order of magnitude under high emissions**, with the Amazon, Mediterranean, and Southern Africa facing the most severe increases. The finding that heat-drought correlation weakens under climate change has important implications for how we model and prepare for compound climate risks.

## Follow-Up Analysis: Seasonal Breakdown

### When Do Compound Events Occur?

We conducted a follow-up analysis examining the seasonal distribution of compound heat-drought events.

### Seasonal Results (Multi-Model Mean)

| Season | Historical | End-Century | Absolute Change | % Increase |
|--------|------------|-------------|-----------------|------------|
| Dec-Jan-Feb | 1.33% | 12.27% | +10.95 pp | +826% |
| Mar-Apr-May | 1.42% | 12.46% | +11.05 pp | +778% |
| **Jun-Jul-Aug** | **1.65%** | **14.79%** | **+13.14 pp** | **+797%** |
| Sep-Oct-Nov | 1.53% | 14.26% | +12.73 pp | +833% |

### Key Seasonal Findings

1. **Summer (JJA) sees the largest absolute increase**: +13.1 percentage points globally
   - This is critical because JJA is the Northern Hemisphere growing season
   - Agricultural impacts will be maximized

2. **No season escapes**: All seasons see 778-833% increases
   - Year-round compound risk, not just summer

3. **Hemispheric Patterns**:
   - **Northern Hemisphere**: JJA (summer) peaks at ~15.5% compound frequency
   - **Southern Hemisphere**: SON (spring) and DJF (summer) peak at ~17.5%

4. **Spatial Hotspots by Season**:
   - **JJA (NH summer)**: Mediterranean, Central Asia, Western North America
   - **DJF (SH summer)**: Amazon, Southern Africa, Australia

### Implications for Agriculture

The concentration of compound event increases in summer months (JJA for NH, DJF for SH) has severe implications:
- **Crop failures**: Heat + drought during growing season is catastrophic
- **Irrigation demand**: Peak water demand coincides with peak scarcity
- **Livestock stress**: Compound heat-drought most dangerous during warm months

### Seasonal Visualizations

- `Artifacts/compound_hazard_seasonal.png`: 4-panel seasonal breakdown
- `Artifacts/compound_hazard_hemispheric.png`: NH vs SH seasonal patterns

---

## Follow-Up Analysis: Mitigation Benefit (SSP1-2.6 vs SSP5-8.5)

### Can We Avoid Compound Risk Through Mitigation?

We compared low-emissions (SSP1-2.6, Paris Agreement aligned) vs high-emissions (SSP5-8.5) scenarios to quantify avoidable compound risk.

### Global Mitigation Results

| Period | SSP5-8.5 | SSP1-2.6 | Avoided | Reduction |
|--------|----------|----------|---------|-----------|
| Mid-Century (2040-2069) | 9.30% | 5.77% | 3.5 pp | **38%** |
| End-Century (2070-2099) | 13.40% | 5.80% | 7.6 pp | **57%** |

### Key Finding: Mitigation "Stops the Bleeding"

Under SSP1-2.6, compound risk **stabilizes** at ~5.8% from mid-century onward. Under SSP5-8.5, it continues climbing to 13.4%. Climate mitigation doesn't just slow the increase—it halts it.

### Regional Mitigation Benefits

| Region | Avoided Risk | Interpretation |
|--------|--------------|----------------|
| **Amazon** | **30.6 pp** | Greatest mitigation benefit globally |
| Southern Africa | 16.1 pp | Major benefit for vulnerable region |
| Mediterranean | 15.3 pp | Significant benefit for agriculture |
| Australia | 9.1 pp | Substantial benefit |

**Critical insight**: The regions facing the greatest compound risk under high emissions (Amazon, Southern Africa) also stand to benefit most from mitigation.

### Policy Implications

1. **More than half of end-century compound risk is avoidable** through emissions reductions
2. **Stabilization is achievable**: Under SSP1-2.6, compound risk stops increasing after mid-century
3. **Highest-risk regions benefit most**: The Amazon could avoid 30+ percentage points of compound risk
4. **Time-sensitive**: The divergence between scenarios grows over time—early action is more valuable

### Mitigation Visualizations

- `Artifacts/mitigation_comparison.png`: 4-panel scenario comparison
- `Artifacts/mitigation_regional.png`: Regional avoided risk maps

---

## Follow-Up Analysis: Consecutive Multi-Month Events

### Why Persistence Matters

A single month of compound heat-drought stress is damaging. Multiple consecutive months are catastrophic—causing crop failures, reservoir depletion, ecosystem collapse, and megafires.

### Explosive Growth in Persistent Events

| Duration | Historical | End-Century | Change |
|----------|------------|-------------|--------|
| 2+ months | 0.13/decade | 2.67/decade | **+1,908%** |
| 3+ months | 0.02/decade | 0.79/decade | **+4,320%** |
| 6+ months | ~0/decade | 0.05/decade | **Emergence of new phenomenon** |

### Key Findings

1. **Multi-month events explode in frequency**: 2+ month events increase 20x, 3+ month events increase 40x

2. **6+ month droughts emerge as new phenomenon**: These essentially don't exist in the historical climate but become real under climate change (~1 every 200 years globally → ~1 every 20 years in hotspots)

3. **Mean event duration increases**: From ~1.0 to ~1.3 months (+30%)—compound events become more persistent

4. **Spatial concentration**: The Amazon and Central Africa show the highest increase in 3+ month events (up to 5 additional events per decade)

### Implications

- **Agricultural planning**: Multi-season crop failures become more likely
- **Water resources**: Consecutive dry months deplete reservoirs and groundwater
- **Ecosystem thresholds**: Persistent stress pushes ecosystems past recovery points
- **Fire risk**: Extended heat-drought creates conditions for megafires

### Persistence Visualizations

- `Artifacts/consecutive_events.png`: 4-panel persistence analysis
- `Artifacts/consecutive_regional.png`: Regional 6+ month event changes

---

## Future Research Directions

1. ~~**Seasonal analysis**: Examine compound events by season~~ ✓ COMPLETED
2. ~~**Lower emissions scenarios**: Compare SSP1-2.6 vs SSP5-8.5~~ ✓ COMPLETED
3. ~~**Consecutive events**: Analyze multi-month compound droughts~~ ✓ COMPLETED
4. **Economic impact**: Overlay compound hazard maps with agricultural/population data
5. **Mechanism analysis**: Investigate why heat-drought correlation weakens under climate change

---

*Analysis conducted using CMIP6 data from ESGF, accessed via DOE data centers at Argonne National Lab and Oak Ridge National Lab.*
