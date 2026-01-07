"""
LLM Interface for ESGF Data Discovery

Translates natural language queries into ESGF API calls using Claude.
"""

import json
import os
from dataclasses import dataclass

from .esgf_client import (
    ESGFClient,
    ESGFSearchResult,
    COMMON_VARIABLES,
    EXPERIMENTS,
    CMIP6_FACETS,
)

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


SYSTEM_PROMPT = """You are an expert assistant for discovering climate data from the Earth System Grid Federation (ESGF). Your role is to translate natural language queries into structured ESGF search parameters.

## About ESGF and CMIP6

ESGF hosts climate model output, primarily from the Coupled Model Intercomparison Project Phase 6 (CMIP6). Key concepts:

### Variables (variable_id)
Common atmospheric variables:
- tas: Near-Surface Air Temperature (2m)
- pr: Precipitation
- psl: Sea Level Pressure
- uas/vas: Near-surface wind components
- huss: Near-Surface Specific Humidity
- rsds/rlds: Surface radiation (shortwave/longwave down)
- clt: Total Cloud Cover

Ocean variables:
- tos: Sea Surface Temperature
- sos: Sea Surface Salinity
- zos: Sea Surface Height

### Experiments (experiment_id)
- historical: Historical simulations (1850-2014)
- piControl: Pre-industrial control
- SSP scenarios (future projections 2015-2100):
  - ssp119: Very low emissions (1.9 W/m²)
  - ssp126: Low emissions (2.6 W/m²)
  - ssp245: Medium emissions (4.5 W/m²)
  - ssp370: High emissions (7.0 W/m²)
  - ssp585: Very high emissions (8.5 W/m²)

### Tables (table_id) - determines frequency and realm
- Amon: Monthly atmosphere
- day: Daily atmosphere
- Omon: Monthly ocean
- Oday: Daily ocean
- 6hrLev: 6-hourly atmospheric levels
- fx: Fixed/static fields

### Models (source_id)
Major models include: CESM2, GFDL-ESM4, MPI-ESM1-2-HR, UKESM1-0-LL, EC-Earth3, NorESM2-LM, ACCESS-ESM1-5, MIROC6, CanESM5, IPSL-CM6A-LR

### Ensemble members (member_id)
Format: r<N>i<M>p<P>f<F> (e.g., r1i1p1f1)
- r: realization, i: initialization, p: physics, f: forcing

## Your Task

Given a user query, extract the relevant ESGF search parameters. Return a JSON object with these fields (omit fields that aren't specified):

{
    "variable_id": "string or array",
    "experiment_id": "string or array",
    "source_id": "string or array",
    "table_id": "string",
    "member_id": "string",
    "activity_id": "string",
    "explanation": "Brief explanation of your interpretation"
}

Be helpful in interpreting user intent:
- "temperature" usually means "tas" (near-surface air temperature)
- "rainfall" or "precipitation" means "pr"
- "worst-case scenario" or "high emissions" means "ssp585"
- "Paris-aligned" or "low emissions" usually means "ssp126"
- "monthly data" suggests table_id="Amon" for atmosphere or "Omon" for ocean
- If the user mentions a time period like "future" or "projections", they likely want SSP experiments
- If they mention "past" or "observed period", they likely want "historical"

Always include an "explanation" field describing how you interpreted the query.
"""


@dataclass
class QueryInterpretation:
    """Result of interpreting a natural language query."""

    params: dict
    explanation: str
    raw_response: str


@dataclass
class SearchResponse:
    """Combined result of query interpretation and search."""

    query: str
    interpretation: QueryInterpretation
    results: list[ESGFSearchResult]
    num_found: int
    summary: str


class ESGFAssistant:
    """LLM-powered assistant for ESGF data discovery."""

    def __init__(
        self,
        api_key: str | None = None,
        esgf_node: str = "llnl",
        model: str = "claude-sonnet-4-20250514",
    ):
        """
        Initialize the assistant.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            esgf_node: ESGF node to search
            model: Claude model to use
        """
        if not HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.esgf = ESGFClient(node=esgf_node)
        self.model = model

    def interpret_query(self, query: str) -> QueryInterpretation:
        """
        Use Claude to interpret a natural language query into ESGF parameters.

        Args:
            query: Natural language query about climate data

        Returns:
            QueryInterpretation with extracted parameters
        """
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Translate this query into ESGF search parameters:\n\n{query}",
                }
            ],
        )

        response_text = message.content[0].text

        # Extract JSON from response
        try:
            # Try to find JSON in the response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0]
            elif "{" in response_text:
                # Find the JSON object
                start = response_text.index("{")
                end = response_text.rindex("}") + 1
                json_str = response_text[start:end]
            else:
                json_str = response_text

            params = json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            # If we can't parse JSON, return empty params with the explanation
            params = {"explanation": f"Could not parse response: {response_text}"}

        explanation = params.pop("explanation", "No explanation provided")

        return QueryInterpretation(
            params=params,
            explanation=explanation,
            raw_response=response_text,
        )

    def search(
        self,
        query: str,
        limit: int = 20,
        include_summary: bool = True,
    ) -> SearchResponse:
        """
        Search ESGF using natural language.

        Args:
            query: Natural language query
            limit: Maximum results to return
            include_summary: Whether to generate a summary of results

        Returns:
            SearchResponse with interpretation and results
        """
        # Interpret the query
        interpretation = self.interpret_query(query)

        # Execute the search
        search_results = self.esgf.search(
            limit=limit,
            **interpretation.params,
        )

        # Generate summary
        summary = ""
        if include_summary and search_results["results"]:
            summary = self._generate_summary(
                query,
                interpretation,
                search_results,
            )

        return SearchResponse(
            query=query,
            interpretation=interpretation,
            results=search_results["results"],
            num_found=search_results["num_found"],
            summary=summary,
        )

    def _generate_summary(
        self,
        query: str,
        interpretation: QueryInterpretation,
        search_results: dict,
    ) -> str:
        """Generate a human-readable summary of search results."""
        results = search_results["results"]
        num_found = search_results["num_found"]

        # Collect unique values
        models = set(r.source_id for r in results if r.source_id)
        experiments = set(r.experiment_id for r in results if r.experiment_id)
        variables = set(r.variable_id for r in results if r.variable_id)

        lines = [
            f"Found {num_found} datasets matching your query.",
            f"",
            f"Interpretation: {interpretation.explanation}",
            f"",
            f"Search parameters: {json.dumps(interpretation.params, indent=2)}",
            f"",
        ]

        if models:
            lines.append(f"Models represented: {', '.join(sorted(models)[:10])}")
        if experiments:
            lines.append(f"Experiments: {', '.join(sorted(experiments))}")
        if variables:
            var_desc = [
                f"{v} ({COMMON_VARIABLES.get(v, 'unknown')})"
                for v in sorted(variables)[:5]
            ]
            lines.append(f"Variables: {', '.join(var_desc)}")

        return "\n".join(lines)

    def ask(self, question: str) -> str:
        """
        Ask a general question about ESGF/CMIP6 data.

        Args:
            question: Question about climate data, variables, experiments, etc.

        Returns:
            Answer from Claude
        """
        context = f"""
Available variables (partial list): {json.dumps(COMMON_VARIABLES, indent=2)}

Experiment descriptions: {json.dumps(EXPERIMENTS, indent=2)}

Search facets: {', '.join(CMIP6_FACETS)}
"""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=f"""You are an expert on ESGF and CMIP6 climate data. Answer questions helpfully and accurately.

{context}

Provide clear, informative answers. If you're not certain about something, say so.
""",
            messages=[{"role": "user", "content": question}],
        )

        return message.content[0].text


def create_assistant(api_key: str | None = None) -> ESGFAssistant:
    """
    Create an ESGF assistant instance.

    Args:
        api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)

    Returns:
        Configured ESGFAssistant
    """
    return ESGFAssistant(api_key=api_key)
