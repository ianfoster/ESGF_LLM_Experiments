"""
ESGF Search API Client

Provides programmatic access to the Earth System Grid Federation search API
for discovering CMIP6 and other climate datasets.
"""

from dataclasses import dataclass, field
from typing import Any
import requests


# ESGF index nodes - any can be used for search
ESGF_NODES = {
    "llnl": "https://esgf-node.llnl.gov/esg-search/search",
    "dkrz": "https://esgf-data.dkrz.de/esg-search/search",
    "ipsl": "https://esgf-node.ipsl.upmc.fr/esg-search/search",
    "ceda": "https://esgf-index1.ceda.ac.uk/esg-search/search",
}

# Common CMIP6 facets for filtering
CMIP6_FACETS = [
    "project",
    "mip_era",
    "activity_id",
    "institution_id",
    "source_id",        # model name
    "experiment_id",    # e.g., historical, ssp585
    "member_id",        # ensemble member (e.g., r1i1p1f1)
    "table_id",         # output frequency/realm (e.g., Amon, day, Omon)
    "variable_id",      # variable name (e.g., tas, pr)
    "grid_label",
    "variant_label",
]

# Common variable descriptions for reference
COMMON_VARIABLES = {
    "tas": "Near-Surface Air Temperature",
    "pr": "Precipitation",
    "psl": "Sea Level Pressure",
    "ts": "Surface Temperature",
    "uas": "Eastward Near-Surface Wind",
    "vas": "Northward Near-Surface Wind",
    "huss": "Near-Surface Specific Humidity",
    "rsds": "Surface Downwelling Shortwave Radiation",
    "rlds": "Surface Downwelling Longwave Radiation",
    "tos": "Sea Surface Temperature",
    "sos": "Sea Surface Salinity",
    "zos": "Sea Surface Height Above Geoid",
    "siconc": "Sea Ice Area Fraction",
    "clt": "Total Cloud Cover Percentage",
    "evspsbl": "Evaporation Including Sublimation and Transpiration",
}

# CMIP6 experiment descriptions
EXPERIMENTS = {
    "historical": "Historical simulation (1850-2014)",
    "piControl": "Pre-industrial control simulation",
    "ssp119": "SSP1-1.9: Very low emissions, 1.9W/m2 forcing by 2100",
    "ssp126": "SSP1-2.6: Low emissions, 2.6W/m2 forcing by 2100",
    "ssp245": "SSP2-4.5: Medium emissions, 4.5W/m2 forcing by 2100",
    "ssp370": "SSP3-7.0: High emissions, 7.0W/m2 forcing by 2100",
    "ssp585": "SSP5-8.5: Very high emissions, 8.5W/m2 forcing by 2100",
    "abrupt-4xCO2": "Abrupt quadrupling of CO2",
    "1pctCO2": "1% per year CO2 increase",
}


@dataclass
class ESGFSearchResult:
    """Represents a single dataset from ESGF search results."""

    id: str
    title: str
    instance_id: str
    data_node: str
    variable_id: str | None = None
    experiment_id: str | None = None
    source_id: str | None = None
    member_id: str | None = None
    table_id: str | None = None
    activity_id: str | None = None
    institution_id: str | None = None
    url: list[str] = field(default_factory=list)
    size: int | None = None

    @classmethod
    def from_doc(cls, doc: dict[str, Any]) -> "ESGFSearchResult":
        """Create from ESGF search response document."""
        # URLs come as list of "url|mime|service" strings
        urls = []
        for url_entry in doc.get("url", []):
            if isinstance(url_entry, str):
                urls.append(url_entry.split("|")[0])

        return cls(
            id=doc.get("id", ""),
            title=doc.get("title", ""),
            instance_id=doc.get("instance_id", ""),
            data_node=doc.get("data_node", ""),
            variable_id=doc.get("variable_id", [None])[0] if doc.get("variable_id") else None,
            experiment_id=doc.get("experiment_id", [None])[0] if doc.get("experiment_id") else None,
            source_id=doc.get("source_id", [None])[0] if doc.get("source_id") else None,
            member_id=doc.get("member_id", [None])[0] if doc.get("member_id") else None,
            table_id=doc.get("table_id", [None])[0] if doc.get("table_id") else None,
            activity_id=doc.get("activity_id", [None])[0] if doc.get("activity_id") else None,
            institution_id=doc.get("institution_id", [None])[0] if doc.get("institution_id") else None,
            url=urls,
            size=doc.get("size"),
        )

    def get_opendap_url(self) -> str | None:
        """Extract OPeNDAP URL if available."""
        for url in self.url:
            if "opendap" in url.lower() or "dodsC" in url:
                return url
        return None

    def get_http_url(self) -> str | None:
        """Extract HTTP download URL if available."""
        for url in self.url:
            if "HTTPServer" in url or url.endswith(".nc"):
                return url
        return None


class ESGFClient:
    """Client for the ESGF Search API."""

    def __init__(self, node: str = "llnl", timeout: int = 30):
        """
        Initialize ESGF client.

        Args:
            node: ESGF index node to use (llnl, dkrz, ipsl, ceda)
            timeout: Request timeout in seconds
        """
        if node not in ESGF_NODES:
            raise ValueError(f"Unknown node: {node}. Available: {list(ESGF_NODES.keys())}")

        self.base_url = ESGF_NODES[node]
        self.timeout = timeout
        self.session = requests.Session()

    def search(
        self,
        project: str = "CMIP6",
        variable_id: str | list[str] | None = None,
        experiment_id: str | list[str] | None = None,
        source_id: str | list[str] | None = None,
        member_id: str | None = None,
        table_id: str | None = None,
        activity_id: str | None = None,
        institution_id: str | None = None,
        frequency: str | None = None,
        limit: int = 50,
        offset: int = 0,
        latest: bool = True,
        distrib: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Search ESGF for datasets matching criteria.

        Args:
            project: Project name (default: CMIP6)
            variable_id: Variable(s) to search for (e.g., "tas", "pr")
            experiment_id: Experiment(s) (e.g., "historical", "ssp585")
            source_id: Model name(s) (e.g., "CESM2", "GFDL-ESM4")
            member_id: Ensemble member (e.g., "r1i1p1f1")
            table_id: Output table (e.g., "Amon" for monthly atmosphere)
            activity_id: MIP activity (e.g., "CMIP", "ScenarioMIP")
            institution_id: Modeling center
            frequency: Temporal frequency (mon, day, 6hr, etc.)
            limit: Maximum results to return
            offset: Result offset for pagination
            latest: Only return latest version
            distrib: Search across all federated nodes
            **kwargs: Additional facet constraints

        Returns:
            Dictionary with 'results', 'num_found', and 'facets'
        """
        params: dict[str, Any] = {
            "project": project,
            "type": "Dataset",
            "format": "application/solr+json",
            "limit": limit,
            "offset": offset,
            "latest": str(latest).lower(),
            "distrib": str(distrib).lower(),
        }

        # Add optional parameters
        if variable_id:
            params["variable_id"] = variable_id if isinstance(variable_id, list) else [variable_id]
        if experiment_id:
            params["experiment_id"] = experiment_id if isinstance(experiment_id, list) else [experiment_id]
        if source_id:
            params["source_id"] = source_id if isinstance(source_id, list) else [source_id]
        if member_id:
            params["member_id"] = member_id
        if table_id:
            params["table_id"] = table_id
        if activity_id:
            params["activity_id"] = activity_id
        if institution_id:
            params["institution_id"] = institution_id
        if frequency:
            params["frequency"] = frequency

        # Add any additional kwargs as facets
        params.update(kwargs)

        response = self.session.get(self.base_url, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        response_data = data.get("response", {})

        results = [
            ESGFSearchResult.from_doc(doc)
            for doc in response_data.get("docs", [])
        ]

        return {
            "results": results,
            "num_found": response_data.get("numFound", 0),
            "facets": data.get("facet_counts", {}).get("facet_fields", {}),
        }

    def get_facet_values(
        self,
        facet: str,
        project: str = "CMIP6",
        limit: int = 100,
        **constraints: Any,
    ) -> list[tuple[str, int]]:
        """
        Get available values for a facet with counts.

        Args:
            facet: Facet name (e.g., "source_id", "variable_id")
            project: Project to search within
            limit: Maximum values to return
            **constraints: Additional facet constraints to filter by

        Returns:
            List of (value, count) tuples
        """
        params: dict[str, Any] = {
            "project": project,
            "type": "Dataset",
            "format": "application/solr+json",
            "limit": 0,
            "facets": facet,
            "distrib": "true",
            "latest": "true",
        }
        params.update(constraints)

        response = self.session.get(self.base_url, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        facet_data = data.get("facet_counts", {}).get("facet_fields", {}).get(facet, [])

        # Facet data comes as [value1, count1, value2, count2, ...]
        results = []
        for i in range(0, len(facet_data), 2):
            if i + 1 < len(facet_data):
                results.append((facet_data[i], facet_data[i + 1]))

        return results[:limit]

    def list_models(self, experiment_id: str | None = None) -> list[tuple[str, int]]:
        """List available models (source_id) with dataset counts."""
        constraints = {}
        if experiment_id:
            constraints["experiment_id"] = experiment_id
        return self.get_facet_values("source_id", **constraints)

    def list_variables(
        self,
        source_id: str | None = None,
        experiment_id: str | None = None,
        table_id: str | None = None,
    ) -> list[tuple[str, int]]:
        """List available variables with dataset counts."""
        constraints = {}
        if source_id:
            constraints["source_id"] = source_id
        if experiment_id:
            constraints["experiment_id"] = experiment_id
        if table_id:
            constraints["table_id"] = table_id
        return self.get_facet_values("variable_id", **constraints)

    def list_experiments(self, source_id: str | None = None) -> list[tuple[str, int]]:
        """List available experiments with dataset counts."""
        constraints = {}
        if source_id:
            constraints["source_id"] = source_id
        return self.get_facet_values("experiment_id", **constraints)

    def describe_variable(self, variable_id: str) -> str:
        """Get human-readable description of a variable."""
        return COMMON_VARIABLES.get(variable_id, f"Unknown variable: {variable_id}")

    def describe_experiment(self, experiment_id: str) -> str:
        """Get human-readable description of an experiment."""
        return EXPERIMENTS.get(experiment_id, f"Unknown experiment: {experiment_id}")
