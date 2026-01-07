#!/usr/bin/env python3
"""
Temperature projection analysis using Globus for data transfer from Argonne.

Globus provides reliable, high-performance transfers for large climate datasets.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import requests
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import globus_sdk
from globus_sdk.scopes import TransferScopes


# ESGF Search API
ESGF_SEARCH = "https://esgf-node.llnl.gov/esg-search/search"

# Argonne ALCF Globus endpoint (from ESGF URLs)
ARGONNE_ENDPOINT = "8896f38e-68d1-4708-bce4-b1b3a3405809"

# Analysis configuration
MODELS = ["MPI-ESM1-2-LR", "CanESM5", "ACCESS-ESM1-5"]
SCENARIOS = ["ssp126", "ssp585"]
VARIABLE = "tas"
TABLE = "Amon"

# Local data directory
DATA_DIR = Path("data/cmip6")


def get_globus_auth():
    """Authenticate with Globus using native app flow."""
    client_id = "61338d24-54d5-408f-a10d-66c06b59f6d2"  # Globus CLI client ID

    client = globus_sdk.NativeAppAuthClient(client_id)
    client.oauth2_start_flow(requested_scopes=TransferScopes.all)

    authorize_url = client.oauth2_get_authorize_url()
    print(f"\nPlease visit this URL to authenticate:\n{authorize_url}\n")

    auth_code = input("Enter the authorization code: ").strip()

    token_response = client.oauth2_exchange_code_for_tokens(auth_code)
    transfer_tokens = token_response.by_resource_server["transfer.api.globus.org"]

    return globus_sdk.TransferClient(
        authorizer=globus_sdk.AccessTokenAuthorizer(transfer_tokens["access_token"])
    )


def search_esgf(source_id: str, experiment_id: str) -> list[dict]:
    """Search ESGF for Globus URLs at Argonne."""
    params = {
        "project": "CMIP6",
        "source_id": source_id,
        "experiment_id": experiment_id,
        "variable_id": VARIABLE,
        "table_id": TABLE,
        "data_node": "eagle.alcf.anl.gov",
        "type": "File",
        "format": "application/solr+json",
        "latest": "true",
        "limit": 100,
    }

    response = requests.get(ESGF_SEARCH, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    # Group by member_id and extract Globus paths
    by_member = {}
    for doc in data.get("response", {}).get("docs", []):
        mem = doc.get("member_id", ["unknown"])[0]
        if mem not in by_member:
            by_member[mem] = []

        for url_entry in doc.get("url", []):
            parts = url_entry.split("|")
            if len(parts) >= 3 and parts[2] == "Globus":
                # Format: globus:<endpoint>/<path>
                globus_uri = parts[0]
                if globus_uri.startswith("globus:"):
                    # Extract path after endpoint ID
                    path = globus_uri.split("/", 1)[1] if "/" in globus_uri else ""
                    path = "/" + path
                    by_member[mem].append({
                        "path": path,
                        "filename": doc.get("title", path.split("/")[-1]),
                    })

    if not by_member:
        return [], None

    # Prefer r1i1p1f1
    chosen = "r1i1p1f1" if "r1i1p1f1" in by_member else sorted(by_member.keys())[0]
    return by_member[chosen], chosen


def download_with_globus(transfer_client, files: list[dict], dest_dir: Path) -> list[Path]:
    """Download files from Argonne using Globus."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    # For local transfers, we need a Globus Connect Personal endpoint
    # This is a simplified version - in practice you'd set up GCP
    print(f"  Would transfer {len(files)} files to {dest_dir}")
    print("  (Globus transfer requires Globus Connect Personal on your machine)")

    # For now, return empty - we'll need GCP setup
    return []


def main():
    print("=" * 70)
    print("Temperature Analysis via Globus (Argonne ALCF)")
    print("=" * 70)

    # First, let's just show what data is available
    print("\nSearching ESGF for data at Argonne...")

    available = {}
    for model in MODELS:
        available[model] = {}
        for scenario in SCENARIOS:
            files, member = search_esgf(model, scenario)
            if files:
                available[model][scenario] = {
                    "files": files,
                    "member": member,
                }
                print(f"  {model} / {scenario}: {len(files)} files ({member})")
            else:
                print(f"  {model} / {scenario}: no data")

    # Show sample Globus paths
    print("\n" + "=" * 70)
    print("Sample Globus transfer paths:")
    print("=" * 70)
    print(f"\nSource endpoint: {ARGONNE_ENDPOINT}")
    print("Sample files:")

    for model, scenarios in available.items():
        for scenario, info in scenarios.items():
            if info["files"]:
                f = info["files"][0]
                print(f"  {f['path']}")
                break
        break

    print("\n" + "=" * 70)
    print("To use Globus transfers:")
    print("=" * 70)
    print("""
1. Install Globus Connect Personal: https://www.globus.org/globus-connect-personal

2. Set up your local endpoint and note its endpoint ID

3. Use the Globus web app or CLI to transfer:
   globus transfer {src_endpoint}:{src_path} {dest_endpoint}:{dest_path}

4. Or run this script with --transfer flag (requires GCP setup)
""".format(src_endpoint=ARGONNE_ENDPOINT[:8] + "..."))

    # Check if user wants to proceed with auth
    print("\nWould you like to authenticate with Globus now? (y/n): ", end="")
    try:
        response = input().strip().lower()
        if response == 'y':
            print("\nAuthenticating with Globus...")
            tc = get_globus_auth()

            # List user's endpoints
            print("\nYour Globus endpoints:")
            for ep in tc.endpoint_search(filter_scope="my-endpoints"):
                print(f"  {ep['display_name']}: {ep['id']}")
    except EOFError:
        print("\n(Running non-interactively, skipping auth)")


if __name__ == "__main__":
    main()
