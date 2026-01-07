#!/usr/bin/env python3
"""Test the ESGF client with sample queries."""

import sys
sys.path.insert(0, "src")

from esgf_llm import ESGFClient


def main():
    client = ESGFClient(node="llnl")

    print("=" * 60)
    print("ESGF Client Test")
    print("=" * 60)

    # Test 1: List some models
    print("\n1. Available models (top 10):")
    models = client.list_models()[:10]
    for model, count in models:
        print(f"   {model}: {count} datasets")

    # Test 2: List experiments for a specific model
    print("\n2. Experiments available for CESM2:")
    experiments = client.list_experiments(source_id="CESM2")[:10]
    for exp, count in experiments:
        desc = client.describe_experiment(exp)
        print(f"   {exp}: {count} datasets - {desc}")

    # Test 3: Search for specific data
    print("\n3. Searching for monthly temperature (tas) from ssp585:")
    results = client.search(
        variable_id="tas",
        experiment_id="ssp585",
        table_id="Amon",
        limit=5,
    )
    print(f"   Found {results['num_found']} total datasets")
    print("   First 5 results:")
    for r in results["results"]:
        print(f"   - {r.source_id} / {r.member_id}")
        if opendap := r.get_opendap_url():
            print(f"     OPeNDAP: {opendap[:80]}...")

    # Test 4: List variables for a model/experiment combo
    print("\n4. Variables available for GFDL-ESM4 ssp585 (monthly atmos):")
    variables = client.list_variables(
        source_id="GFDL-ESM4",
        experiment_id="ssp585",
        table_id="Amon",
    )[:15]
    for var, count in variables:
        desc = client.describe_variable(var)
        print(f"   {var}: {desc}")

    print("\n" + "=" * 60)
    print("Tests completed successfully!")


if __name__ == "__main__":
    main()
