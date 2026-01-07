#!/usr/bin/env python3
"""
Example: Using the LLM interface to search ESGF with natural language.

Requires ANTHROPIC_API_KEY environment variable to be set.
"""

import sys
sys.path.insert(0, "src")

from esgf_llm import create_assistant


def main():
    # Create the assistant (uses ANTHROPIC_API_KEY from environment)
    assistant = create_assistant()

    print("=" * 70)
    print("ESGF Natural Language Search")
    print("=" * 70)

    # Example queries to demonstrate capabilities
    queries = [
        "Find monthly precipitation data from the high emissions scenario",
        "I need sea surface temperature from CESM2 for future projections",
        "What temperature data is available from the historical period?",
    ]

    for query in queries:
        print(f"\n{'─' * 70}")
        print(f"Query: {query}")
        print("─" * 70)

        response = assistant.search(query, limit=5)

        print(f"\n{response.summary}")

        print("\nSample results:")
        for i, result in enumerate(response.results[:3], 1):
            print(f"  {i}. {result.source_id} / {result.experiment_id} / {result.member_id}")

    # Demonstrate the ask functionality
    print(f"\n{'=' * 70}")
    print("Ask a question about ESGF/CMIP6")
    print("=" * 70)

    question = "What's the difference between SSP2-4.5 and SSP5-8.5 scenarios?"
    print(f"\nQuestion: {question}\n")
    answer = assistant.ask(question)
    print(f"Answer:\n{answer}")


if __name__ == "__main__":
    main()
