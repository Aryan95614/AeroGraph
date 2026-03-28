#!/usr/bin/env python3
"""Build the AeroGraph knowledge graph from extracted entities."""

from aerograph.extract import run_extraction
from aerograph.graph import build_graph


def main():
    print("Running entity extraction...")
    run_extraction()
    print("Building knowledge graph...")
    build_graph()
    print("Done.")


if __name__ == "__main__":
    main()
