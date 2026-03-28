#!/usr/bin/env python3
"""Run the AeroGraph evaluation suite."""

from aerograph.eval import run_evaluation


def main():
    results = run_evaluation()
    print(f"Evaluation complete. Results: {results}")


if __name__ == "__main__":
    main()
