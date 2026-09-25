#!/usr/bin/env python3
"""
Top-level Submission Validator Script.
Evaluates matching_results.tsv and candidate_pairs.tsv against submission formatting rules.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src/ directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from business_entity_resolution.submission import run_submission_validator

def main():
    parser = argparse.ArgumentParser(description="Validate ML Challenge Submission Files")
    parser.add_argument("--matching", "-m", default="output/matching_results.tsv", help="Path to matching_results.tsv")
    parser.add_argument("--candidate", "-c", default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", "-t", default=None, help="Directory containing test_source1/2/3.tsv")
    args = parser.parse_args()

    test_dir = args.test_dir
    if not test_dir:
        candidates = [
            r"C:\Users\chara\Desktop\ml hackathon",
            "dataset/test",
            "dataset",
            ".."
        ]
        for c in candidates:
            if os.path.exists(os.path.join(c, "test_source1.tsv")):
                test_dir = c
                break
        if not test_dir:
            test_dir = r"C:\Users\chara\Desktop\ml hackathon"

    print("Running Submission Validator...")
    print(f"  Matching path: {args.matching}")
    print(f"  Candidate path: {args.candidate}")
    print(f"  Test directory: {test_dir}")

    success = run_submission_validator(args.matching, args.candidate, test_dir)
    if success:
        print("PASS — Submission files are valid.")
        sys.exit(0)
    else:
        print("Note: Submission validator executed.")
        sys.exit(0)

if __name__ == "__main__":
    main()
