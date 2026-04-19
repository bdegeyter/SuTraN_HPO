"""Collect all Phase 1 trial results into a single DataFrame.

Scans the results/ directory for summary_metrics.csv files and
concatenates them, adding model/dataset/experiment columns.

Usage:
    python -m hpo.collect_results
    python -m hpo.collect_results --results_dir results --output phase1_all.csv
"""
import argparse
import os
import sys
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def collect(results_dir: str, experiment_type: str = "individual") -> pd.DataFrame:
    """Walk results tree and collect all summary CSVs."""
    rows = []
    for model_dir in sorted(os.listdir(results_dir)):
        model_path = os.path.join(results_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        for dataset_dir in sorted(os.listdir(model_path)):
            dataset_path = os.path.join(model_path, dataset_dir)
            if not os.path.isdir(dataset_path):
                continue
            exp_path = os.path.join(dataset_path, experiment_type)
            if not os.path.isdir(exp_path):
                continue
            for trial_dir in sorted(os.listdir(exp_path)):
                csv_path = os.path.join(exp_path, trial_dir, "summary_metrics.csv")
                if os.path.isfile(csv_path):
                    df = pd.read_csv(csv_path)
                    df["model_type"] = model_dir
                    df["dataset"] = dataset_dir
                    df["trial_name"] = trial_dir
                    rows.append(df)

    if not rows:
        print("No results found.")
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--experiment_type", default="individual")
    parser.add_argument("--output", default="phase1_all_results.csv")
    args = parser.parse_args()

    df = collect(args.results_dir, args.experiment_type)
    if not df.empty:
        df.to_csv(args.output, index=False)
        print(f"Collected {len(df)} trials → {args.output}")
        print(f"\nBreakdown:")
        print(df.groupby(["model_type", "dataset", "hp_name"]).size().to_string())
