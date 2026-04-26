"""
Rebuild summary.csv for one or more (model_type, dataset) combinations
from all results.pkl files already on disk.

Run from the project root after merging results folders:
    python -m hpo.phase1.rebuild_summary --model_type DA --dataset BPIC17_DR
    python -m hpo.phase1.rebuild_summary --model_type DA --dataset BAC_OG
    python -m hpo.phase1.rebuild_summary --all
"""

import argparse
import csv
import glob
import os
import pickle
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hpo.phase1.config import DATASET_REGISTRY


def rebuild(results_dir, model_type, dataset):
    individual_dir = os.path.join(results_dir, f"sutran_{model_type}", dataset, "individual")
    pattern   = os.path.join(individual_dir, "*", "*", "*", "results.pkl")
    pkl_files = sorted(glob.glob(pattern))

    if not pkl_files:
        print(f"  No results.pkl files found under {individual_dir}")
        return

    rows = []
    for path in pkl_files:
        with open(path, "rb") as f:
            rows.append(pickle.load(f))

    summary_path = os.path.join(individual_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Written {len(rows)} rows → {summary_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["DA", "NDA"])
    p.add_argument("--dataset",    choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--all",        action="store_true",
                   help="Rebuild summary.csv for every (model_type, dataset) that has results.")
    p.add_argument("--results_dir", type=str, default="")
    args = p.parse_args()

    results_dir = args.results_dir or os.path.join(_REPO_ROOT, "results")

    if args.all:
        for mt in ("DA", "NDA"):
            for ds in DATASET_REGISTRY:
                individual_dir = os.path.join(results_dir, f"sutran_{mt}", ds, "individual")
                if os.path.isdir(individual_dir):
                    print(f"{mt} / {ds}")
                    rebuild(results_dir, mt, ds)
    else:
        if not args.model_type or not args.dataset:
            p.error("Provide --model_type and --dataset, or use --all")
        print(f"{args.model_type} / {args.dataset}")
        rebuild(results_dir, args.model_type, args.dataset)


if __name__ == "__main__":
    main()
