"""
Rebuild summary.csv for a completed (or partial) phase 3 SMAC run.

Run from the project root after the SLURM job finishes (or at any point to
get a partial summary):

    python -m hpo.phase3_smac.rebuild_summary --model_type DA --dataset BAC_OG
    python -m hpo.phase3_smac.rebuild_summary --model_type NDA --dataset BAC_OG

The summary is written to:
    results/sutran_<model_type>/<dataset>/smac_search/summary.csv
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

from hpo.phase3_smac.config import DATASET_REGISTRY


def rebuild(results_dir: str, model_type: str, dataset: str):
    smac_dir  = os.path.join(results_dir, f"sutran_{model_type}", dataset, "smac_search")
    pattern   = os.path.join(smac_dir, "trial_*", "results.pkl")
    pkl_files = sorted(glob.glob(pattern))

    if not pkl_files:
        print(f"  No results.pkl files found under {smac_dir}")
        return

    rows = []
    for path in pkl_files:
        with open(path, "rb") as f:
            r = pickle.load(f)
        rows.append({k: v for k, v in r.items() if k != "smac_config_values"})

    rows.sort(key=lambda r: r["trial_idx"])

    summary_path = os.path.join(smac_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written {len(rows)} rows → {summary_path}")


def main():
    p = argparse.ArgumentParser(
        description="Rebuild phase 3 summary.csv from results.pkl files on disk.",
    )
    p.add_argument("--model_type",  required=True, choices=["DA", "NDA"])
    p.add_argument("--dataset",     required=True, choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--results_dir", type=str,      default="")
    args = p.parse_args()

    repo_root   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = args.results_dir or os.path.join(repo_root, "results")

    print(f"Rebuilding summary for {args.model_type}/{args.dataset}...")
    rebuild(results_dir, args.model_type, args.dataset)


if __name__ == "__main__":
    main()
