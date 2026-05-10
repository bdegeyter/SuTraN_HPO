"""
Rebuild summary.csv for a completed (or partial) phase 3 Optuna run.

Run from the project root after the SLURM job finishes (or at any point
for a partial summary of completed — non-pruned — trials):

    python -m hpo.phase3_optuna.rebuild_summary --model_type DA --dataset BAC_OG
    python -m hpo.phase3_optuna.rebuild_summary --model_type NDA --dataset BAC_OG

The summary is written to:
    results/sutran_<model_type>/<dataset>/optuna_search/summary.csv

Only fully-completed trials (those not pruned by Hyperband) are included.
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hpo.phase3_optuna.config import DATASET_REGISTRY
from hpo.phase3_optuna.run import rebuild_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type",  required=True, choices=["DA", "NDA"])
    p.add_argument("--dataset",     required=True, choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--results_dir", type=str, default="")
    return p.parse_args()


def main():
    args        = parse_args()
    repo_root   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = args.results_dir or os.path.join(repo_root, "results")

    optuna_search_dir = os.path.join(
        results_dir, f"sutran_{args.model_type}", args.dataset, "optuna_search",
    )
    rebuild_summary(optuna_search_dir)


if __name__ == "__main__":
    main()
