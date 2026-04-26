"""
One-time backfill script.

Finds every results.pkl produced before weight_decay / lr_decay were swept
and injects the fixed default values that were used during those runs:
  - weight_decay = 0.0001  (was FIXED["weight_decay"])
  - lr_decay     = 0.96    (was FIXED["lr_decay_factor"])

Run once from the project root:
    python -m hpo.phase1.backfill_pkl_defaults
or with an explicit results dir:
    python -m hpo.phase1.backfill_pkl_defaults --results_dir /path/to/results
"""

import argparse
import glob
import os
import pickle

# defaults that were hard-coded when these experiments were run
BACKFILL_DEFAULTS = {
    "weight_decay": 0.0001,
    "lr_decay":     0.96,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="")
    args = p.parse_args()

    repo_root   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = args.results_dir or os.path.join(repo_root, "results")

    pattern   = os.path.join(results_dir, "**", "results.pkl")
    pkl_files = sorted(glob.glob(pattern, recursive=True))

    if not pkl_files:
        print(f"No results.pkl files found under {results_dir}")
        return

    updated = 0
    skipped = 0

    for path in pkl_files:
        with open(path, "rb") as f:
            data = pickle.load(f)

        needs_update = {k: v for k, v in BACKFILL_DEFAULTS.items() if k not in data}

        if not needs_update:
            skipped += 1
            continue

        data.update(needs_update)
        with open(path, "wb") as f:
            pickle.dump(data, f)

        print(f"Updated {path}  (+{list(needs_update.keys())}")
        updated += 1

    print(f"\nDone. Updated: {updated}  |  Already had keys: {skipped}")


if __name__ == "__main__":
    main()
