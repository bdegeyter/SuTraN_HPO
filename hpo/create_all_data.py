"""Create preprocessed tensor datasets for specified event logs.

Usage:
    python -m hpo.create_all_data                  # all 3 phase-1 datasets
    python -m hpo.create_all_data --datasets BPIC17_DR BAC_adj
"""
import argparse
import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def create_dataset(name: str):
    """Dispatch to the appropriate data creation function."""
    if name == "BPIC17_DR":
        from data_creation.create_BPIC17_DR_data import construct_BPIC17_DR_datasets
        print(f"Creating {name} ...")
        construct_BPIC17_DR_datasets()
    elif name == "BPIC17_OG":
        from data_creation.create_BPIC17_OG_data import construct_BPIC17_datasets
        print(f"Creating {name} ...")
        construct_BPIC17_datasets()
    elif name == "BPIC19":
        from data_creation.create_BPIC19_data import construct_BPIC19_datasets
        print(f"Creating {name} ...")
        construct_BPIC19_datasets()
    elif name == "BAC_adj":
        from data_creation.create_BAC_data_adj import construct_BAC_adj_datasets
        print(f"Creating {name} ...")
        construct_BAC_adj_datasets()
    elif name == "BAC_OG":
        from data_creation.create_BAC_data_OG import construct_BAC_OG_datasets
        print(f"Creating {name} ...")
        construct_BAC_OG_datasets()
    elif name == "BAC_dep":
        from data_creation.create_BAC_data_dep import construct_BAC_dep_datasets
        print(f"Creating {name} ...")
        construct_BAC_dep_datasets()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    print(f"  Done: {name}")


if __name__ == "__main__":
    from hpo.config import DATASET_REGISTRY

    all_datasets = list(DATASET_REGISTRY.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=all_datasets,
                        help="Which datasets to create")
    args = parser.parse_args()

    for ds in args.datasets:
        create_dataset(ds)

    print("\nAll datasets created.")
