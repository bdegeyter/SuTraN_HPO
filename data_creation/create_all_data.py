"""Run all dataset creation scripts sequentially from the repo root.

Usage (from repo root):
    python data_creation/create_all_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_creation.create_BPIC17_DR_data import construct_BPIC17_DR_datasets
from data_creation.create_BPIC17_OG_data import construct_BPIC17_datasets
from data_creation.create_BPIC19_data import construct_BPIC19_datasets
from data_creation.create_BAC_data_OG import construct_BAC_OG_datasets
from data_creation.create_BAC_data_adj import construct_BAC_adj_datasets
from data_creation.create_BAC_data_dep import construct_BAC_dep_datasets

print("Creating BPIC17_DR dataset...")
construct_BPIC17_DR_datasets()

print("Creating BPIC17_OG dataset...")
construct_BPIC17_datasets()

print("Creating BPIC19 dataset...")
construct_BPIC19_datasets()

print("Creating BAC_OG dataset...")
construct_BAC_OG_datasets()

print("Creating BAC_adj dataset...")
construct_BAC_adj_datasets()

print("Creating BAC_dep dataset...")
construct_BAC_dep_datasets()

print("All datasets created successfully.")
