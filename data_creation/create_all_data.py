
# Run all dataset creation scripts sequentially

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_creation.create_BPIC17_DR_data import construct_BPIC17_DR_datasets
from data_creation.create_BPIC17_OG_data import construct_BPIC17_datasets
from data_creation.create_BPIC19_data import construct_BPIC19_datasets
from data_creation.create_BAC_data_OG import construct_BAC_OG_datasets

print("Creating BPIC17_DR dataset...")
construct_BPIC17_DR_datasets()

print("Creating BPIC17_OG dataset...")
construct_BPIC17_datasets()

print("Creating BPIC19 dataset...")
construct_BPIC19_datasets()

print("Creating BAC_OG dataset...")
construct_BAC_OG_datasets()

print("All datasets created successfully.")
