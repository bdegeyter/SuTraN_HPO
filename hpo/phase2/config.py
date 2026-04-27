"""
Phase 2 configuration — LHS-seeded random search over all HPs jointly.
"""

# ---------------------------------------------------------------------------
# Dataset registry: CLI name -> folder name on disk
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    "BPIC17_DR": "BPIC_17_DR",
    "BPIC17_OG": "BPIC_17",
    "BPIC19":    "BPIC_19",
    "BAC_adj":   "BAC_adj",
    "BAC_OG":    "BAC_OG",
    "BAC_dep":   "BAC_dep",
}

# ---------------------------------------------------------------------------
# Fixed training settings
# ---------------------------------------------------------------------------
FIXED = {
    "batch_size":     512,
    "num_epochs":     100,
    "patience":       12,
    "max_norm":       2.0,
    "batch_interval": 800,
    "lr_decay":       0.96,
}

# ---------------------------------------------------------------------------
# Search space
#
# Two formats are supported:
#
#   List  — categorical / fixed set of values (original behaviour)
#       "dropout": [0.0, 0.2, 0.5]
#
#   Dict  — continuous range mapped through LHS unit-interval samples
#       {"type": "float", "low": 0.0,  "high": 0.7}   # linear
#       {"type": "log",   "low": 1e-5, "high": 1e-2}   # log-uniform
#       {"type": "int",   "low": 1,    "high": 10}      # integer-uniform
#
# ---------------------------------------------------------------------------
HP_SEARCH_SPACE = {
    # Categorical — must be a multiple of num_heads so keep as discrete set
    "d_model":         [8, 16, 32, 48, 64, 96],
    "num_layers":      {"type": "int",   "low": 2,    "high": 8},
    "d_ff_multiplier": {"type": "int",   "low": 2,    "high": 8},    
    "learning_rate":   {"type": "log",   "low": 1e-5, "high": 1e-2},
    "dropout":         {"type": "float", "low": 0.0,  "high": 0.7},
    "weight_decay":    {"type": "log",   "low": 1e-5, "high": 1e-2},
}

# ---------------------------------------------------------------------------
# LHS / trial settings
# ---------------------------------------------------------------------------
LHS_SEED   = 42      # seed used to generate the LHS plan (fixed for reproducibility)
NUM_TRIALS = 100     # total number of trials in the LHS plan
NUM_PARTS  = 4       # number of SLURM jobs the search is split across
                     # => each part runs NUM_TRIALS // NUM_PARTS trials