"""
Phase 1 configuration — one-at-a-time HP sensitivity sweep.
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
# Fixed training settings (not swept)
# ---------------------------------------------------------------------------
FIXED = {
    "batch_size":     512,
    "num_epochs":     100,
    "patience":       12,
    "max_norm":       2.0,
    "batch_interval": 800,
}

# ---------------------------------------------------------------------------
# HP values to sweep (base value included in each list, but the runner
# will train the base only once and reuse it for all sweeps)
# ---------------------------------------------------------------------------
HP_VALUES = {
    "d_model":         [4, 8, 16, 24, 32, 48, 64, 128, 256],
    "d_ff_multiplier": [1, 2, 3, 4, 6, 8],
    "num_layers":      [1, 2, 4, 6, 8, 16],
    "learning_rate":   [0.0002, 0.0004, 0.0008, 0.001, 0.005],
    "dropout":         [0.0, 0.2, 0.5, 0.8],
    "weight_decay":    [0.0, 0.00001, 0.0001, 0.001, 0.01],
    "lr_decay":        [0.90, 0.93, 0.96, 0.98, 1.0],
}

# ---------------------------------------------------------------------------
# Base (paper-default) HP values — all other HPs are held at these
# ---------------------------------------------------------------------------
DEFAULTS = {
    "d_model":         32,
    "d_ff_multiplier": 4,
    "num_layers":      4,
    "num_heads":       8,
    "learning_rate":   0.0002,
    "dropout":         0.2,
    "weight_decay":    0.0001,
    "lr_decay":        0.96,
}
