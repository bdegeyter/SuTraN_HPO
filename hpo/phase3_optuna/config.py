"""
Phase 3 configuration — Optuna TPE + Hyperband.

Optuna distributions are imported lazily inside get_distributions() so
this module can be imported on machines without Optuna installed (e.g.
when running generate_slurm.py locally).
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
# Fixed training settings.
# Note: no 'patience' here — Hyperband handles early stopping instead.
# ---------------------------------------------------------------------------
FIXED = {
    "batch_size":     512,
    "max_norm":       2.0,
    "batch_interval": 800,
    "lr_decay":       0.96,
}

# ---------------------------------------------------------------------------
# Total trial budget (pruned + complete combined).
# With Hyperband (eta=3, rungs at 15/45/100), roughly 1/9 of trials
# will run all 100 epochs; the rest are pruned early.
# ---------------------------------------------------------------------------
NUM_TRIALS = 300

# ---------------------------------------------------------------------------
# Performance capping for warm-start data.  Composite objective values above
# this percentile are clipped to the threshold so the TPE surrogate focuses
# on the competitive region rather than fitting catastrophic outliers.
# Set to None to disable.
# ---------------------------------------------------------------------------
WARMSTART_CAP_PERCENTILE = 60

# ---------------------------------------------------------------------------
# Valid d_model values — identical to phase3_smac for a consistent space.
# ---------------------------------------------------------------------------
D_MODEL_CHOICES = [8, 16, 32, 48, 64, 96]

# ---------------------------------------------------------------------------
# Hyperband pruner settings.
#   min_resource      = 15  → grace period at epoch 15 (BAC training
#                             trajectories are noisy at epoch 5; epoch 15
#                             gives a more reliable signal)
#   max_resource      = 100 → must equal num_epochs (full training budget)
#   reduction_factor  = 3   → standard eta; rungs at epochs 15, 45, 100
#                             (Li et al., 2018, JMLR)
# ---------------------------------------------------------------------------
HYPERBAND_MIN_RESOURCE    = 15
HYPERBAND_MAX_RESOURCE    = 100   # keep in sync with FIXED num_epochs in phase3_smac
HYPERBAND_REDUCTION_FACTOR = 3

# ---------------------------------------------------------------------------
# TPE sampler: 0 startup trials — warm-start observations already in the
# study, so the surrogate is ready immediately.
# ---------------------------------------------------------------------------
N_STARTUP_TRIALS = 0


# ---------------------------------------------------------------------------
# Optuna distributions (shared between warm-start injection and suggest_*)
# ---------------------------------------------------------------------------
def get_distributions():
    """Return the Optuna distribution objects for the 6 tuned HPs.

    Imported lazily so config.py can be used without Optuna installed.
    """
    from optuna.distributions import (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )
    return {
        "d_model":         CategoricalDistribution(D_MODEL_CHOICES),
        "num_layers":      IntDistribution(2, 8),
        "d_ff_multiplier": IntDistribution(2, 8),
        "learning_rate":   FloatDistribution(1e-5, 1e-2, log=True),
        "dropout":         FloatDistribution(0.0, 0.7),
        "weight_decay":    FloatDistribution(1e-5, 1e-2, log=True),
    }
