"""
Phase 3 configuration — SMAC3 Bayesian optimisation.
"""

# ConfigSpace is imported lazily inside build_configspace() so that this
# module can be imported on machines where ConfigSpace is not installed
# (e.g. when running generate_slurm.py locally).

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
# Fixed training settings (unchanged from phase 2)
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
# Total number of new trials SMAC will run (not counting warm-start).
# ---------------------------------------------------------------------------
NUM_TRIALS = 200

# ---------------------------------------------------------------------------
# Performance capping for warm-start data.  Composite objective values above
# this percentile are clipped to the threshold so the RF surrogate focuses
# on the competitive region rather than fitting catastrophic outliers.
# Set to None to disable.
# ---------------------------------------------------------------------------
WARMSTART_CAP_PERCENTILE = 60

# ---------------------------------------------------------------------------
# Valid d_model values.  Must be multiples of 4 so num_heads = d_model // 4
# divides evenly.  Kept identical to phase 2 for a consistent search space.
# ---------------------------------------------------------------------------
D_MODEL_CHOICES = [8, 16, 32, 48, 64, 96]


# ---------------------------------------------------------------------------
# ConfigSpace definition (shared with warm-start clamping and SMAC facade)
# ---------------------------------------------------------------------------
def build_configspace():
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.hyperparameters import (
        OrdinalHyperparameter,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
    )
    cs = ConfigurationSpace(seed=42)
    cs.add_hyperparameters([
        OrdinalHyperparameter("d_model",             sequence=D_MODEL_CHOICES),
        UniformIntegerHyperparameter("num_layers",      lower=2, upper=8),
        UniformIntegerHyperparameter("d_ff_multiplier", lower=2, upper=8),
        UniformFloatHyperparameter("learning_rate",   lower=1e-5, upper=1e-2, log=True),
        UniformFloatHyperparameter("dropout",         lower=0.0,  upper=0.7),
        UniformFloatHyperparameter("weight_decay",    lower=1e-5, upper=1e-2, log=True),
    ])
    return cs
