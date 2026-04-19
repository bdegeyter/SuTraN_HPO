"""Experiment configuration for HPO trials.

Single source of truth for all hyperparameters, dataset info, and
experiment metadata. Configs are loaded from CLI arguments.
"""
import argparse
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Dataset registry – maps dataset key to the log_name used by preprocessing
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    "BPIC17_DR": "BPIC_17_DR",
    "BPIC17_OG": "BPIC_17",
    "BPIC19": "BPIC_19",
    "BAC_adj": "BAC_adj",
    "BAC_OG": "BAC_OG",
    "BAC_dep": "BAC_dep",
}


@dataclass
class ExperimentConfig:
    # ---- Experiment metadata ----
    model_type: str = "DA"           # "DA" or "NDA"
    dataset: str = "BPIC17_DR"       # key into DATASET_REGISTRY
    experiment_type: str = "individual"  # "individual", "lhs", "tpe"
    hp_name: str = ""                # which HP is varied (for individual)
    seed: int = 42

    # ---- Model hyperparameters ----
    d_model: int = 32
    num_heads: int = 8
    d_ff_multiplier: int = 4         # d_ff = d_model * d_ff_multiplier
    num_layers: int = 4              # sets both encoder and decoder layers
    dropout: float = 0.2

    # ---- Training hyperparameters ----
    learning_rate: float = 0.0002
    weight_decay: float = 0.0001
    lr_decay_factor: float = 0.96
    batch_size: int = 512
    num_epochs: int = 100
    patience: int = 24
    max_norm: float = 2.0
    batch_interval: int = 800

    # ---- Fixed flags (not varied) ----
    remaining_runtime_head: bool = True
    outcome_bool: bool = False
    layernorm_embeds: bool = True

    # ---- Paths (auto-derived, can be overridden) ----
    data_dir: str = ""       # root dir where {log_name}/ folders live
    results_dir: str = ""    # root dir for results/

    # ---- Trial label (auto-generated if empty) ----
    trial_name: str = ""

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def log_name(self) -> str:
        return DATASET_REGISTRY[self.dataset]

    @property
    def d_ff(self) -> int:
        return self.d_model * self.d_ff_multiplier

    @property
    def trial_dir(self) -> str:
        """Full path to this trial's output directory."""
        model_tag = f"sutran_{self.model_type}"
        base = os.path.join(self.results_dir, model_tag, self.dataset, self.experiment_type)
        if self.experiment_type == "individual" and self.hp_name:
            val = getattr(self, self.hp_name)
            return os.path.join(base, self.hp_name, f"{self.hp_name}_{val}", f"seed_{self.seed}")
        name = self.trial_name or f"trial_seed_{self.seed}"
        return os.path.join(base, name)

    def _default_trial_name(self) -> str:
        if self.experiment_type == "individual" and self.hp_name:
            val = getattr(self, self.hp_name)
            return f"{self.hp_name}_{val}_seed_{self.seed}"
        return f"trial_seed_{self.seed}"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["d_ff"] = self.d_ff
        d["log_name"] = self.log_name
        d["trial_dir"] = self.trial_dir
        return d

    def save(self, path: str):
        import yaml
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # Remove derived keys that aren't constructor args
        for key in ["d_ff", "log_name", "trial_dir"]:
            data.pop(key, None)
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})

    @classmethod
    def from_args(cls, args: Optional[list] = None) -> "ExperimentConfig":
        """Build config from defaults + CLI overrides."""
        parser = argparse.ArgumentParser(description="HPO Experiment Runner")
        # Add every dataclass field as a CLI argument
        for name, fld in cls.__dataclass_fields__.items():
            t = fld.type
            if t == "bool":
                t = lambda x: x.lower() in ("true", "1", "yes")
            elif t == "float":
                t = float
            elif t == "int":
                t = int
            else:
                t = str
            parser.add_argument(f"--{name}", type=t, default=None)

        parsed = parser.parse_args(args)
        cfg = cls()

        # Override with any CLI arguments that were explicitly set
        for name in cls.__dataclass_fields__:
            val = getattr(parsed, name, None)
            if val is not None:
                setattr(cfg, name, val)

        # Auto-set paths if not specified
        if not cfg.data_dir:
            # Default: data lives in the repo root
            cfg.data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not cfg.results_dir:
            cfg.results_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results"
            )

        return cfg


# ---------------------------------------------------------------------------
# Phase 1 HP values
# ---------------------------------------------------------------------------
PHASE1_HP_VALUES = {
    "d_model": [4, 8, 16, 24, 32, 48, 64, 128],
    "d_ff_multiplier": [1, 2, 3, 4],
    "num_layers": [1, 2, 4, 6, 8],  # sets both encoder and decoder layers
    "learning_rate": [0.0002, 0.0004, 0.0008, 0.001, 0.005],
    "dropout": [0, 0.2, 0.5, 0.8],
    "num_heads": [8, 16],  # d_model/4 and d_model/2 at base d_model=32
}

# For the d_model sweep, num_heads = d_model // 4  (constant head_dim=4)
# For the num_heads sweep, d_model is fixed at 32
PHASE1_SEEDS = [42]  # add more seeds later (43, 44) without re-running existing

# Default HP values (the "control" configuration)
DEFAULTS = {
    "d_model": 32,
    "d_ff_multiplier": 4,
    "num_layers": 4,
    "learning_rate": 0.0002,
    "dropout": 0.2,
    "num_heads": 8,
}

PHASE1_DATASETS = ["BAC_adj", "BAC_OG"]
PHASE1_MODELS = ["DA", "NDA"]
