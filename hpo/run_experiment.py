"""Unified experiment runner for HPO trials.

Replicates the logic of TRAIN_EVAL_SUTRAN_DA.py / NDA.py but driven
entirely by an ExperimentConfig.  Does NOT modify any existing code in
SuTraN/ or Preprocessing/.
"""
import os
import sys
import random
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

# Ensure repo root is on sys.path so SuTraN/ imports work
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hpo.config import ExperimentConfig


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------
# Data loading (mirrors the original training scripts exactly)
# ---------------------------------------------------------------
def load_dataset_metadata(cfg: ExperimentConfig):
    """Load all pickled metadata for the dataset. Returns a dict."""
    log = cfg.log_name
    base = os.path.join(cfg.data_dir, log)

    meta = {}
    meta["cardinality_dict"] = _load_pkl(os.path.join(base, f"{log}_cardin_dict.pkl"))
    meta["num_activities"] = meta["cardinality_dict"]["concept:name"] + 2
    meta["cardinality_list_prefix"] = _load_pkl(os.path.join(base, f"{log}_cardin_list_prefix.pkl"))
    meta["cardinality_list_suffix"] = _load_pkl(os.path.join(base, f"{log}_cardin_list_suffix.pkl"))
    meta["num_cols_dict"] = _load_pkl(os.path.join(base, f"{log}_num_cols_dict.pkl"))
    meta["cat_cols_dict"] = _load_pkl(os.path.join(base, f"{log}_cat_cols_dict.pkl"))
    meta["train_means_dict"] = _load_pkl(os.path.join(base, f"{log}_train_means_dict.pkl"))
    meta["train_std_dict"] = _load_pkl(os.path.join(base, f"{log}_train_std_dict.pkl"))

    means = meta["train_means_dict"]
    stds = meta["train_std_dict"]
    meta["mean_std_ttne"] = [means["timeLabel_df"][0], stds["timeLabel_df"][0]]
    meta["mean_std_tsp"] = [means["suffix_df"][1], stds["suffix_df"][1]]
    meta["mean_std_tss"] = [means["suffix_df"][0], stds["suffix_df"][0]]
    meta["mean_std_rrt"] = [means["timeLabel_df"][1], stds["timeLabel_df"][1]]

    meta["num_numericals_pref"] = len(meta["num_cols_dict"]["prefix_df"])
    meta["num_numericals_suf"] = len(meta["num_cols_dict"]["suffix_df"])
    meta["num_categoricals_pref"] = len(meta["cat_cols_dict"]["prefix_df"])
    meta["num_categoricals_suf"] = len(meta["cat_cols_dict"]["suffix_df"])

    # tss_index for NDA slicing
    meta["tss_index"] = meta["num_cols_dict"]["prefix_df"].index("ts_start")

    return meta


def load_tensors(cfg: ExperimentConfig, meta: dict):
    """Load train/val/test tensors and apply NDA slicing if needed."""
    base = os.path.join(cfg.data_dir, cfg.log_name)

    train_dataset = torch.load(os.path.join(base, "train_tensordataset.pt"))
    val_dataset = torch.load(os.path.join(base, "val_tensordataset.pt"))
    test_dataset = torch.load(os.path.join(base, "test_tensordataset.pt"))

    if cfg.model_type == "NDA":
        ncp = meta["num_categoricals_pref"]
        tss_idx = meta["tss_index"]
        exc = tss_idx + 2
        # Retain only activity label + time features (tss, tsp)
        train_dataset = (train_dataset[ncp - 1],) + (train_dataset[ncp][:, :, tss_idx:exc],) + train_dataset[ncp + 1:]
        val_dataset = (val_dataset[ncp - 1],) + (val_dataset[ncp][:, :, tss_idx:exc],) + val_dataset[ncp + 1:]
        test_dataset = (test_dataset[ncp - 1],) + (test_dataset[ncp][:, :, tss_idx:exc],) + test_dataset[ncp + 1:]

    # Wrap training set in TensorDataset
    train_dataset = TensorDataset(*train_dataset)

    return train_dataset, val_dataset, test_dataset


# ---------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------
def create_model(cfg: ExperimentConfig, meta: dict):
    if cfg.model_type == "DA":
        from SuTraN.SuTraN import SuTraN
        model = SuTraN(
            num_activities=meta["num_activities"],
            d_model=cfg.d_model,
            cardinality_categoricals_pref=meta["cardinality_list_prefix"],
            num_numericals_pref=meta["num_numericals_pref"],
            num_prefix_encoder_layers=cfg.num_layers,
            num_decoder_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            remaining_runtime_head=cfg.remaining_runtime_head,
            layernorm_embeds=cfg.layernorm_embeds,
            outcome_bool=cfg.outcome_bool,
        )
    elif cfg.model_type == "NDA":
        from SuTraN.SuTraN import SuTraN_no_context
        model = SuTraN_no_context(
            num_activities=meta["num_activities"],
            d_model=cfg.d_model,
            num_prefix_encoder_layers=cfg.num_layers,
            num_decoder_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            remaining_runtime_head=cfg.remaining_runtime_head,
            layernorm_embeds=cfg.layernorm_embeds,
            outcome_bool=cfg.outcome_bool,
        )
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")
    return model


# ---------------------------------------------------------------
# Best-epoch selection (same logic as original scripts)
# ---------------------------------------------------------------
def select_best_epoch(backup_path: str) -> int:
    csv_path = os.path.join(backup_path, "backup_results.csv")
    df = pd.read_csv(csv_path)
    dl_col = "Activity suffix: 1-DL (validation)"
    rrt_col = "RRT - mintues MAE validation"
    df["rrt_rank_val"] = df[rrt_col].rank(method="min").astype(int)
    df["dl_rank_val"] = df[dl_col].rank(method="min", ascending=False).astype(int)
    df["summed_rank_val"] = df["rrt_rank_val"] + df["dl_rank_val"]
    best_row = df.loc[df["summed_rank_val"].idxmin()]
    return int(best_row["epoch"])


# ---------------------------------------------------------------
# Load checkpoint (mirrors original)
# ---------------------------------------------------------------
def load_best_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------
def run_experiment(cfg: ExperimentConfig):
    """Full train → evaluate pipeline driven by config."""

    # Skip if already completed
    trial_dir = cfg.trial_dir
    done_marker = os.path.join(trial_dir, "summary_metrics.csv")
    if os.path.isfile(done_marker):
        print(f"SKIP (already done): {trial_dir}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: model={cfg.model_type}, dataset={cfg.dataset}, "
          f"seed={cfg.seed}, hp_name={cfg.hp_name}")

    # 1. Seed
    _set_seed(cfg.seed)

    # 2. Load metadata & tensors
    meta = load_dataset_metadata(cfg)
    train_dataset, val_dataset, test_dataset = load_tensors(cfg, meta)

    # 3. Setup output directory
    trial_dir = cfg.trial_dir
    backup_path = os.path.join(trial_dir, "training")
    os.makedirs(backup_path, exist_ok=True)

    # Save frozen config snapshot
    cfg.save(os.path.join(trial_dir, "config.yaml"))

    # 4. Create model
    model = create_model(cfg, meta)
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=cfg.lr_decay_factor)

    # 6. Train
    from SuTraN.train_procedure import train_model

    num_cat_pref = 1 if cfg.model_type == "NDA" else meta["num_categoricals_pref"]

    train_model(
        model,
        optimizer,
        train_dataset,
        val_dataset,
        start_epoch=0,
        num_epochs=cfg.num_epochs,
        remaining_runtime_head=cfg.remaining_runtime_head,
        outcome_bool=cfg.outcome_bool,
        num_classes=meta["num_activities"],
        batch_interval=cfg.batch_interval,
        path_name=backup_path,
        num_categoricals_pref=num_cat_pref,
        mean_std_ttne=meta["mean_std_ttne"],
        mean_std_tsp=meta["mean_std_tsp"],
        mean_std_tss=meta["mean_std_tss"],
        mean_std_rrt=meta["mean_std_rrt"],
        batch_size=cfg.batch_size,
        patience=cfg.patience,
        lr_scheduler_present=True,
        lr_scheduler=lr_scheduler,
        max_norm=cfg.max_norm,
    )

    # 7. Select best epoch & reload
    best_epoch = select_best_epoch(backup_path)
    print(f"Best epoch: {best_epoch}")

    best_ckpt = os.path.join(backup_path, f"model_epoch_{best_epoch}.pt")
    model = create_model(cfg, meta)
    model = load_best_model(model, best_ckpt, device)

    # 8. Test-set inference
    from SuTraN.inference_procedure import inference_loop

    results_path = os.path.join(trial_dir, "test_results")
    os.makedirs(results_path, exist_ok=True)

    inf_results = inference_loop(
        model,
        test_dataset,
        cfg.remaining_runtime_head,
        cfg.outcome_bool,
        num_cat_pref,
        meta["mean_std_ttne"],
        meta["mean_std_tsp"],
        meta["mean_std_tss"],
        meta["mean_std_rrt"],
        results_path=results_path,
        val_batch_size=2048,
    )

    # 9. Save summary metrics
    avg_results = {
        "MAE_TTNE_stand": float(inf_results[0]),
        "MAE_TTNE_minutes": float(inf_results[1]),
        "DL_sim": float(inf_results[2]),
        "perc_too_early": float(inf_results[3]),
        "perc_too_late": float(inf_results[4]),
        "perc_correct": float(inf_results[5]),
        "mean_absolute_length_diff": float(inf_results[6]),
        "mean_too_early": float(inf_results[7]),
        "mean_too_late": float(inf_results[8]),
        "MAE_RRT_stand": float(inf_results[9]),
        "MAE_RRT_minutes": float(inf_results[10]),
        "best_epoch": best_epoch,
        "seed": cfg.seed,
        "hp_name": cfg.hp_name,
    }

    # Include the actual HP values for easy aggregation
    for hp in ["d_model", "num_heads", "d_ff_multiplier",
               "num_prefix_encoder_layers", "num_decoder_layers",
               "learning_rate", "dropout"]:
        avg_results[hp] = getattr(cfg, hp)

    # Save as pickle (for programmatic access)
    with open(os.path.join(trial_dir, "summary_metrics.pkl"), "wb") as f:
        pickle.dump(avg_results, f)

    # Also save as CSV row (for easy aggregation later)
    pd.DataFrame([avg_results]).to_csv(
        os.path.join(trial_dir, "summary_metrics.csv"), index=False)

    # Save prefix/suffix length results
    with open(os.path.join(results_path, "prefix_length_results_dict.pkl"), "wb") as f:
        pickle.dump(inf_results[-2], f)
    with open(os.path.join(results_path, "suffix_length_results_dict.pkl"), "wb") as f:
        pickle.dump(inf_results[-1], f)

    print(f"\nTest results saved to {trial_dir}")
    print(f"  MAE TTNE: {avg_results['MAE_TTNE_minutes']:.2f} min")
    print(f"  DL sim:   {avg_results['DL_sim']:.4f}")
    print(f"  MAE RRT:  {avg_results['MAE_RRT_minutes']:.2f} min")

    return avg_results


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    cfg = ExperimentConfig.from_args()
    run_experiment(cfg)
