"""
Phase 1 runner — one-at-a-time HP sensitivity analysis.

Accepts (model_type, dataset, seed) via CLI. Loops through every HP
value in HP_VALUES, trains with all other HPs at their defaults,
and evaluates on the test set.

Data is loaded ONCE and reused across all trials.

Usage (from project root):
    python -m hpo.phase1.run --model_type DA --dataset BAC_adj --seed 42
"""

import argparse
import os
import sys
import glob
import csv
import pickle
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hpo.phase1.config import DATASET_REGISTRY, FIXED, HP_VALUES, DEFAULTS


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 HPO runner")
    p.add_argument("--model_type",  required=True, choices=["DA", "NDA"])
    p.add_argument("--dataset",     required=True, choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--data_dir",    type=str, default="")
    p.add_argument("--results_dir", type=str, default="")
    return p.parse_args()


# ---------------------------------------------------------------
# Seed
# ---------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------

def load_metadata(log_name, data_dir):
    base = os.path.join(data_dir, log_name)

    def _pkl(name):
        with open(os.path.join(base, f"{log_name}_{name}.pkl"), "rb") as f:
            return pickle.load(f)

    meta = {}
    meta["cardinality_dict"]        = _pkl("cardin_dict")
    meta["num_activities"]          = meta["cardinality_dict"]["concept:name"] + 2
    meta["cardinality_list_prefix"] = _pkl("cardin_list_prefix")
    meta["cardinality_list_suffix"] = _pkl("cardin_list_suffix")
    meta["num_cols_dict"]           = _pkl("num_cols_dict")
    meta["cat_cols_dict"]           = _pkl("cat_cols_dict")
    meta["train_means_dict"]        = _pkl("train_means_dict")
    meta["train_std_dict"]          = _pkl("train_std_dict")

    means = meta["train_means_dict"]
    stds  = meta["train_std_dict"]
    meta["mean_std_ttne"] = [means["timeLabel_df"][0], stds["timeLabel_df"][0]]
    meta["mean_std_tsp"]  = [means["suffix_df"][1],    stds["suffix_df"][1]]
    meta["mean_std_tss"]  = [means["suffix_df"][0],    stds["suffix_df"][0]]
    meta["mean_std_rrt"]  = [means["timeLabel_df"][1], stds["timeLabel_df"][1]]

    meta["num_numericals_pref"]   = len(meta["num_cols_dict"]["prefix_df"])
    meta["num_numericals_suf"]    = len(meta["num_cols_dict"]["suffix_df"])
    meta["num_categoricals_pref"] = len(meta["cat_cols_dict"]["prefix_df"])
    meta["num_categoricals_suf"]  = len(meta["cat_cols_dict"]["suffix_df"])
    meta["tss_index"]             = meta["num_cols_dict"]["prefix_df"].index("ts_start")

    return meta


def load_tensors(log_name, data_dir, model_type, meta):
    base = os.path.join(data_dir, log_name)

    train_dataset = torch.load(os.path.join(base, "train_tensordataset.pt"))
    val_dataset   = torch.load(os.path.join(base, "val_tensordataset.pt"))
    test_dataset  = torch.load(os.path.join(base, "test_tensordataset.pt"))

    if model_type == "NDA":
        ncp     = meta["num_categoricals_pref"]
        tss_idx = meta["tss_index"]
        exc     = tss_idx + 2
        train_dataset = (train_dataset[ncp - 1],) + (train_dataset[ncp][:, :, tss_idx:exc],) + train_dataset[ncp + 1:]
        val_dataset   = (val_dataset[ncp - 1],)   + (val_dataset[ncp][:, :, tss_idx:exc],)   + val_dataset[ncp + 1:]
        test_dataset  = (test_dataset[ncp - 1],)  + (test_dataset[ncp][:, :, tss_idx:exc],)  + test_dataset[ncp + 1:]

    train_dataset = TensorDataset(*train_dataset)
    return train_dataset, val_dataset, test_dataset


# ---------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------

def create_model(model_type, meta, params):
    if model_type == "DA":
        from SuTraN.SuTraN import SuTraN
        return SuTraN(
            num_activities=meta["num_activities"],
            d_model=params["d_model"],
            cardinality_categoricals_pref=meta["cardinality_list_prefix"],
            num_numericals_pref=meta["num_numericals_pref"],
            num_prefix_encoder_layers=params["num_layers"],
            num_decoder_layers=params["num_layers"],
            num_heads=params["num_heads"],
            d_ff=params["d_ff"],
            dropout=params["dropout"],
            remaining_runtime_head=True,
            layernorm_embeds=True,
            outcome_bool=False,
        )
    else:
        from SuTraN.SuTraN import SuTraN_no_context
        return SuTraN_no_context(
            num_activities=meta["num_activities"],
            d_model=params["d_model"],
            num_prefix_encoder_layers=params["num_layers"],
            num_decoder_layers=params["num_layers"],
            num_heads=params["num_heads"],
            d_ff=params["d_ff"],
            dropout=params["dropout"],
            remaining_runtime_head=True,
            layernorm_embeds=True,
            outcome_bool=False,
        )


# ---------------------------------------------------------------
# Best-epoch selection (rank-sum on RRT MAE + DL similarity)
# ---------------------------------------------------------------

def select_best_epoch(backup_path):
    df = pd.read_csv(os.path.join(backup_path, "backup_results.csv"))
    df["rrt_rank"] = df["RRT - mintues MAE validation"].rank(method="min").astype(int)
    df["dl_rank"]  = df["Activity suffix: 1-DL (validation)"].rank(method="min", ascending=False).astype(int)
    df["rank_sum"] = df["rrt_rank"] + df["dl_rank"]
    return int(df.loc[df["rank_sum"].idxmin(), "epoch"])


# ---------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------

def get_trial_dir(results_dir, model_type, dataset, hp_name, hp_value, seed):
    """results/sutran_DA/BAC_OG/individual/d_model/d_model_64/seed_42"""
    return os.path.join(
        results_dir, f"sutran_{model_type}", dataset, "individual",
        hp_name, f"{hp_name}_{hp_value}", f"seed_{seed}",
    )


def get_summary_csv(results_dir, model_type, dataset):
    """results/sutran_DA/BAC_OG/individual/summary.csv"""
    return os.path.join(
        results_dir, f"sutran_{model_type}", dataset, "individual", "summary.csv",
    )


# ---------------------------------------------------------------
# Rebuild summary CSV from all results.pkl files
# ---------------------------------------------------------------

def rebuild_summary(results_dir, model_type, dataset):
    individual_dir = os.path.join(
        results_dir, f"sutran_{model_type}", dataset, "individual",
    )
    pattern = os.path.join(individual_dir, "*", "*", "*", "results.pkl")
    pkl_files = sorted(glob.glob(pattern))

    if not pkl_files:
        return

    rows = []
    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            rows.append(pickle.load(f))

    summary_path = os.path.join(individual_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------
# Single trial: train + evaluate
# ---------------------------------------------------------------

def run_trial(model_type, meta, train_dataset, val_dataset, test_dataset,
              params, hp_name, hp_value, seed, trial_dir):
    """Train one config, evaluate on test set, save and return result dict."""

    results_pkl = os.path.join(trial_dir, "results.pkl")
    if os.path.exists(results_pkl):
        with open(results_pkl, "rb") as f:
            return pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    model     = create_model(model_type, meta, params)
    model.to(device)

    optimizer    = torch.optim.AdamW(model.parameters(),
                                     lr=params["learning_rate"],
                                     weight_decay=params["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                       optimizer, gamma=params["lr_decay"])

    backup_path  = os.path.join(trial_dir, "training")
    os.makedirs(backup_path, exist_ok=True)

    num_cat_pref = 1 if model_type == "NDA" else meta["num_categoricals_pref"]

    from SuTraN.train_procedure import train_model
    train_model(
        model, optimizer, train_dataset, val_dataset,
        start_epoch=0,
        num_epochs=FIXED["num_epochs"],
        remaining_runtime_head=True,
        outcome_bool=False,
        num_classes=meta["num_activities"],
        batch_interval=FIXED["batch_interval"],
        path_name=backup_path,
        num_categoricals_pref=num_cat_pref,
        mean_std_ttne=meta["mean_std_ttne"],
        mean_std_tsp=meta["mean_std_tsp"],
        mean_std_tss=meta["mean_std_tss"],
        mean_std_rrt=meta["mean_std_rrt"],
        batch_size=FIXED["batch_size"],
        patience=FIXED["patience"],
        lr_scheduler_present=True,
        lr_scheduler=lr_scheduler,
        max_norm=FIXED["max_norm"],
    )

    # Select best epoch and reload
    best_epoch = select_best_epoch(backup_path)
    print(f"  Best epoch: {best_epoch}")

    best_ckpt = os.path.join(backup_path, f"model_epoch_{best_epoch}.pt")
    model = create_model(model_type, meta, params)
    ckpt  = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Delete all checkpoints except the best
    for fname in os.listdir(backup_path):
        if fname.startswith("model_epoch_") and fname != f"model_epoch_{best_epoch}.pt":
            os.remove(os.path.join(backup_path, fname))

    # Test inference
    from SuTraN.inference_procedure import inference_loop
    test_results_path = os.path.join(trial_dir, "test_results")
    os.makedirs(test_results_path, exist_ok=True)

    inf = inference_loop(
        model, test_dataset,
        remaining_runtime_head=True,
        outcome_bool=False,
        num_categoricals_pref=num_cat_pref,
        mean_std_ttne=meta["mean_std_ttne"],
        mean_std_tsp=meta["mean_std_tsp"],
        mean_std_tss=meta["mean_std_tss"],
        mean_std_rrt=meta["mean_std_rrt"],
        results_path=test_results_path,
        val_batch_size=2048,
    )

    # Read best-epoch val metrics from backup_results.csv
    val_df   = pd.read_csv(os.path.join(backup_path, "backup_results.csv"))
    best_row = val_df[val_df["epoch"] == best_epoch].iloc[0]

    result = {
        "hp_name":     hp_name,
        "hp_value":    hp_value,
        "seed":        seed,
        "best_epoch":  best_epoch,
        "n_epochs":    len(val_df),
        # HPs used
        "d_model":         params["d_model"],
        "d_ff_multiplier": params["d_ff_multiplier"],
        "d_ff":            params["d_ff"],
        "num_layers":      params["num_layers"],
        "num_heads":       params["num_heads"],
        "learning_rate":   params["learning_rate"],
        "dropout":         params["dropout"],
        "weight_decay":    params["weight_decay"],
        "lr_decay":        params["lr_decay"],
        # Test metrics
        "test_MAE_ttne_stand":       float(inf[0]),
        "test_MAE_ttne_minutes":     float(inf[1]),
        "test_DL_similarity":        float(inf[2]),
        "test_perc_too_early":       float(inf[3]),
        "test_perc_too_late":        float(inf[4]),
        "test_perc_correct":         float(inf[5]),
        "test_mean_abs_length_diff": float(inf[6]),
        "test_mean_too_early":       float(inf[7]),
        "test_mean_too_late":        float(inf[8]),
        "test_MAE_rrt_stand":        float(inf[9]),
        "test_MAE_rrt_minutes":      float(inf[10]),
        # Val metrics (best epoch)
        "val_MAE_ttne_stand":   float(best_row["TTNE - standardized MAE validation"]),
        "val_MAE_ttne_minutes": float(best_row["TTNE - minutes MAE validation"]),
        "val_DL_similarity":    float(best_row["Activity suffix: 1-DL (validation)"]),
        "val_MAE_rrt_stand":    float(best_row["RRT - standardized MAE validation"]),
        "val_MAE_rrt_minutes":  float(best_row["RRT - mintues MAE validation"]),
    }

    with open(os.path.join(trial_dir, "results.pkl"), "wb") as f:
        pickle.dump(result, f)

    with open(os.path.join(test_results_path, "prefix_length_results_dict.pkl"), "wb") as f:
        pickle.dump(inf[-2], f)
    with open(os.path.join(test_results_path, "suffix_length_results_dict.pkl"), "wb") as f:
        pickle.dump(inf[-1], f)

    return result


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    args = parse_args()
    log_name = DATASET_REGISTRY[args.dataset]

    repo_root   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir    = args.data_dir    or repo_root
    results_dir = args.results_dir or os.path.join(repo_root, "results")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Running: {args.model_type} | {args.dataset} ({log_name}) | seed={args.seed}")
    print()

    # Load data once
    meta = load_metadata(log_name, data_dir)
    train_dataset, val_dataset, test_dataset = load_tensors(
        log_name, data_dir, args.model_type, meta)
    print(f"Data loaded: {meta['num_activities']} activities, "
          f"{len(train_dataset)} training instances\n")

    total = 1 + sum(
        1 for hp, vals in HP_VALUES.items()
        for v in vals if v != DEFAULTS[hp]
    )
    done = 0

    # --- Train base model ---
    base_params = dict(DEFAULTS)
    base_params["d_ff"] = base_params["d_model"] * base_params["d_ff_multiplier"]
    base_dir = get_trial_dir(results_dir, args.model_type, args.dataset,
                             "base", "default", args.seed)
    done += 1
    print(f"{'='*60}")
    print(f"[{done}/{total}]  BASE MODEL (paper defaults)")
    print(f"  d_model={base_params['d_model']}  d_ff={base_params['d_ff']}  "
          f"heads={base_params['num_heads']}  layers={base_params['num_layers']}  "
          f"lr={base_params['learning_rate']}  dropout={base_params['dropout']}")
    print(f"{'='*60}")

    base_result = run_trial(
        args.model_type, meta, train_dataset, val_dataset, test_dataset,
        base_params, "base", "default", args.seed, base_dir)

    print(f"  Test: MAE_rrt={base_result['test_MAE_rrt_minutes']:.2f} min | "
          f"DL_sim={base_result['test_DL_similarity']:.4f}\n")

    # --- Sweep each HP ---
    for hp_name, values in HP_VALUES.items():
        for hp_value in values:
            if hp_value == DEFAULTS[hp_name]:
                continue

            done += 1
            params = dict(DEFAULTS)
            params[hp_name] = hp_value

            if hp_name == "d_model":
                params["num_heads"] = max(1, hp_value // 4)

            params["d_ff"] = params["d_model"] * params["d_ff_multiplier"]

            td = get_trial_dir(results_dir, args.model_type, args.dataset,
                               hp_name, hp_value, args.seed)
            cached = " (cached)" if os.path.exists(os.path.join(td, "results.pkl")) else ""

            print(f"\n{'='*60}")
            print(f"[{done}/{total}]  {hp_name} = {hp_value}{cached}")
            print(f"  d_model={params['d_model']}  d_ff={params['d_ff']}  "
                  f"heads={params['num_heads']}  layers={params['num_layers']}  "
                  f"lr={params['learning_rate']}  dropout={params['dropout']}")
            print(f"{'='*60}")

            result = run_trial(
                args.model_type, meta, train_dataset, val_dataset, test_dataset,
                params, hp_name, hp_value, args.seed, td)

            print(f"  Test: MAE_rrt={result['test_MAE_rrt_minutes']:.2f} min | "
                  f"DL_sim={result['test_DL_similarity']:.4f}")

    # --- Rebuild summary CSV ---
    rebuild_summary(results_dir, args.model_type, args.dataset)
    summary_csv = get_summary_csv(results_dir, args.model_type, args.dataset)

    print(f"\n{'='*60}")
    print(f"All {total} trials complete!")
    print(f"Summary: {summary_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
