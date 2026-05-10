"""
Phase 3 runner — SMAC3 Bayesian optimisation, warm-started from phases 1 & 2.

Strictly sequential: SMAC suggests one config → train fully → observe result →
update surrogate → repeat.  Warm-start data from the phase 1 and (if available)
phase 2 summary CSVs is injected before the first suggestion so the surrogate
is already conditioned on all prior observations.

Usage (from project root):
    python -m hpo.phase3_smac.run \\
        --model_type DA --dataset BAC_OG --seed 42 \\
        --data_dir /scratch/leuven/383/vsc38329/Thesis_Final \\
        --results_dir /scratch/leuven/383/vsc38329/Thesis_Final/results

Smoke-test (1 epoch per trial — verifies plumbing without full training cost):
    python -m hpo.phase3_smac.run ... --num_epochs_override 1

Resume after a time-out: simply resubmit the same command.  The script detects
completed trials on disk and resumes from where it left off.
"""

import argparse
import csv
import glob
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from ConfigSpace import Configuration
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialInfo, TrialValue

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hpo.phase3_smac.config import (
    DATASET_REGISTRY, D_MODEL_CHOICES, FIXED, NUM_TRIALS, WARMSTART_CAP_PERCENTILE,
    build_configspace,
)


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 SMAC3 Bayesian optimisation runner")
    p.add_argument("--model_type",          required=True, choices=["DA", "NDA"])
    p.add_argument("--dataset",             required=True, choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--seed",                type=int,      default=42)
    p.add_argument("--data_dir",            type=str,      default="")
    p.add_argument("--results_dir",         type=str,      default="")
    p.add_argument("--num_epochs_override", type=int,      default=None,
                   help="Override num_epochs for smoke-testing (e.g. --num_epochs_override 1).")
    return p.parse_args()


# ---------------------------------------------------------------
# Seed
# ---------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------
# Data loading  (identical to phase 2)
# ---------------------------------------------------------------

def load_metadata(log_name: str, data_dir: str) -> dict:
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


def load_tensors(log_name: str, data_dir: str, model_type: str, meta: dict):
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
# Model creation  (identical to phase 2)
# ---------------------------------------------------------------

def create_model(model_type: str, meta: dict, params: dict):
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
# Best-epoch selection  (identical to phase 2)
# ---------------------------------------------------------------

def select_best_epoch(backup_path: str) -> int:
    df = pd.read_csv(os.path.join(backup_path, "backup_results.csv"))
    df["rrt_rank"] = df["RRT - mintues MAE validation"].rank(method="min").astype(int)
    df["dl_rank"]  = df["Activity suffix: 1-DL (validation)"].rank(method="min", ascending=False).astype(int)
    df["rank_sum"] = df["rrt_rank"] + df["dl_rank"]
    return int(df.loc[df["rank_sum"].idxmin(), "epoch"])


# ---------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------

def compute_objective(result: dict,
                       mu_ttne: float, sigma_ttne: float,
                       mu_rrt: float, sigma_rrt: float,
                       mu_1dl: float, sigma_1dl: float) -> float:
    """
    Standardised sum of the three validation metrics — lower is better.

    Each term is z-scored using the mean and std computed from the top-50%
    of warm-start trials, so all three contributions are on the same scale.
    """
    ttne = (result["val_MAE_ttne_stand"] - mu_ttne)       / sigma_ttne
    rrt  = (result["val_MAE_rrt_stand"]  - mu_rrt)        / sigma_rrt
    dl   = ((1.0 - result["val_DL_similarity"]) - mu_1dl) / sigma_1dl
    return float(ttne + rrt + dl)


# ---------------------------------------------------------------
# Warm-start helpers
# ---------------------------------------------------------------

def _clamp_to_space(row) -> dict:
    """
    Clamp a summary-CSV row's HP values into the phase-3 search space.

    Phase 1 contains values outside the phase-2/3 bounds (e.g. d_model=4,
    dropout=0.8, weight_decay=0.0).  We map them to the nearest valid value
    so they can be used to create a valid ConfigSpace Configuration.
    """
    return {
        "d_model":         int(min(D_MODEL_CHOICES, key=lambda x: abs(x - int(float(row["d_model"]))))),
        "num_layers":      int(max(2, min(8, round(float(row["num_layers"]))))),
        "d_ff_multiplier": int(max(2, min(8, round(float(row["d_ff_multiplier"]))))),
        "learning_rate":   float(max(1e-5, min(1e-2, float(row["learning_rate"])))),
        "dropout":         float(max(0.0,  min(0.7,  float(row["dropout"])))),
        "weight_decay":    float(max(1e-5, min(1e-2, float(row["weight_decay"])))),
    }


def load_warmstart(results_dir: str, model_type: str, dataset: str, cs):
    """
    Load phase 1 and (optionally) phase 2 summary CSVs.

    Computes per-metric standardisation params (mean + std) from the top-50%
    rows (ranked by combined rank-sum over all three val metrics).

    Returns (pairs, mu_ttne, sigma_ttne, mu_rrt, sigma_rrt, mu_1dl, sigma_1dl)
    where pairs is a list of (TrialInfo, TrialValue) tuples ready to be told
    to SMAC.
    """
    phase2_csv = os.path.join(results_dir, f"sutran_{model_type}", dataset,
                              "lhs_search", "summary.csv")

    dfs = []
    if os.path.isfile(phase2_csv):
        dfs.append(pd.read_csv(phase2_csv))
        print(f"  Warm-start: loaded phase 2 LHS ({len(dfs[-1])} rows) from {phase2_csv}")
    else:
        print(f"  Warm-start: phase 2 LHS CSV not found, skipping.")

    if not dfs:
        print("  Warm-start: no prior data found; using fallback standardisation params (mu=0, sigma=1).")
        return [], 0.0, 1.0, 0.0, 1.0, 0.0, 1.0

    df = pd.concat(dfs, ignore_index=True)

    # Rank by all three val metrics (lower combined rank = better trial)
    df["_r_ttne"] = df["val_MAE_ttne_stand"].rank(method="min")
    df["_r_rrt"]  = df["val_MAE_rrt_stand"].rank(method="min")
    df["_r_1dl"]  = (1.0 - df["val_DL_similarity"]).rank(method="min")
    df["_rank"]   = df["_r_ttne"] + df["_r_rrt"] + df["_r_1dl"]

    n_top = max(1, len(df) // 2)  # top 50%
    top   = df.nsmallest(n_top, "_rank")
    top_1dl = 1.0 - top["val_DL_similarity"]

    mu_ttne    = float(top["val_MAE_ttne_stand"].mean())
    sigma_ttne = float(max(top["val_MAE_ttne_stand"].std(ddof=0), 1e-8))
    mu_rrt     = float(top["val_MAE_rrt_stand"].mean())
    sigma_rrt  = float(max(top["val_MAE_rrt_stand"].std(ddof=0), 1e-8))
    mu_1dl     = float(top_1dl.mean())
    sigma_1dl  = float(max(top_1dl.std(ddof=0), 1e-8))

    print(f"  Warm-start: top-50% standardisation params — "
          f"mu_ttne={mu_ttne:.4f} σ={sigma_ttne:.4f}  "
          f"mu_rrt={mu_rrt:.4f} σ={sigma_rrt:.4f}  "
          f"mu_1dl={mu_1dl:.4f} σ={sigma_1dl:.4f}")

    pairs   = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            hp_values = _clamp_to_space(row)
            config    = Configuration(cs, values=hp_values)
            cost      = float(
                (row["val_MAE_ttne_stand"] - mu_ttne) / sigma_ttne
                + (row["val_MAE_rrt_stand"]  - mu_rrt)  / sigma_rrt
                + ((1.0 - row["val_DL_similarity"]) - mu_1dl) / sigma_1dl
            )
            pairs.append((TrialInfo(config=config, seed=42), TrialValue(cost=cost)))
        except Exception:
            skipped += 1

    if WARMSTART_CAP_PERCENTILE is not None and pairs:
        costs     = np.array([tv.cost for _, tv in pairs])
        cap       = float(np.percentile(costs, WARMSTART_CAP_PERCENTILE))
        n_capped  = int(np.sum(costs > cap))
        pairs     = [(ti, TrialValue(cost=min(tv.cost, cap))) for ti, tv in pairs]
        print(f"  Warm-start: performance cap at {WARMSTART_CAP_PERCENTILE}th pct "
              f"(threshold={cap:.4f}, {n_capped}/{len(pairs)} trials capped)")

    msg = f"  Warm-start: {len(pairs)} trials injected"
    if skipped:
        msg += f" ({skipped} skipped — HP values could not be mapped to search space)"
    print(msg)
    return pairs, mu_ttne, sigma_ttne, mu_rrt, sigma_rrt, mu_1dl, sigma_1dl


# ---------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------

def load_completed_trials(smac_search_dir: str) -> dict:
    """
    Return {trial_idx: result_dict} for all trials that have been fully
    saved to disk in a previous run (i.e. results.pkl contains both
    'objective' and 'smac_config_values').
    """
    completed = {}
    for path in sorted(glob.glob(os.path.join(smac_search_dir, "trial_*", "results.pkl"))):
        try:
            with open(path, "rb") as f:
                r = pickle.load(f)
            if "smac_config_values" in r and "objective" in r:
                completed[r["trial_idx"]] = r
        except Exception:
            pass
    return completed


# ---------------------------------------------------------------
# Single trial runner  (from phase 2, with num_epochs parameter)
# ---------------------------------------------------------------

def run_trial(model_type: str, meta: dict,
              train_dataset, val_dataset, test_dataset,
              params: dict, trial_idx: int, seed: int,
              trial_dir: str, num_epochs: int) -> dict:
    """Train one configuration, evaluate on test set, save and return result dict."""

    results_pkl = os.path.join(trial_dir, "results.pkl")
    if os.path.exists(results_pkl):
        with open(results_pkl, "rb") as f:
            cached = pickle.load(f)
        if "val_DL_similarity" in cached:
            return cached

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trial_seed = seed + trial_idx
    set_seed(trial_seed)

    model        = create_model(model_type, meta, params)
    model.to(device)
    optimizer    = torch.optim.AdamW(model.parameters(),
                                     lr=params["learning_rate"],
                                     weight_decay=params["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=FIXED["lr_decay"])

    backup_path  = os.path.join(trial_dir, "training")
    os.makedirs(backup_path, exist_ok=True)

    num_cat_pref = 1 if model_type == "NDA" else meta["num_categoricals_pref"]

    from SuTraN.train_procedure import train_model
    train_model(
        model, optimizer, train_dataset, val_dataset,
        start_epoch=0,
        num_epochs=num_epochs,
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

    best_epoch = select_best_epoch(backup_path)
    print(f"  Best epoch: {best_epoch}")

    best_ckpt = os.path.join(backup_path, f"model_epoch_{best_epoch}.pt")
    model = create_model(model_type, meta, params)
    ckpt  = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    for fname in os.listdir(backup_path):
        if fname.startswith("model_epoch_") and fname != f"model_epoch_{best_epoch}.pt":
            os.remove(os.path.join(backup_path, fname))

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

    val_df   = pd.read_csv(os.path.join(backup_path, "backup_results.csv"))
    best_row = val_df[val_df["epoch"] == best_epoch].iloc[0]

    result = {
        "trial_idx":   trial_idx,
        "seed":        trial_seed,
        "best_epoch":  best_epoch,
        "n_epochs":    len(val_df),
        "d_model":         params["d_model"],
        "d_ff_multiplier": params["d_ff_multiplier"],
        "d_ff":            params["d_ff"],
        "num_layers":      params["num_layers"],
        "num_heads":       params["num_heads"],
        "learning_rate":   params["learning_rate"],
        "dropout":         params["dropout"],
        "weight_decay":    params["weight_decay"],
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
        "val_MAE_ttne_stand":   float(best_row["TTNE - standardized MAE validation"]),
        "val_MAE_ttne_minutes": float(best_row["TTNE - minutes MAE validation"]),
        "val_DL_similarity":    float(best_row["Activity suffix: 1-DL (validation)"]),
        "val_MAE_rrt_stand":    float(best_row["RRT - standardized MAE validation"]),
        "val_MAE_rrt_minutes":  float(best_row["RRT - mintues MAE validation"]),
    }

    # Base result saved here; objective + smac_config_values are appended by the caller
    with open(results_pkl, "wb") as f:
        pickle.dump(result, f)

    with open(os.path.join(test_results_path, "prefix_length_results_dict.pkl"), "wb") as f:
        pickle.dump(inf[-2], f)
    with open(os.path.join(test_results_path, "suffix_length_results_dict.pkl"), "wb") as f:
        pickle.dump(inf[-1], f)

    return result


# ---------------------------------------------------------------
# Summary rebuild  (called automatically at the end of the run)
# ---------------------------------------------------------------

def rebuild_summary(smac_search_dir: str):
    pkl_files = sorted(glob.glob(os.path.join(smac_search_dir, "trial_*", "results.pkl")))
    if not pkl_files:
        return

    rows = []
    for path in pkl_files:
        with open(path, "rb") as f:
            r = pickle.load(f)
        # Exclude the internal SMAC config dict — not useful in the CSV
        rows.append({k: v for k, v in r.items() if k != "smac_config_values"})

    rows.sort(key=lambda r: r["trial_idx"])

    summary_path = os.path.join(smac_search_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary written: {summary_path} ({len(rows)} rows)")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    args = parse_args()

    repo_root   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir    = args.data_dir    or repo_root
    results_dir = args.results_dir or os.path.join(repo_root, "results")
    num_epochs  = args.num_epochs_override if args.num_epochs_override is not None else FIXED["num_epochs"]

    log_name        = DATASET_REGISTRY[args.dataset]
    smac_search_dir = os.path.join(results_dir, f"sutran_{args.model_type}",
                                   args.dataset, "smac_search")
    smac_output_dir = os.path.join(smac_search_dir, "smac_output")
    os.makedirs(smac_search_dir, exist_ok=True)

    print(f"Phase 3 SMAC: {args.model_type} | {args.dataset} | seed={args.seed}")
    print(f"  num_epochs={num_epochs}  num_trials={NUM_TRIALS}")
    print()

    meta = load_metadata(log_name, data_dir)
    train_ds, val_ds, test_ds = load_tensors(log_name, data_dir, args.model_type, meta)
    print(f"Data loaded: {meta['num_activities']} activities, "
          f"{len(train_ds)} training instances\n")

    cs = build_configspace()

    print("Loading warm-start data...")
    warmstart_pairs, mu_ttne, sigma_ttne, mu_rrt, sigma_rrt, mu_1dl, sigma_1dl = load_warmstart(
        results_dir, args.model_type, args.dataset, cs,
    )
    print()

    completed = load_completed_trials(smac_search_dir)
    if completed:
        print(f"Resuming: {len(completed)} trials already completed on disk.\n")

    # --- SMAC setup ---
    # n_trials must accommodate warm-start tells + re-injected completed trials
    # + the NUM_TRIALS new trials.  We add a buffer so SMAC never refuses an ask().
    n_budget = NUM_TRIALS + len(warmstart_pairs) + len(completed) + 50
    scenario = Scenario(
        cs,
        deterministic=True,
        n_trials=n_budget,
        seed=args.seed,
        output_directory=smac_output_dir,
    )
    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario, max_config_calls=1,
    )
    smac = HyperparameterOptimizationFacade(
        scenario,
        target_function=lambda cfg, seed: 0.0,
        model=HyperparameterOptimizationFacade.get_model(scenario, n_trees=100, min_samples_leaf=3),
        intensifier=intensifier,
        initial_design=HyperparameterOptimizationFacade.get_initial_design(
            scenario, n_configs=0,
        ),
        overwrite=True,
    )

    # Inject warm-start data
    for info, value in warmstart_pairs:
        smac.tell(info, value)
    print(f"Warm-start: {len(warmstart_pairs)} prior trials injected into SMAC surrogate.\n")

    # Re-inject previously-completed phase-3 trials so surrogate is up to date
    for trial_idx in sorted(completed):
        r = completed[trial_idx]
        try:
            config = Configuration(cs, values=r["smac_config_values"])
            smac.tell(TrialInfo(config=config, seed=r["seed"]),
                      TrialValue(cost=r["objective"]))
        except Exception as e:
            print(f"  Warning: could not re-inject trial {trial_idx}: {e}")

    # --- Main optimisation loop ---
    for trial_idx in range(NUM_TRIALS):
        if trial_idx in completed:
            print(f"  Trial {trial_idx:04d} already completed — skipping.")
            continue

        info   = smac.ask()
        config = info.config

        params = dict(config)
        # Cast to plain Python types — ConfigSpace may return wrapped values
        # that fail isinstance(x, int) checks inside SuTraN.
        params["d_model"]         = int(params["d_model"])
        params["num_layers"]      = int(params["num_layers"])
        params["d_ff_multiplier"] = int(params["d_ff_multiplier"])
        params["learning_rate"]   = float(params["learning_rate"])
        params["dropout"]         = float(params["dropout"])
        params["weight_decay"]    = float(params["weight_decay"])
        params["num_heads"] = max(1, params["d_model"] // 4)
        params["d_ff"]      = params["d_model"] * params["d_ff_multiplier"]

        trial_dir = os.path.join(smac_search_dir, f"trial_{trial_idx:04d}")

        print(f"\n{'='*60}")
        print(f"Trial {trial_idx:04d}/{NUM_TRIALS}")
        print(f"  d_model={params['d_model']}  d_ff={params['d_ff']}  "
              f"heads={params['num_heads']}  layers={params['num_layers']}")
        print(f"  lr={params['learning_rate']:.2e}  dropout={params['dropout']:.3f}  "
              f"wd={params['weight_decay']:.2e}")
        print(f"{'='*60}")

        result = run_trial(
            args.model_type, meta, train_ds, val_ds, test_ds,
            params, trial_idx, args.seed, trial_dir, num_epochs,
        )

        objective = compute_objective(result, mu_ttne, sigma_ttne, mu_rrt, sigma_rrt, mu_1dl, sigma_1dl)
        result["objective"]          = objective
        result["smac_config_values"] = dict(config)

        with open(os.path.join(trial_dir, "results.pkl"), "wb") as f:
            pickle.dump(result, f)

        smac.tell(info, TrialValue(cost=objective))

        print(f"  objective={objective:.4f} | "
              f"val_DL={result['val_DL_similarity']:.4f} | "
              f"val_rrt={result['val_MAE_rrt_stand']:.4f} | "
              f"val_ttne={result['val_MAE_ttne_stand']:.4f}")

    rebuild_summary(smac_search_dir)
    print(f"\nPhase 3 complete.  Output: {smac_search_dir}")


if __name__ == "__main__":
    main()
