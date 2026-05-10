"""
Phase 3 runner — Optuna TPE + Hyperband, warm-started from phases 1 & 2.

Uses an epoch-by-epoch training loop so Hyperband can prune unpromising
trials early.  Only trials that survive all rung comparisons are written to
results.pkl and evaluated on the test set. With the default settings
(min_resource=15, max_resource=100, eta=3), rungs are at epochs 15, 45,
and 100 — roughly 1/9 of trials run to completion.

Key differences vs phase3_smac:
  - Training: train_epoch() called in a loop (not train_model black-box)
  - Pruning:  Hyperband prunes ~26/27 trials before epoch 100
  - Resume:   SQLite study (load_if_exists=True) + results.pkl cache
  - Warm-start: injected as complete Optuna trials via study.add_trial()

Usage (from project root):
    python -m hpo.phase3_optuna.run \\
        --model_type DA --dataset BAC_OG --seed 42 \\
        --data_dir /scratch/leuven/383/vsc38329/Thesis_Final \\
        --results_dir /scratch/leuven/383/vsc38329/Thesis_Final/results

Smoke-test (num_epochs_override overrides max_resource; 1 epoch disables
pruning and lets all trials complete):
    python -m hpo.phase3_optuna.run ... --num_epochs_override 1

Resume after a time-out: resubmit the same command.  The SQLite study
restores all completed/pruned trials automatically.
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
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hpo.phase3_optuna.config import (
    D_MODEL_CHOICES, DATASET_REGISTRY, FIXED, HYPERBAND_MAX_RESOURCE,
    HYPERBAND_MIN_RESOURCE, HYPERBAND_REDUCTION_FACTOR, N_STARTUP_TRIALS,
    NUM_TRIALS, WARMSTART_CAP_PERCENTILE, get_distributions,
)


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 Optuna TPE+Hyperband runner")
    p.add_argument("--model_type",          required=True, choices=["DA", "NDA"])
    p.add_argument("--dataset",             required=True, choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--seed",                type=int, default=42)
    p.add_argument("--data_dir",            type=str, default="")
    p.add_argument("--results_dir",         type=str, default="")
    p.add_argument("--num_epochs_override", type=int, default=None,
                   help="Override max training epochs for smoke-testing (e.g. 1).")
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
# Data loading  (identical to phase3_smac)
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
# Model creation  (identical to phase3_smac)
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
# Objective helper  (identical to phase3_smac)
# ---------------------------------------------------------------

def compute_objective(val_ttne: float, val_rrt: float, val_dl: float,
                       mu_ttne: float, sigma_ttne: float,
                       mu_rrt: float, sigma_rrt: float,
                       mu_1dl: float, sigma_1dl: float) -> float:
    return float(
        (val_ttne - mu_ttne) / sigma_ttne
        + (val_rrt  - mu_rrt)  / sigma_rrt
        + ((1.0 - val_dl) - mu_1dl) / sigma_1dl
    )


# ---------------------------------------------------------------
# Warm-start helpers  (adapted from phase3_smac)
# ---------------------------------------------------------------

def _clamp_to_space(row) -> dict:
    return {
        "d_model":         int(min(D_MODEL_CHOICES, key=lambda x: abs(x - int(float(row["d_model"]))))),
        "num_layers":      int(max(2, min(8, round(float(row["num_layers"]))))),
        "d_ff_multiplier": int(max(2, min(8, round(float(row["d_ff_multiplier"]))))),
        "learning_rate":   float(max(1e-5, min(1e-2, float(row["learning_rate"])))),
        "dropout":         float(max(0.0,  min(0.7,  float(row["dropout"])))),
        "weight_decay":    float(max(1e-5, min(1e-2, float(row["weight_decay"])))),
    }


def load_warmstart(results_dir: str, model_type: str, dataset: str):
    """Load phase 1 and (optionally) phase 2 CSVs.

    Returns (pairs, mu_ttne, sigma_ttne, mu_rrt, sigma_rrt, mu_1dl, sigma_1dl)
    where pairs is a list of (hp_values_dict, cost) tuples ready to be
    injected into an Optuna study.
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

    df["_r_ttne"] = df["val_MAE_ttne_stand"].rank(method="min")
    df["_r_rrt"]  = df["val_MAE_rrt_stand"].rank(method="min")
    df["_r_1dl"]  = (1.0 - df["val_DL_similarity"]).rank(method="min")
    df["_rank"]   = df["_r_ttne"] + df["_r_rrt"] + df["_r_1dl"]

    n_top   = max(1, len(df) // 2)  # top 50%
    top     = df.nsmallest(n_top, "_rank")
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
            cost = float(
                (row["val_MAE_ttne_stand"] - mu_ttne) / sigma_ttne
                + (row["val_MAE_rrt_stand"]  - mu_rrt)  / sigma_rrt
                + ((1.0 - row["val_DL_similarity"]) - mu_1dl) / sigma_1dl
            )
            pairs.append((hp_values, cost))
        except Exception:
            skipped += 1

    if WARMSTART_CAP_PERCENTILE is not None and pairs:
        costs     = np.array([c for _, c in pairs])
        cap       = float(np.percentile(costs, WARMSTART_CAP_PERCENTILE))
        n_capped  = int(np.sum(costs > cap))
        pairs     = [(hp, min(c, cap)) for hp, c in pairs]
        print(f"  Warm-start: performance cap at {WARMSTART_CAP_PERCENTILE}th pct "
              f"(threshold={cap:.4f}, {n_capped}/{len(pairs)} trials capped)")

    msg = f"  Warm-start: {len(pairs)} trials ready for injection"
    if skipped:
        msg += f" ({skipped} skipped)"
    print(msg)
    return pairs, mu_ttne, sigma_ttne, mu_rrt, sigma_rrt, mu_1dl, sigma_1dl


def inject_warmstart(study, pairs: list, distributions: dict):
    """Add (hp_values, cost) pairs to an Optuna study as COMPLETE trials."""
    import optuna
    from optuna.trial import create_trial

    injected = 0
    skipped  = 0
    for hp_values, cost in pairs:
        try:
            trial = create_trial(
                params=hp_values,
                distributions=distributions,
                value=cost,
            )
            study.add_trial(trial)
            injected += 1
        except Exception as e:
            skipped += 1

    msg = f"  Warm-start: {injected} prior trials injected into Optuna study"
    if skipped:
        msg += f" ({skipped} skipped — distribution mismatch)"
    print(msg)


# ---------------------------------------------------------------
# Objective closure (epoch-by-epoch, Hyperband-compatible)
# ---------------------------------------------------------------

def make_objective(model_type, meta, train_ds, val_ds, test_ds,
                   seed, optuna_search_dir,
                   mu_ttne, sigma_ttne, mu_rrt, sigma_rrt, mu_1dl, sigma_1dl,
                   num_epochs):
    """Return a closure that Optuna calls once per trial.

    Trains epoch-by-epoch so HyperbandPruner can kill unpromising trials
    at rungs (epochs 5, 15, 45, 100 by default).

    Only trials that complete all epochs without pruning are written to
    results.pkl and evaluated on the test set.
    """
    import optuna
    from SuTraN.inference_procedure import inference_loop
    from SuTraN.train_procedure import train_epoch
    from SuTraN.train_utils import MultiOutputLoss

    def objective(trial):
        trial_dir   = os.path.join(optuna_search_dir, f"trial_{trial.number:04d}")
        results_pkl = os.path.join(trial_dir, "results.pkl")

        # Resume: if this trial was fully completed in a previous run,
        # return the cached objective without retraining.
        if os.path.exists(results_pkl):
            with open(results_pkl, "rb") as f:
                cached = pickle.load(f)
            if "objective" in cached:
                return cached["objective"]

        os.makedirs(trial_dir, exist_ok=True)

        # ── 1. Sample HPs ──
        d_model         = trial.suggest_categorical("d_model",         D_MODEL_CHOICES)
        num_layers      = trial.suggest_int("num_layers",              2, 8)
        d_ff_multiplier = trial.suggest_int("d_ff_multiplier",         2, 8)
        learning_rate   = trial.suggest_float("learning_rate",   1e-5, 1e-2, log=True)
        dropout         = trial.suggest_float("dropout",          0.0,  0.7)
        weight_decay    = trial.suggest_float("weight_decay",    1e-5, 1e-2, log=True)

        params = {
            "d_model":         d_model,
            "num_layers":      num_layers,
            "d_ff_multiplier": d_ff_multiplier,
            "d_ff":            d_model * d_ff_multiplier,
            "num_heads":       max(1, d_model // 4),
            "learning_rate":   learning_rate,
            "dropout":         dropout,
            "weight_decay":    weight_decay,
        }

        # ── 2. Seed ──
        trial_seed = seed + trial.number
        set_seed(trial_seed)

        # ── 3. Build model / optimizer / scheduler / loss ──
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model     = create_model(model_type, meta, params)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=learning_rate,
                                      weight_decay=weight_decay)
        lr_sched  = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=FIXED["lr_decay"])
        loss_fn   = MultiOutputLoss(meta["num_activities"], True, False)

        num_cat_pref = 1 if model_type == "NDA" else meta["num_categoricals_pref"]

        train_loader = DataLoader(
            train_ds,
            batch_size=FIXED["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )

        # ── 4. Epoch loop ──
        best_composite = float("inf")
        best_epoch     = 0
        best_val       = {}
        best_ckpt_path = os.path.join(trial_dir, "best_model.pt")

        for epoch in range(num_epochs):
            torch.manual_seed(epoch)   # deterministic shuffle each epoch

            model.train()
            model, optimizer, _ = train_epoch(
                model=model,
                training_loader=train_loader,
                remaining_runtime_head=True,
                outcome_bool=False,
                optimizer=optimizer,
                loss_fn=loss_fn,
                batch_interval=FIXED["batch_interval"],
                epoch_number=epoch,
                max_norm=FIXED["max_norm"],
            )

            model.eval()
            val_inf = inference_loop(
                model, val_ds,
                remaining_runtime_head=True,
                outcome_bool=False,
                num_categoricals_pref=num_cat_pref,
                mean_std_ttne=meta["mean_std_ttne"],
                mean_std_tsp=meta["mean_std_tsp"],
                mean_std_tss=meta["mean_std_tss"],
                mean_std_rrt=meta["mean_std_rrt"],
                results_path=None,
                val_batch_size=2048,
            )

            val_ttne_stand = float(val_inf[0])
            val_ttne_min   = float(val_inf[1])
            val_dl         = float(val_inf[2])
            val_rrt_stand  = float(val_inf[9])
            val_rrt_min    = float(val_inf[10])

            composite = compute_objective(val_ttne_stand, val_rrt_stand, val_dl,
                                          mu_ttne, sigma_ttne, mu_rrt, sigma_rrt, mu_1dl, sigma_1dl)

            if composite < best_composite:
                best_composite = composite
                best_epoch     = epoch
                best_val = {
                    "val_MAE_ttne_stand":   val_ttne_stand,
                    "val_MAE_ttne_minutes": val_ttne_min,
                    "val_DL_similarity":    val_dl,
                    "val_MAE_rrt_stand":    val_rrt_stand,
                    "val_MAE_rrt_minutes":  val_rrt_min,
                }
                torch.save({"model_state_dict": model.state_dict(),
                            "epoch": epoch}, best_ckpt_path)

            # Report to Hyperband and check for pruning
            trial.report(composite, epoch)
            if trial.should_prune():
                # Clean up checkpoint if pruned (no test eval needed)
                if os.path.exists(best_ckpt_path):
                    os.remove(best_ckpt_path)
                raise optuna.TrialPruned()

            lr_sched.step()

            print(f"  Trial {trial.number} | Epoch {epoch}/{num_epochs-1} | "
                  f"obj={composite:.4f}  best={best_composite:.4f}  "
                  f"DL={val_dl:.4f}  rrt={val_rrt_stand:.4f}  "
                  f"ttne={val_ttne_stand:.4f}")

        # ── 5. Trial completed — run test evaluation on best checkpoint ──
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model = create_model(model_type, meta, params)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        test_results_path = os.path.join(trial_dir, "test_results")
        os.makedirs(test_results_path, exist_ok=True)

        inf = inference_loop(
            model, test_ds,
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

        # Clean up checkpoint after successful test eval
        if os.path.exists(best_ckpt_path):
            os.remove(best_ckpt_path)

        result = {
            "trial_idx":   trial.number,
            "seed":        trial_seed,
            "best_epoch":  best_epoch,
            "n_epochs":    num_epochs,
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
            **best_val,
            "objective": best_composite,
        }

        with open(results_pkl, "wb") as f:
            pickle.dump(result, f)

        with open(os.path.join(test_results_path, "prefix_length_results_dict.pkl"), "wb") as f:
            pickle.dump(inf[-2], f)
        with open(os.path.join(test_results_path, "suffix_length_results_dict.pkl"), "wb") as f:
            pickle.dump(inf[-1], f)

        trial.set_user_attr("best_epoch",           best_epoch)
        trial.set_user_attr("val_DL_similarity",    best_val["val_DL_similarity"])
        trial.set_user_attr("val_MAE_rrt_stand",    best_val["val_MAE_rrt_stand"])
        trial.set_user_attr("val_MAE_ttne_stand",   best_val["val_MAE_ttne_stand"])

        print(f"  Trial {trial.number} COMPLETE | "
              f"best_epoch={best_epoch}  objective={best_composite:.4f}")

        return best_composite

    return objective


# ---------------------------------------------------------------
# Summary rebuild
# ---------------------------------------------------------------

def rebuild_summary(optuna_search_dir: str):
    pkl_files = sorted(glob.glob(os.path.join(optuna_search_dir, "trial_*", "results.pkl")))
    if not pkl_files:
        print("No completed trial results found.")
        return

    rows = []
    for path in pkl_files:
        with open(path, "rb") as f:
            r = pickle.load(f)
        rows.append(r)

    rows.sort(key=lambda r: r["trial_idx"])

    summary_path = os.path.join(optuna_search_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary written: {summary_path} ({len(rows)} complete trials)")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    import optuna

    args = parse_args()

    repo_root   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir    = args.data_dir    or repo_root
    results_dir = args.results_dir or os.path.join(repo_root, "results")
    num_epochs  = args.num_epochs_override if args.num_epochs_override is not None \
                  else HYPERBAND_MAX_RESOURCE

    # Adjust Hyperband bounds to num_epochs (important for smoke test)
    hb_min = min(HYPERBAND_MIN_RESOURCE, num_epochs)
    hb_max = num_epochs

    log_name           = DATASET_REGISTRY[args.dataset]
    optuna_search_dir  = os.path.join(results_dir, f"sutran_{args.model_type}",
                                      args.dataset, "optuna_search")
    os.makedirs(optuna_search_dir, exist_ok=True)

    print(f"Phase 3 Optuna: {args.model_type} | {args.dataset} | seed={args.seed}")
    print(f"  num_epochs={num_epochs}  num_trials={NUM_TRIALS}")
    print(f"  Hyperband: min_resource={hb_min}  max_resource={hb_max}  "
          f"eta={HYPERBAND_REDUCTION_FACTOR}")
    print()

    meta = load_metadata(log_name, data_dir)
    train_ds, val_ds, test_ds = load_tensors(log_name, data_dir, args.model_type, meta)
    print(f"Data loaded: {meta['num_activities']} activities, "
          f"{len(train_ds)} training instances\n")

    distributions = get_distributions()

    print("Loading warm-start data...")
    warmstart_pairs, mu_ttne, sigma_ttne, mu_rrt, sigma_rrt, mu_1dl, sigma_1dl = load_warmstart(
        results_dir, args.model_type, args.dataset,
    )
    print()

    # ── Build (or reload) Optuna study ──
    storage_path = os.path.join(optuna_search_dir, "optuna_study.db")
    storage_url  = f"sqlite:///{storage_path}"
    study_name   = f"sutran_{args.model_type}_{args.dataset}_s{args.seed}"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=args.seed,
            n_startup_trials=N_STARTUP_TRIALS,
            multivariate=True,
            warn_independent_sampling=False,  # d_model is categorical; suppress fallback warnings
        ),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=hb_min,
            max_resource=hb_max,
            reduction_factor=HYPERBAND_REDUCTION_FACTOR,
        ),
        storage=storage_url,
        load_if_exists=True,
    )

    # Inject warm-start observations if this is a fresh study
    existing_trials = [t for t in study.trials]
    if not existing_trials:
        inject_warmstart(study, warmstart_pairs, distributions)
        print()
    else:
        n_complete = sum(1 for t in existing_trials
                         if t.state.name == "COMPLETE")
        n_pruned   = sum(1 for t in existing_trials
                         if t.state.name == "PRUNED")
        print(f"Resuming study '{study_name}': "
              f"{n_complete} complete, {n_pruned} pruned.\n")

    # ── Run optimisation ──
    objective = make_objective(
        args.model_type, meta, train_ds, val_ds, test_ds,
        args.seed, optuna_search_dir,
        mu_ttne, sigma_ttne, mu_rrt, sigma_rrt, mu_1dl, sigma_1dl, num_epochs,
    )

    study.optimize(
        objective,
        n_trials=NUM_TRIALS,
        gc_after_trial=True,
    )

    # ── Summary ──
    completed = [t for t in study.trials if t.state.name == "COMPLETE"
                 and not str(t.number).startswith("-")]
    pruned    = [t for t in study.trials if t.state.name == "PRUNED"]
    print(f"\nOptimisation finished: {len(completed)} complete, {len(pruned)} pruned.")

    rebuild_summary(optuna_search_dir)
    print(f"\nPhase 3 Optuna complete.  Output: {optuna_search_dir}")


if __name__ == "__main__":
    main()
