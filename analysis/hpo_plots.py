"""
Generate HPO analysis plots from phase-2 LHS and phase-3 SMAC summary CSVs.

Usage (from repo root):

    python -m analysis.hpo_plots --model_type DA --dataset BAC_OG
    python -m analysis.hpo_plots --model_type NDA --dataset BAC_OG
    python -m analysis.hpo_plots --model_type DA --dataset BPIC17_DR

Two kinds of outputs are written to  Visuals/hpo/<model_type>/<dataset>/:

    lhs_ttne.png         — HP vs MAE TTNE,        top-80% of LHS trials
    lhs_dl.png           — HP vs DL similarity,   top-80% of LHS trials
    lhs_rrt.png          — HP vs MAE RRT,          top-80% of LHS trials (DA only)
    smac_convergence.png — per-trial objective and running incumbent trajectory

Each LHS figure contains one subplot per hyperparameter (2 rows × 3 cols):
  • Discrete HPs  (d_model, num_layers, d_ff_multiplier) → box + jitter plot
  • Continuous HPs (learning_rate, weight_decay)          → scatter, log x-axis
  • Continuous HPs (dropout)                              → scatter, linear x-axis

"Top 80%" means the 80% of trials with the best value for that target metric,
so the y-axis is not crowded by very poor runs.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")            # non-interactive backend; works on HPC nodes too
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Hyperparameter metadata
# ---------------------------------------------------------------------------

HP_ORDER = [
    "d_model", "num_layers", "d_ff_multiplier",
    "learning_rate", "dropout", "weight_decay",
]

HP_META = {
    "d_model":         {"label": "d_model",        "xscale": "linear", "discrete": True},
    "num_layers":      {"label": "num_layers",      "xscale": "linear", "discrete": True},
    "d_ff_multiplier": {"label": "d_ff_multiplier", "xscale": "linear", "discrete": True},
    "learning_rate":   {"label": "learning rate",   "xscale": "log",    "discrete": False},
    "dropout":         {"label": "dropout",         "xscale": "linear", "discrete": False},
    "weight_decay":    {"label": "weight decay",    "xscale": "log",    "discrete": False},
}

# ---------------------------------------------------------------------------
# Target metric metadata
# ---------------------------------------------------------------------------

METRIC_META = {
    "test_MAE_ttne_minutes": {
        "label":     "MAE TTNE (minutes)",
        "col_label": "ttne",
        "direction": "lower",    # lower value = better trial
    },
    "test_DL_similarity": {
        "label":     "DL similarity",
        "col_label": "dl",
        "direction": "higher",   # higher value = better trial
    },
    "test_MAE_rrt_minutes": {
        "label":     "MAE RRT (minutes)",
        "col_label": "rrt",
        "direction": "lower",
    },
}

TOP_FRACTION = 0.80   # fraction of trials shown in LHS plots
_RNG_SEED    = 0      # seed for jitter reproducibility


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> pd.DataFrame:
    """
    Read a summary CSV, handling the optional stray non-header first line
    that some SMAC summaries contain (e.g. a bare integer like "5").
    """
    with open(path) as f:
        first = f.readline().strip()
    skiprows = 1 if "," not in first else 0
    return pd.read_csv(path, skiprows=skiprows)


def _top_fraction(df: pd.DataFrame, metric: str, direction: str,
                  frac: float) -> pd.DataFrame:
    """Return the `frac` fraction of rows with the best metric value."""
    n = max(1, int(round(len(df) * frac)))
    return df.nsmallest(n, metric) if direction == "lower" else df.nlargest(n, metric)


# ---------------------------------------------------------------------------
# LHS distribution figures
# ---------------------------------------------------------------------------

def _draw_hp_subplot(ax, df: pd.DataFrame, hp: str, metric: str) -> None:
    """Draw one subplot: HP value on x-axis, target metric on y-axis."""
    info = HP_META[hp]
    rng  = np.random.default_rng(_RNG_SEED)

    if info["discrete"]:
        categories = sorted(df[hp].unique())
        groups     = [df.loc[df[hp] == c, metric].values for c in categories]
        positions  = list(range(len(categories)))

        ax.boxplot(
            groups,
            positions=positions,
            patch_artist=True,
            widths=0.50,
            boxprops=     {"facecolor": "steelblue", "alpha": 0.50, "linewidth": 0.8},
            medianprops=  {"color": "black",         "linewidth": 2.0},
            whiskerprops= {"linewidth": 0.9},
            capprops=     {"linewidth": 0.9},
            flierprops=   {"marker": ""},   # individual points shown via scatter below
        )
        for i, g in enumerate(groups):
            jitter = rng.uniform(-0.18, 0.18, len(g))
            ax.scatter(
                i + jitter, g,
                s=14, color="navy", alpha=0.40, linewidths=0, zorder=4,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels([str(int(float(c))) for c in categories], fontsize=8)

    else:
        if info["xscale"] == "log":
            ax.set_xscale("log")
        ax.scatter(
            df[hp].values, df[metric].values,
            s=16, color="steelblue", alpha=0.55, linewidths=0,
        )

    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)


def plot_lhs(df: pd.DataFrame, metric: str,
             model_type: str, dataset: str, out_path: str) -> None:
    """
    One 2×3 figure with one subplot per hyperparameter, showing the
    relationship between each HP value and the target metric for the
    top-80% of LHS trials (by that metric).
    """
    meta    = METRIC_META[metric]
    df_top  = _top_fraction(df, metric, meta["direction"], TOP_FRACTION)
    n_total = len(df)
    n_shown = len(df_top)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, hp in zip(axes, HP_ORDER):
        _draw_hp_subplot(ax, df_top, hp, metric)
        ax.set_title(HP_META[hp]["label"], fontsize=10, fontweight="bold", pad=4)
        ax.set_ylabel(meta["label"], fontsize=8)
        ax.set_xlabel(HP_META[hp]["label"], fontsize=8)

    fig.suptitle(
        f"LHS random search  —  {model_type} / {dataset}  |  {meta['label']}\n"
        f"(top {int(TOP_FRACTION * 100)}% of trials by this metric, "
        f"n = {n_shown} / {n_total})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# SMAC convergence figure
# ---------------------------------------------------------------------------

def plot_smac_convergence(df: pd.DataFrame,
                          model_type: str, dataset: str, out_path: str) -> None:
    """
    Scatter of per-trial composite objective values overlaid with the
    running incumbent (minimum achieved so far) as a step line.

    The objective is a z-scored sum of three validation metrics (TTNE + RRT +
    1-DL), so lower is better.  Negative values simply mean the trial
    performed below the warm-start mean.

    Y-axis is clipped to the 5th–95th percentile range (+ 20 % padding) so
    that occasional diverged trials do not collapse the interesting region
    to a flat line.  Trials outside the window are still plotted; they
    simply appear at the edge of the axis.
    """
    df  = df.sort_values("trial_idx").reset_index(drop=True)
    obj = df["objective"].values
    incumbent = np.minimum.accumulate(obj)

    # Clip y-axis: lower bound from the 5th percentile (with padding),
    # upper bound capped at 10 so that diverged runs don't waste axis space.
    q05       = np.percentile(obj, 5)
    pad       = 0.5
    y_lo      = q05 - pad
    y_hi      = 10.0
    n_outside  = int(np.sum((obj < y_lo) | (obj > y_hi)))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        df["trial_idx"], np.clip(obj, y_lo, y_hi),
        s=18, color="steelblue", alpha=0.50, linewidths=0,
        label="Trial objective", zorder=2,
    )
    ax.step(
        df["trial_idx"], np.clip(incumbent, y_lo, y_hi), where="post",
        color="crimson", linewidth=2.0,
        label="Running incumbent", zorder=3,
    )

    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Trial index", fontsize=11)
    ax.set_ylabel("Composite objective  (z-scored, lower = better)", fontsize=10)

    title = f"SMAC convergence  —  {model_type} / {dataset}"
    if n_outside:
        title += f"\n({n_outside} trial{'s' if n_outside > 1 else ''} outside y-axis range not shown)"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LHS distribution plots and SMAC convergence plot.",
    )
    parser.add_argument(
        "--model_type", required=True, choices=["DA", "NDA"],
        help="Model variant (DA = dataset-aware, NDA = dataset-agnostic).",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset key matching the results/ folder, e.g. BAC_OG, BPIC17_DR.",
    )
    parser.add_argument(
        "--results_dir", default="",
        help="Path to the results/ directory (default: <repo_root>/results).",
    )
    parser.add_argument(
        "--out_dir", default="",
        help=(
            "Output directory for PNG files "
            "(default: <repo_root>/Visuals/hpo/<model_type>/<dataset>)."
        ),
    )
    args = parser.parse_args()

    repo_root   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = args.results_dir or os.path.join(repo_root, "results")
    out_dir     = args.out_dir or os.path.join(
        repo_root, "Visuals", "hpo", args.model_type, args.dataset,
    )

    base      = os.path.join(results_dir, f"sutran_{args.model_type}", args.dataset)
    lhs_path  = os.path.join(base, "lhs_search",  "summary.csv")
    smac_path = os.path.join(base, "smac_search", "summary.csv")

    print(f"\nHPO plots  |  {args.model_type} / {args.dataset}")
    print(f"  results : {base}")
    print(f"  output  : {out_dir}\n")

    # -- LHS distribution plots ---------------------------------------------
    if os.path.isfile(lhs_path):
        df_lhs = _read_csv(lhs_path)
        for metric, meta in METRIC_META.items():
            if metric not in df_lhs.columns:
                continue
            out_path = os.path.join(out_dir, f"lhs_{meta['col_label']}.png")
            plot_lhs(df_lhs, metric, args.model_type, args.dataset, out_path)
    else:
        print(f"  LHS summary not found: {lhs_path}")

    # -- SMAC convergence ---------------------------------------------------
    if os.path.isfile(smac_path):
        df_smac = _read_csv(smac_path)
        if "objective" in df_smac.columns:
            out_path = os.path.join(out_dir, "smac_convergence.png")
            plot_smac_convergence(df_smac, args.model_type, args.dataset, out_path)
        else:
            print("  SMAC summary has no 'objective' column — skipping convergence plot.")
    else:
        print(f"  SMAC summary not found: {smac_path}")


if __name__ == "__main__":
    main()
