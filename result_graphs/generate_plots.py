"""
Generate one result graph per test.

Tests (in results/):
  sutran_DA  / BAC_adj  / individual / summary.csv
  sutran_DA  / BAC_OG   / individual / summary.csv
  sutran_DA  / BPIC17_DR/ individual / summary.csv
  sutran_NDA / BAC_OG   / individual / summary.csv

Each graph:
  - 6 lines: test (solid) / val (dashed) × 3 metrics
      MAE-TTNE  (blue,  left y-axis)
      MAE-RRT   (red,   right y-axis 1)
      1-DL sim  (green, right y-axis 2)
  - X-axis: experiments grouped by HP name, sorted numerically within each group
  - Faded horizontal reference lines at the base-model value for each metric
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# ── Paths & test definitions ───────────────────────────────────────────────────
HERE    = Path(__file__).parent
RESULTS = HERE.parent / "results"

TESTS = [
    ("sutran_DA",  "BAC_adj",   "SuTraN-DA  |  BAC Adj"),
    ("sutran_DA",  "BAC_OG",    "SuTraN-DA  |  BAC OG"),
    ("sutran_DA",  "BPIC17_DR", "SuTraN-DA  |  BPIC17-DR"),
    ("sutran_NDA", "BAC_OG",    "SuTraN-NDA |  BAC OG"),
]

# ── Colours ────────────────────────────────────────────────────────────────────
C_TTNE = "#1f77b4"   # blue
C_RRT  = "#d62728"   # red
C_DLS  = "#2ca02c"   # green


def _sort_key(val):
    """Sort numerically where possible, alphabetically otherwise."""
    try:
        return (0, float(val))
    except (ValueError, TypeError):
        return (1, str(val))


def make_plot(variant: str, dataset: str, title: str) -> None:
    csv_path = RESULTS / variant / dataset / "individual" / "summary.csv"
    df = pd.read_csv(csv_path)

    # ── Separate base row ──────────────────────────────────────────────────────
    base_row = df[df["hp_name"] == "base"].iloc[0]
    df_exp   = df[df["hp_name"] != "base"].copy()

    # ── Filter outliers ────────────────────────────────────────────────────────
    outliers = [("d_model", "4"), ("dropout", "0.8")]
    for hp_name, hp_val in outliers:
        df_exp = df_exp[~((df_exp["hp_name"] == hp_name) &
                          (df_exp["hp_value"].astype(str) == hp_val))]

    # ── Sort: group by hp_name, then by hp_value numerically ──────────────────
    df_exp["_sort_key"] = df_exp["hp_value"].apply(_sort_key)
    df_exp = df_exp.sort_values(["hp_name", "_sort_key"]).reset_index(drop=True)

    # ── Derived metric: 1 – DL similarity ────────────────────────────────────
    df_exp["test_1_DL"] = 1.0 - df_exp["test_DL_similarity"]
    df_exp["val_1_DL"]  = 1.0 - df_exp["val_DL_similarity"]

    base_ttne = float(base_row["test_MAE_ttne_minutes"])
    base_rrt  = float(base_row["test_MAE_rrt_minutes"])
    base_dls  = 1.0 - float(base_row["test_DL_similarity"])

    x      = np.arange(len(df_exp))
    labels = (df_exp["hp_name"] + "=" + df_exp["hp_value"].astype(str)).tolist()

    # ── Figure & axes ─────────────────────────────────────────────────────────
    fig_w = max(14, len(df_exp) * 0.72)
    fig, ax1 = plt.subplots(figsize=(fig_w, 6))

    ax2 = ax1.twinx()   # RRT
    ax3 = ax1.twinx()   # 1-DL
    ax3.spines["right"].set_position(("axes", 1.13))

    # ── Plot 6 lines ──────────────────────────────────────────────────────────
    l1, = ax1.plot(x, df_exp["test_MAE_ttne_minutes"], color=C_TTNE, lw=2,
                   marker="o", ms=5, label="TTNE – test")
    l2, = ax1.plot(x, df_exp["val_MAE_ttne_minutes"],  color=C_TTNE, lw=2,
                   marker="o", ms=5, ls="--", label="TTNE – val")

    l3, = ax2.plot(x, df_exp["test_MAE_rrt_minutes"], color=C_RRT, lw=2,
                   marker="s", ms=5, label="RRT – test")
    l4, = ax2.plot(x, df_exp["val_MAE_rrt_minutes"],  color=C_RRT, lw=2,
                   marker="s", ms=5, ls="--", label="RRT – val")

    l5, = ax3.plot(x, df_exp["test_1_DL"], color=C_DLS, lw=2,
                   marker="^", ms=5, label="1-DL – test")
    l6, = ax3.plot(x, df_exp["val_1_DL"],  color=C_DLS, lw=2,
                   marker="^", ms=5, ls="--", label="1-DL – val")

    # ── Base-model reference lines (faded, labelled once in legend) ───────────
    ax1.axhline(base_ttne, color=C_TTNE, lw=1.5, ls=":", alpha=0.35,
                label=f"TTNE base ({base_ttne:.1f})")
    ax2.axhline(base_rrt,  color=C_RRT,  lw=1.5, ls=":", alpha=0.35,
                label=f"RRT base ({base_rrt:.1f})")
    ax3.axhline(base_dls,  color=C_DLS,  lw=1.5, ls=":", alpha=0.35,
                label=f"1-DL base ({base_dls:.3f})")

    # ── Axis labels & spine colours ───────────────────────────────────────────
    ax1.set_ylabel("TTNE MAE (min)",    color=C_TTNE, fontsize=11)
    ax2.set_ylabel("RRT MAE (min)",     color=C_RRT,  fontsize=11)
    ax3.set_ylabel("1 – DL Similarity", color=C_DLS,  fontsize=11)

    ax1.tick_params(axis="y", colors=C_TTNE)
    ax2.tick_params(axis="y", colors=C_RRT)
    ax3.tick_params(axis="y", colors=C_DLS)

    ax1.spines["left"].set_edgecolor(C_TTNE)
    ax2.spines["right"].set_edgecolor(C_RRT)
    ax3.spines["right"].set_edgecolor(C_DLS)

    # ── X-axis ────────────────────────────────────────────────────────────────
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_xlim(-0.5, len(df_exp) - 0.5)
    ax1.set_xlabel("Hyperparameter value", fontsize=11)

    # ── Grid ──────────────────────────────────────────────────────────────────
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.grid(axis="y", which="major", ls=":", alpha=0.35)
    ax1.grid(axis="x", which="major", ls=":", alpha=0.25)

    # ── Vertical separators & group-name labels ───────────────────────────────
    for hp_name, grp in df_exp.groupby("hp_name", sort=False):
        last_idx = grp.index[-1]
        if last_idx < len(df_exp) - 1:
            ax1.axvline(x=last_idx + 0.5, color="grey", lw=0.8,
                        ls="--", alpha=0.45)
        mid = (grp.index[0] + grp.index[-1]) / 2.0
        ax1.text(mid, 1.01, hp_name,
                 transform=ax1.get_xaxis_transform(),
                 ha="center", va="bottom",
                 fontsize=8, fontstyle="italic", color="dimgrey")

    # ── Legend ────────────────────────────────────────────────────────────────
    # Collect lines from all axes so the base-reference entries are included
    all_lines  = [l1, l2, l3, l4, l5, l6]
    # Add base handles directly from the axes
    base_handles = (
        ax1.get_lines()[-1],   # axhline for TTNE
        ax2.get_lines()[-1],   # axhline for RRT
        ax3.get_lines()[-1],   # axhline for 1-DL
    )
    all_lines += list(base_handles)
    all_labels = [l.get_label() for l in all_lines]
    ax1.legend(all_lines, all_labels, loc="upper left",
               fontsize=8, framealpha=0.88, ncol=3)

    # ── Title & save ──────────────────────────────────────────────────────────
    fig.suptitle(f"{title}  |  Individual HP Study", fontsize=13,
                 fontweight="bold", y=1.03)
    fig.tight_layout()

    out_name = f"{variant}_{dataset}_hp_study.png"
    out_path = HERE / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved  →  {out_path}")
    plt.close(fig)


# ── Run all tests ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for variant, dataset, title in TESTS:
        make_plot(variant, dataset, title)
    print("Done.")
