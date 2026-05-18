# SuTraN — Thesis Extension

This repository accompanies the master's thesis *"A Systematic Hyperparameter Analysis and Optimisation for Transformer-Based Predictive Process Monitoring"* by **Bert De Geyter** and **Pieter Schrooten** (KU Leuven, 2025–2026), supervised by Prof. Dr. Jochen De Weerdt with daily supervision by Brecht Wuyts.

It extends the [original SuTraN codebase](README_SuTraN_Paper.md) by Brecht Wuyts with a two-stage experimental framework that subjects SuTraN to a systematic hyperparameter analysis and a targeted hyperparameter optimisation.

---

## Methodology

The thesis applies two complementary stages to SuTraN across two event logs (BPIC17-DR and BAC) and two model configurations (data-aware and non-data-aware):

**Stage I — Hyperparameter Analysis (LHS + fANOVA)**  
100 SuTraN configurations are sampled over a six-dimensional hyperparameter subgrid using Latin Hypercube Sampling (`hpo/phase2/`). The resulting configuration–performance dataset is fed into the fANOVA framework (`fANOVA analysis/`), which fits a random forest surrogate and decomposes its output variance to quantify the individual and pairwise importance of each hyperparameter.

**Stage II — Hyperparameter Optimisation (SMAC3)**  
SMAC3 Bayesian optimisation (`hpo/phase3_smac/`) is warm-started directly from the LHS dataset collected in Stage I, allowing the full evaluation budget to be spent on targeted search rather than cold-start exploration.

The six hyperparameters studied are: `d_model`, `num_layers`, `d_ff_multiplier`, `learning_rate`, `dropout`, and `weight_decay`. All remaining SuTraN hyperparameters are held at their published baseline values throughout.

> The `hpo/phase1/` module implements a preliminary one-at-a-time sweep that was used to select and prune the hyperparameter subgrid. It is not part of the core methodology.

---

## What was added

### `hpo/` — HPO runners and SLURM scripts

| Module | Method | Role |
|---|---|---|
| `hpo/phase2/` | Latin Hypercube Sampling | Generates the 100-trial configuration–performance dataset for Stage I and II |
| `hpo/phase3_smac/` | SMAC3 Bayesian optimisation | Targeted search warm-started from the LHS data |
| `hpo/phase1/` | One-at-a-time sweep | Preliminary subgrid selection only |

Each module writes per-trial `results.pkl` files and a consolidated `summary.csv` under `results/`. The `rebuild_summary.py` script in each module regenerates the CSV from the pickled results at any time. SLURM job scripts for the KU Leuven HPC cluster (Genius/wICE) can be generated with `generate_slurm.py`.

### `fANOVA analysis/` — Hyperparameter importance notebooks

- `Preprocessing.ipynb` — loads and prepares the LHS summary CSVs
- `RandomForest.ipynb` — fits the random forest surrogate on the LHS results
- `hyperparameter_analysis.ipynb` — runs fANOVA to decompose variance by hyperparameter

---

## Repository structure

```
hpo/
    phase2/         LHS random search (Stage I data collection)
    phase3_smac/    SMAC3 Bayesian optimisation (Stage II)
    phase1/         Preliminary one-at-a-time sweep (subgrid selection)
fANOVA analysis/    Hyperparameter importance notebooks (Stage I analysis)
data_creation/      Dataset preparation scripts
Preprocessing/      Feature engineering and tensor creation (original)
SuTraN/             Model architecture, training and inference (original)
results/            Experiment outputs (summary CSVs only in this repo)
visualization/      Result visualisation utilities (original)
```

---

## Running the experiments

All commands are run from the project root.

**Stage I — LHS sampling** (split across `NUM_PARTS` SLURM jobs for parallelism):
```bash
python -m hpo.phase2.run --model_type DA --dataset BAC_OG --seed 42 --part 0
```

**Stage II — SMAC3 optimisation** (resumes automatically if resubmitted after a timeout):
```bash
python -m hpo.phase3_smac.run --model_type DA --dataset BAC_OG --seed 42
```

To regenerate SLURM scripts for the HPC cluster, edit the `JOBS` list in the relevant `generate_slurm.py` and run it from the project root:
```bash
python -m hpo.phase2.generate_slurm
python -m hpo.phase3_smac.generate_slurm
```

---

## Dependencies

In addition to the original SuTraN dependencies, the HPO pipeline requires:

- `scipy >= 1.7` (LHS sampling)
- `smac >= 2.0` (SMAC3 Bayesian optimisation)
- `ConfigSpace` (search space definition for SMAC3)
- `fanova` (functional ANOVA in the analysis notebooks)

---

## Datasets

| Dataset | CLI name | Configuration |
|---|---|---|
| BPIC17-DR | `BPIC17_DR` | DA only |
| BAC | `BAC_OG` | DA and NDA |

Data preparation scripts are in `data_creation/`.

---

## Original SuTraN

For the full description of the SuTraN model, preprocessing procedure, and benchmark reimplementations, see [README_SuTraN_Paper.md](README_SuTraN_Paper.md).
