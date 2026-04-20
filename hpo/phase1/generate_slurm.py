"""
Generate one SLURM script per (model_type, dataset, seed) combination
for Phase 1. Scripts go into slurm_jobs/phase1/.

Edit JOBS below, then run:
    python -m hpo.phase1.generate_slurm
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# No global hpo/config needed — all config lives in hpo/phase1/config.py
OUTPUT_DIR = os.path.join(_REPO_ROOT, "slurm_jobs", "phase1")

SCRATCH = "/scratch/leuven/383/vsc38329/Thesis_Final"
DATA    = "/data/leuven/383/vsc38329"

# ---------------------------------------------------------------------------
# Edit this list to specify which jobs to generate
# Each entry: (model_type, dataset, seed)
# ---------------------------------------------------------------------------
JOBS = [
    ("DA",  "BPIC17_DR", 42),
    ("DA",  "BAC_OG",    42),
    ("DA",  "BAC_OG",    43),
    ("NDA", "BAC_OG",    42),
    ("DA",  "BAC_adj",   42),
]

SLURM_TEMPLATE = """\
#!/usr/bin/env bash
#SBATCH --job-name="p1_{model}_{dataset}_s{seed}"
#SBATCH --cluster="genius"
#SBATCH --partition="gpu_v100"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --account="lp_lirisnlp"
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --mail-type="FAIL"
#SBATCH --mail-user="bert.degeyter@student.kuleuven.be"
#SBATCH --chdir="{scratch}"
#SBATCH --output="{data}/phase1_logs/%x.o%A"
#SBATCH --error="{data}/phase1_logs/%x.e%A"

source /data/leuven/383/vsc38329/miniconda3/etc/profile.d/conda.sh
conda activate sutran_env

python -m hpo.phase1.run \\
    --model_type {model} \\
    --dataset {dataset} \\
    --seed {seed} \\
    --data_dir {scratch} \\
    --results_dir {scratch}/results
"""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Note: phase1_logs/ must exist on the cluster before submitting:
    #   mkdir -p /data/leuven/383/vsc38329/phase1_logs

    generated = []
    for model, dataset, seed in JOBS:
        filename = f"p1_{model}_{dataset}_s{seed}.slurm"
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", newline="\n") as f:
            f.write(SLURM_TEMPLATE.format(
                model=model, dataset=dataset, seed=seed, scratch=SCRATCH, data=DATA))
        generated.append(filename)

    print(f"Generated {len(generated)} SLURM scripts in {OUTPUT_DIR}/:")
    for fn in generated:
        print(f"  sbatch slurm_jobs/phase1/{fn}")


if __name__ == "__main__":
    main()
