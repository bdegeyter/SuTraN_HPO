"""
Generate SLURM scripts for Phase 3 (SMAC3 Bayesian optimisation).

One script is created per (model_type, dataset, seed) combination.
Scripts go into slurm_jobs/phase3_smac/.

Run from the project root:
    python -m hpo.phase3_smac.generate_slurm

Submit the generated scripts:
    sbatch slurm_jobs/phase3_smac/p3_DA_BAC_OG_s42.slurm
    sbatch slurm_jobs/phase3_smac/p3_NDA_BAC_OG_s42.slurm

Each job runs all NUM_TRIALS=200 trials sequentially.  If the 72-hour wall
time is reached before all trials complete, simply resubmit the same script
(with --dependency=afterany:JOBID if you want clean chaining).  The runner
detects completed trials on disk and resumes automatically.

After the final job finishes, rebuild the summary with:
    python -m hpo.phase3_smac.rebuild_summary --model_type DA --dataset BAC_OG
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hpo.phase3_smac.config import DATASET_REGISTRY

OUTPUT_DIR = os.path.join(_REPO_ROOT, "slurm_jobs", "phase3_smac")

SCRATCH = "/scratch/leuven/383/vsc38329/Thesis_Final"
DATA    = "/data/leuven/383/vsc38329"

# Add / remove (model_type, dataset, seed) entries here as needed.
JOBS = [
    ("DA",  "BAC_OG",    42),
    ("DA",  "BPIC17_DR", 42),
    ("NDA", "BAC_OG",    42),
]

SLURM_TEMPLATE = """\
#!/usr/bin/env bash
#SBATCH --job-name="p3_{model}_{dataset}_s{seed}"
#SBATCH --cluster="wice"
#SBATCH --partition="gpu"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --account="lp_lirisnlp"
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --mail-type="FAIL"
#SBATCH --mail-user="bert.geyter@student.kuleuven.be"
#SBATCH --chdir="{scratch}"
#SBATCH --output="{data}/phase3_logs/%x.o%A"
#SBATCH --error="{data}/phase3_logs/%x.e%A"

source /data/leuven/383/vsc38329/miniconda3/etc/profile.d/conda.sh
conda activate sutran_env

python -m hpo.phase3_smac.run \\
    --model_type {model} \\
    --dataset {dataset} \\
    --seed {seed} \\
    --data_dir {scratch} \\
    --results_dir {scratch}/results
"""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Note: phase3_logs/ must exist on the cluster before submitting:
    #   mkdir -p /data/leuven/383/vsc38329/phase3_logs

    generated = []
    for model, dataset, seed in JOBS:
        filename = f"p3_{model}_{dataset}_s{seed}.slurm"
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", newline="\n") as f:
            f.write(SLURM_TEMPLATE.format(
                model=model, dataset=dataset, seed=seed,
                scratch=SCRATCH, data=DATA,
            ))
        generated.append(filename)

    print(f"Generated {len(generated)} SLURM scripts in {OUTPUT_DIR}/:")
    for fn in generated:
        print(f"  sbatch slurm_jobs/phase3_smac/{fn}")
    print()
    print("If a job hits the 72h wall time before all 200 trials finish,")
    print("resubmit with a dependency to resume automatically:")
    print("  sbatch --dependency=afterany:JOBID slurm_jobs/phase3_smac/<script>.slurm")
    print()
    print("Remember to create the log directory on the cluster first:")
    print(f"  mkdir -p {DATA}/phase3_logs")


if __name__ == "__main__":
    main()
