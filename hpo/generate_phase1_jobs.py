"""Generate Phase 1 trial list and a single SLURM job-array script.

Reads PHASE1_DATASETS, PHASE1_MODELS, PHASE1_HP_VALUES, PHASE1_SEEDS
from config.py and produces:

  slurm_jobs/phase1_trials.txt   – one line per trial (CLI args)
  slurm_jobs/phase1.slurm        – single SLURM script using job array

To change what runs, edit the lists in hpo/config.py, then re-run:
    python -m hpo.generate_phase1_jobs
    sbatch slurm_jobs/phase1.slurm
"""
import os
import sys
import itertools

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hpo.config import (
    PHASE1_HP_VALUES, PHASE1_SEEDS, PHASE1_DATASETS, PHASE1_MODELS, DEFAULTS,
)

OUTPUT_DIR = os.path.join(_REPO_ROOT, "slurm_jobs")
TRIALS_FILE = os.path.join(OUTPUT_DIR, "phase1_trials.txt")

# ---------------------------------------------------------------------------
# Trial generation
# ---------------------------------------------------------------------------

def generate_trial_lines():
    """Build the CLI args string for every Phase 1 trial."""
    lines = []
    for dataset, model_type in itertools.product(PHASE1_DATASETS, PHASE1_MODELS):
        for hp_name, values in PHASE1_HP_VALUES.items():
            for val in values:
                # For d_model sweep: set num_heads = d_model // 4 (min 1)
                extra = ""
                if hp_name == "d_model":
                    nh = max(1, val // 4)
                    extra = f" --num_heads {nh}"

                for seed in PHASE1_SEEDS:
                    line = (
                        f"--model_type {model_type} "
                        f"--dataset {dataset} "
                        f"--hp_name {hp_name} "
                        f"--{hp_name} {val} "
                        f"--seed {seed}"
                        f"{extra}"
                    )
                    lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# SLURM template
# ---------------------------------------------------------------------------

SLURM_TEMPLATE = """\
#!/usr/bin/env bash
#SBATCH --job-name="phase1_hpo"
#SBATCH --cluster="genius"
#SBATCH --partition="gpu_v100"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --account="lp_lirisnlp"
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --array=0-{max_idx}
#SBATCH --mail-type="FAIL"
#SBATCH --mail-user="bert.degeyter@student.kuleuven.be"
#SBATCH --output="/data/leuven/383/vsc38329/logs/phase1_%A_%a.out"
#SBATCH --error="/data/leuven/383/vsc38329/logs/phase1_%A_%a.err"

source /data/leuven/383/vsc38329/miniconda3/etc/profile.d/conda.sh
conda activate sutran_env
cd /scratch/leuven/383/vsc38329/Thesis_Final

# Read this trial's CLI args from the trials file (line = array index + 1)
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" slurm_jobs/phase1_trials.txt)
echo "Trial $SLURM_ARRAY_TASK_ID: $ARGS"

python -m hpo.run_experiment $ARGS
"""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lines = generate_trial_lines()

    # Write trials file
    with open(TRIALS_FILE, "w", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")

    # Write SLURM script
    slurm_path = os.path.join(OUTPUT_DIR, "phase1.slurm")
    with open(slurm_path, "w", newline="\n") as f:
        f.write(SLURM_TEMPLATE.format(max_idx=len(lines) - 1))

    print(f"Generated {len(lines)} trials → {TRIALS_FILE}")
    print(f"SLURM script         → {slurm_path}")
    print(f"\nTo submit:  sbatch slurm_jobs/phase1.slurm")


if __name__ == "__main__":
    main()
