"""
Generate SLURM scripts for Phase 2 (LHS random search).

One script is created per (model_type, dataset, seed, part) combination.
Scripts go into slurm_jobs/phase2/.

Edit JOBS below, then run from the project root:
    python -m hpo.phase2.generate_slurm

Each generated script runs one *part* of the LHS plan.  Submit them
independently (manually) — they are completely parallel with no
dependencies between parts.

Example submission:
    sbatch slurm_jobs/phase2/p2_DA_BPIC17_DR_s42_part0.slurm
    sbatch slurm_jobs/phase2/p2_DA_BPIC17_DR_s42_part1.slurm
    ...

After all parts have finished, merge the summary with:
    python -m hpo.phase2.rebuild_summary --model_type DA --dataset BPIC17_DR
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hpo.phase2.config import DATASET_REGISTRY, NUM_PARTS

OUTPUT_DIR = os.path.join(_REPO_ROOT, "slurm_jobs", "phase2")

SCRATCH = "/scratch/leuven/383/vsc38329/Thesis_Final"
DATA    = "/data/leuven/383/vsc38329"

# ---------------------------------------------------------------------------
# Edit this list to specify which (model_type, dataset, seed) combinations
# to generate scripts for.  One SLURM script per part will be created for
# each entry (so 4 scripts per entry when NUM_PARTS == 4).
# ---------------------------------------------------------------------------
JOBS = [
    ("DA",  "BPIC17_DR", 42),
    ("DA",  "BAC_OG",    42),
    ("NDA", "BAC_OG",    42),
]

SLURM_TEMPLATE = """\
#!/usr/bin/env bash
#SBATCH --job-name="p2_{model}_{dataset}_s{seed}_part{part}"
#SBATCH --cluster="genius"
#SBATCH --partition="gpu_v100"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --account="lp_lirisnlp"
#SBATCH --mem=16G
#SBATCH --time=60:00:00
#SBATCH --mail-type="FAIL"
#SBATCH --mail-user="bert.degeyter@student.kuleuven.be"
#SBATCH --chdir="{scratch}"
#SBATCH --output="{data}/phase2_logs/%x.o%A"
#SBATCH --error="{data}/phase2_logs/%x.e%A"

source /data/leuven/383/vsc38329/miniconda3/etc/profile.d/conda.sh
conda activate sutran_env

python -m hpo.phase2.run \\
    --model_type {model} \\
    --dataset {dataset} \\
    --seed {seed} \\
    --part {part} \\
    --data_dir {scratch} \\
    --results_dir {scratch}/results
"""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Note: phase2_logs/ must exist on the cluster before submitting:
    #   mkdir -p /data/leuven/383/vsc38329/phase2_logs

    generated = []
    for model, dataset, seed in JOBS:
        for part in range(NUM_PARTS):
            filename = f"p2_{model}_{dataset}_s{seed}_part{part}.slurm"
            path = os.path.join(OUTPUT_DIR, filename)
            with open(path, "w", newline="\n") as f:
                f.write(SLURM_TEMPLATE.format(
                    model=model, dataset=dataset, seed=seed,
                    part=part, scratch=SCRATCH, data=DATA,
                ))
            generated.append(filename)

    print(f"Generated {len(generated)} SLURM scripts in {OUTPUT_DIR}/:")
    for fn in generated:
        print(f"  sbatch slurm_jobs/phase2/{fn}")

    print()
    print("After all parts finish, rebuild the full summary with:")
    for model, dataset, seed in JOBS:
        print(f"  python -m hpo.phase2.rebuild_summary "
              f"--model_type {model} --dataset {dataset}")


if __name__ == "__main__":
    main()
