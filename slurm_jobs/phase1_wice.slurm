#!/usr/bin/env bash
#SBATCH --job-name="phase1_hpo"
#SBATCH --cluster="wice"
#SBATCH --partition="gpu_a100"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --account="lp_lirisnlp"
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --array=0-111
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
