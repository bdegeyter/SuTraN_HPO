#!/usr/bin/env bash
#SBATCH --job-name="p1_DA_BAC_OG_s42_debug"
#SBATCH --cluster="genius"
#SBATCH --partition="gpu_p100_debug"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --account="lp_lirisnlp"
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --mail-type="FAIL"
#SBATCH --mail-user="bert.degeyter@student.kuleuven.be"
#SBATCH --chdir="/scratch/leuven/383/vsc38329/Thesis_Final"
#SBATCH --output="/data/leuven/383/vsc38329/phase1_logs/%x.o%A"
#SBATCH --error="/data/leuven/383/vsc38329/phase1_logs/%x.e%A"

source /data/leuven/383/vsc38329/miniconda3/etc/profile.d/conda.sh
conda activate sutran_env

python -m hpo.phase1.run \
    --model_type DA \
    --dataset BAC_OG \
    --seed 42 \
    --data_dir /scratch/leuven/383/vsc38329/Thesis_Final \
    --results_dir /scratch/leuven/383/vsc38329/Thesis_Final/results
