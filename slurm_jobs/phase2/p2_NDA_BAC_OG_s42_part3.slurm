#!/usr/bin/env bash
#SBATCH --job-name="p2_NDA_BAC_OG_s42_part3"
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
#SBATCH --mail-user="bert.geyter@student.kuleuven.be"
#SBATCH --chdir="/scratch/leuven/383/vsc38329/Thesis_Final"
#SBATCH --output="/data/leuven/383/vsc38329/phase2_logs/%x.o%A"
#SBATCH --error="/data/leuven/383/vsc38329/phase2_logs/%x.e%A"

source /data/leuven/383/vsc38329/miniconda3/etc/profile.d/conda.sh
conda activate sutran_env

python -m hpo.phase2.run \
    --model_type NDA \
    --dataset BAC_OG \
    --seed 42 \
    --part 3 \
    --data_dir /scratch/leuven/383/vsc38329/Thesis_Final \
    --results_dir /scratch/leuven/383/vsc38329/Thesis_Final/results
