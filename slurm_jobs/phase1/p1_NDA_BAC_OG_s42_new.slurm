#!/usr/bin/env bash
#SBATCH --job-name="p1_NDA_BAC_OG_s42_new"
#SBATCH --cluster="wice"
#SBATCH --partition="gpu_a100"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --account="lp_lirisnlp"
#SBATCH --mem=16G
#SBATCH --time=18:00:00
#SBATCH --mail-type="FAIL"
#SBATCH --mail-user="bert.degeyter@student.kuleuven.be"
#SBATCH --chdir="/scratch/leuven/383/vsc38329/Thesis_Final"
#SBATCH --output="/data/leuven/383/vsc38329/phase1_logs/%x.o%A"
#SBATCH --error="/data/leuven/383/vsc38329/phase1_logs/%x.e%A"

source /data/leuven/383/vsc38329/miniconda3/etc/profile.d/conda.sh
conda activate sutran_env

python -m hpo.phase1.run \
    --model_type NDA \
    --dataset BAC_OG \
    --seed 42 \
    --data_dir /scratch/leuven/383/vsc38329/Thesis_Final \
    --results_dir /scratch/leuven/383/vsc38329/Thesis_Final/results \
    --only d_model=256 \
           weight_decay=0 weight_decay=0.00001 weight_decay=0.001 weight_decay=0.01 \
           lr_decay=0.90 lr_decay=0.93 lr_decay=0.98 lr_decay=1.0
