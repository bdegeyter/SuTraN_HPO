#!/usr/bin/env bash
#SBATCH --job-name="create_all_data"
#SBATCH --cluster="wice"
#SBATCH --partition="gpu_a100_debug"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account="lp_lirisnlp"
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --mail-type="BEGIN,END,FAIL"
#SBATCH --mail-user="bert.degeyter@student.kuleuven.be"
#SBATCH --output="/data/leuven/383/vsc38329/%x.o%A"
#SBATCH --error="/data/leuven/383/vsc38329/%x.e%A"

source /data/leuven/383/vsc38329/miniconda3/etc/profile.d/conda.sh
conda activate sutran_env
cd /scratch/leuven/383/vsc38329/Thesis_Final_new

echo "Creating all datasets..."
mkdir -p datasets
python data_creation/create_all_data.py
echo "Done!"
