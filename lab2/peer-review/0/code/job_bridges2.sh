#!/bin/bash
#SBATCH --job-name=lab2-autoencoder
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --account=mth250011p
#SBATCH --output=slurm-%j.out

module load anaconda3
conda activate stat214

cd /ocean/projects/mth250011p/asoh/stat-214/lab2/code
python run_autoencoder.py configs/bridges2.yaml
