#!/bin/bash
#SBATCH --job-name=emb2BiLSTM
#SBATCH --output=emb2BiLSTM_output.txt
#SBATCH --error=emb2BiLSTM_error.txt
#SBATCH --time=30-10:00:00
#SBATCH --partition=prod
#SBATCH --nodes=3
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shadi.ali.ismael@gmail.com

module load singularity
singularity exec python_container.sif python emb2BiLSTM.py
