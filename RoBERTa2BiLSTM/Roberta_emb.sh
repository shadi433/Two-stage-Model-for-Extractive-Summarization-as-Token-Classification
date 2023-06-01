#!/bin/bash
#SBATCH --job-name=Roberta_emb
#SBATCH --output=Roberta_emb_output.txt
#SBATCH --error=Roberta_emb_error.txt
#SBATCH --time=30-10:00:00
#SBATCH --partition=prod2
#SBATCH --nodes=2
#SBATCH --cpus-per-task=9
#SBATCH --ntasks-per-node=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shadi.ali.ismael@gmail.com

module load singularity
singularity exec python_container.sif python Roberta_emb.py
