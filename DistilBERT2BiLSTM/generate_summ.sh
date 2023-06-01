#!/bin/bash
#SBATCH --job-name=generate_summ
#SBATCH --output=generate_summ_output.txt
#SBATCH --error=generate_summ_error.txt
#SBATCH --time=30-10:00:00
#SBATCH --partition=prod
#SBATCH --nodes=2
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shadi.ali.ismael@gmail.com

module load singularity
singularity exec python_container.sif python generate_summ.py
