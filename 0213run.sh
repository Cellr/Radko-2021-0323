#!/bin/bash
#SBATCH --partition=RM # 



#SBATCH --nodes=1



#SBATCH --ntasks=8



#SBATCH --time=48:00:00



export SLURM_EXPORT_ENV=ALL



#SBATCH --job-name=dedalus0213-20



#SBATCH --output=output.log



#SBATCH --error=error.log



#SBATCH --cpus-per-task=32



#SBATCH --mem=64G


#SBATCH --mail-type=END,FAIL



#SBATCH --mail-user=zekai2@illinois.edu



module load anaconda3

source activate my_dedalus_env

python radko.py


