#!/bin/bash
#SBATCH --job-name=Nelson_Replication
#SBATCH --account=et97
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=comp
#SBATCH --output=TEA_logs_original/nelson_%j.out
#SBATCH --error=TEA_logs_original/nelson_%j.err

module load miniforge3
conda activate ismn

cd /home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition

echo "========================================="
echo " Nelson et al. (2020) Replication Run"
echo " Start: $(date)"
echo " Node:  $(hostname)"
echo "========================================="

python3 replicate_nelson_final.py

echo "========================================="
echo " Complete: $(date)"
echo "========================================="
