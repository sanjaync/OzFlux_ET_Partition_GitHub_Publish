#!/bin/bash
#SBATCH --job-name=Comp_Report
#SBATCH --account=et97
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --partition=comp
#SBATCH --output=TEA_logs_original/comprehensive_%j.out
#SBATCH --error=TEA_logs_original/comprehensive_%j.err

module load miniforge3
conda activate ismn

cd /home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition

echo "========================================="
echo " Comprehensive Manuscript Generation Run"
echo " Start: $(date)"
echo " Node:  $(hostname)"
echo "========================================="

python3 compile_comprehensive_report.py

echo "========================================="
echo " Complete: $(date)"
echo "========================================="
