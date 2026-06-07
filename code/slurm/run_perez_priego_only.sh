#!/bin/bash
#SBATCH --job-name=PP_partition
#SBATCH --account=et97
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=comp
#SBATCH --mail-user=sanjays@monash.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition/logs/PP_partition_%j.out
#SBATCH --error=/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition/logs/PP_partition_%j.err

# =============================================================================
# Pérez-Priego ET Partitioning (R only)
# =============================================================================

echo "Start time: $(date)"

WORK_DIR=/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition
cd $WORK_DIR

# Load R module
module load R/4.4.0-mkl

# Install ncdf4 if missing
Rscript -e 'if (!requireNamespace("ncdf4", quietly=TRUE)) install.packages("ncdf4", repos="https://cloud.r-project.org")'

echo "R: $(which Rscript)"
echo "Running Pérez-Priego partitioning..."

Rscript run_perez_priego.R

echo "Exit status: $?"
echo "End time: $(date)"
