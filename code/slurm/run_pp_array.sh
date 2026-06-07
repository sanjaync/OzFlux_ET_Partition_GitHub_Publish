#!/bin/bash
#SBATCH --job-name=PP_array
#SBATCH --account=et97
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=comp
#SBATCH --array=1-45
#SBATCH --output=/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition/logs/PP_array_%A_%a.out
#SBATCH --error=/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition/logs/PP_array_%A_%a.err

WORK_DIR=/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition
cd $WORK_DIR

# Map array ID to a specific preprocessed file
FILES=(output/*_preprocessed.nc)
FILE=${FILES[$SLURM_ARRAY_TASK_ID - 1]}
SITE=$(basename "$FILE" _preprocessed.nc)

echo "=============================================="
echo "Pérez-Priego Array Job"
echo "Array ID: $SLURM_ARRAY_TASK_ID"
echo "Site: $SITE"
echo "Start time: $(date)"
echo "=============================================="

# Load R module and specify user library path for installed packages
module load R/4.4.0-mkl
export R_LIBS_USER=~/R/x86_64-pc-linux-gnu-library/4.4

# Run the R script specifically for this site
Rscript run_perez_priego.R "$SITE"

echo "Exit status: $?"
echo "End time: $(date)"
