#!/bin/bash
#SBATCH --job-name=TEA_partition
#SBATCH --account=et97
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=comp
#SBATCH --mail-user=sanjays@monash.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition/logs/TEA_partition_%j.out
#SBATCH --error=/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition/logs/TEA_partition_%j.err

# =============================================================================
# TEA + Zhou ET Partitioning for OzFlux L6 data
# =============================================================================
# Runs:
#   1. Python: TEA (Nelson 2018) + Zhou/uWUE (Zhou 2016) on all 45 sites
#   2. R: Pérez-Priego (2018) on all 45 sites
#
# Submit with:
#   sbatch run_partitioning.sh
#
# Monitor with:
#   squeue -u sanjays
#   tail -f logs/TEA_partition_<jobid>.out
# =============================================================================

echo "=============================================="
echo "TEA/Zhou/Pérez-Priego ET Partitioning"
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=============================================="

# Directory setup
WORK_DIR=/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition
cd $WORK_DIR

# Create output and log directories
mkdir -p output
mkdir -p logs

# Load environment
module load miniforge3
conda activate ecosystem-transpiration

echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# =============================================================================
# Step 1: Run TEA + Zhou (Python)
# =============================================================================
echo "=============================================="
echo "Step 1: Running TEA + Zhou partitioning"
echo "=============================================="

python run_all_partitioning.py \
    --l6_dir /home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/L6 \
    --output_dir output \
    --methods tea zhou \
    --gpp_variant GPP_SOLO \
    --er_variant ER_SOLO \
    --n_jobs $SLURM_CPUS_PER_TASK

TEA_STATUS=$?
echo ""
echo "TEA + Zhou exit status: $TEA_STATUS"

# =============================================================================
# Step 2: Run Pérez-Priego (R)
# =============================================================================
echo ""
echo "=============================================="
echo "Step 2: Running Pérez-Priego partitioning (R)"
echo "=============================================="

# Deactivate conda and load system R module on MASSIVE
conda deactivate
module load R/4.4.0-mkl

echo "R: $(which Rscript)"
echo "R version: $(Rscript --version 2>&1)"

Rscript run_perez_priego.R

PP_STATUS=$?
echo ""
echo "Pérez-Priego exit status: $PP_STATUS"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "COMPLETED"
echo "End time: $(date)"
echo "TEA+Zhou status: $TEA_STATUS"
echo "Pérez-Priego status: $PP_STATUS"
echo "=============================================="
echo ""
echo "Output files:"
ls -la output/
echo ""
echo "Summary files:"
ls -la output/*summary*
