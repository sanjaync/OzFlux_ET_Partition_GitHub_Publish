#!/bin/bash
#SBATCH --job-name=TEA_Batch_Imp
#SBATCH --output=logs/TEA_Batch_%j.out
#SBATCH --error=logs/TEA_Batch_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=et97
#SBATCH --time=12:00:00
#SBATCH --partition=comp

echo "Start time: $(date)"
echo "Host: $(hostname)"

WORK_DIR=/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition
cd $WORK_DIR

# Load Conda environment
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate ecosystem-transpiration

# Set PYTHONPATH to include current directory (for modified modules)
export PYTHONPATH=$PWD:$PWD/ecosystem-transpiration:$PYTHONPATH

# Define directories
L6_DIR=/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/L6
OUTPUT_DIR=$WORK_DIR/output_final
mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "Running Improved Batch Partitioning for all OzFlux sites..."
echo "Using GPP_SOLO and dynamic 30/60-min logic fixes."

python -u run_all_partitioning.py \
    --l6_dir $L6_DIR \
    --output_dir $OUTPUT_DIR \
    --gpp_variant GPP_SOLO \
    --n_jobs 16

echo "Batch processing complete."
echo "Exit status: $?"
echo "End time: $(date)"
