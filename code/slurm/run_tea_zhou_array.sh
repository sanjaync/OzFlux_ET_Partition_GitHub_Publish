#!/bin/bash
#SBATCH --job-name=TEA_Zhou_Array
#SBATCH --account=et97
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=comp
#SBATCH --array=0-44
#SBATCH --output=/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition/logs/tea_zhou_%A_%a.out
#SBATCH --error=/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition/logs/tea_zhou_%A_%a.err

WORK_DIR=/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition
cd $WORK_DIR

# Get list of L6 sites (excluding daily/monthly files)
SITES=($(ls /home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/L6/*_L6.nc | grep -vE "Daily|Monthly|Annual" | xargs -n 1 basename | sed "s/_L6.nc//"))

SITE=${SITES[$SLURM_ARRAY_TASK_ID]}

echo "=============================================="
echo "TEA & Zhou Partitioning (Corrected Code)"
echo "Array ID: $SLURM_ARRAY_TASK_ID"
echo "Site: $SITE"
echo "Start time: $(date)"
echo "=============================================="

module load miniforge3
conda activate ecosystem-transpiration

# Run TEA and Zhou for this specific site
python run_all_partitioning.py --sites "$SITE" --methods tea zhou --n_jobs 4

echo "Exit status: $?"
echo "End time: $(date)"
