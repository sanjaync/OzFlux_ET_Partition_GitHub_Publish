#!/bin/bash
#SBATCH --job-name=TEA_ozflux
#SBATCH --account=et97
#SBATCH --partition=comp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=TEA_logs/slurm_all_%j.out
#SBATCH --error=TEA_logs/slurm_all_%j.err

# ── Full run: all 45 OzFlux L6 sites ─────────────────────────────────────────
echo "=== TEA full OzFlux run ==="
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

cd /home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition
mkdir -p TEA_logs TEA_output

module load miniforge3

conda run -p /scratch/et97/sanjays/conda_envs/conda/envs/ecosystem-transpiration \
    python run_TEA_ozflux.py \
        --n_jobs $SLURM_CPUS_PER_TASK "$@"

echo "=== Converting NetCDF to CSV ==="
conda run -p /scratch/et97/sanjays/conda_envs/conda/envs/ecosystem-transpiration \
    python nc_to_csv.py

echo "Done: $(date)"
