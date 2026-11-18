#!/bin/bash
#SBATCH --job-name=gls_python
#SBATCH -n 1 -c 16 -N 1
#SBATCH -t 24:00:00
#SBATCH --mem=200G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# -----------------------------------------------------------------------------
# Run a Python step of the pipeline with the configuration loaded.
# Usage: sbatch slurm/python_job.sh scripts/<step>.py [arguments]
#        bash   slurm/python_job.sh scripts/<step>.py [arguments]
# -----------------------------------------------------------------------------

GLS_ROOT="${GLS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$GLS_ROOT/scripts/common.sh"

export OMP_NUM_THREADS="$N_JOBS" MKL_NUM_THREADS="$N_JOBS" OPENBLAS_NUM_THREADS="$N_JOBS"
export PYTHONPATH="$GLS_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

[ $# -ge 1 ] || die "usage: python_job.sh <script.py> [arguments]"
log "Running $*"
python3 -u "$@"
log "Finished $1"
