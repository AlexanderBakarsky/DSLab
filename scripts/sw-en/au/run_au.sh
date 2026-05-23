#!/bin/bash
#SBATCH --gpus gb10:1
#SBATCH -A dslab_jobs
#SBATCH -t 4:00:00
#SBATCH --job-name apertus-ft
#SBATCH --output logs/%j.out
#SBATCH --error logs/%j.err


/usr/bin/drop-caches
cd ${MAIN_DIR}
source /work/courses/dslab/team4/miniconda3/etc/profile.d/conda.sh
conda activate venv
export PYTORCH_ALLOC_CONF=expandable_segments:True
python ./investigate.py ./args/alignment_uniformity/au_sw-en_CL.yaml
