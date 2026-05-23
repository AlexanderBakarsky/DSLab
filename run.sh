/usr/bin/drop-caches
cd ${MAIN_DIR}
source /work/courses/dslab/team4/miniconda3/etc/profile.d/conda.sh
source activate venv
python ${MAIN_DIR}/finetuning.py