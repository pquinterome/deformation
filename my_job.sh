#!/bin/bash
#BSUB -n 24                     # 24 cores
#BSUB -W 8:00                   # 8-hour run-time
#BSUB -R "rusage[mem=4000]"     # 4000 MB per core
#BSUB -J phantom
#BSUB -o model.out
#BSUB -e model.err
#BSUB -N

module purge

echo ">>> Open conda environemet";
conda activate xcat_phantom
echo ">>> Installing Requirements";
conda run -n xcat_phantom pip install -r /cluster/home/quintep/deformation/requirements.txt
echo ">>> Running Code";
python /cluster/home/quintep/deformation/code.py
deactivatecd