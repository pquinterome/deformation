#!/bin/bash
#BSUB -n 120                     # 24 cores
#BSUB -W 8:00                   # 8-hour run-time
#BSUB -R "rusage[mem=40000]"     # 4000 MB per core
#BSUB -J phantom
#BSUB -o model.out
#BSUB -e model.err
#BSUB -N
#BSUB -m “pllimphsing5 pllimphsing6”

module purge
#module load cuda

echo ">>> Installing Requirements";
conda run -n xcat pip install -r /cluster/home/quintep/deformation/requirements.txt
echo ">>> Running Code";
python /cluster/home/quintep/deformation/xcat_code.py
echo ">>> End";