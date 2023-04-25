#!/bin/bash
#BSUB -n 24                     # 24 cores
#BSUB -W 8:00                   # 8-hour run-time
#BSUB -R "rusage[mem=4000]"     # 4000 MB per core
#BSUB -J analysis1
#BSUB -o analysis1.out
#BSUB -e analysis1.err
#BSUB -N

module purge
module load cuda/9.0
source myproject_env/bin/activate
python deformation/code.py
conda deactivate