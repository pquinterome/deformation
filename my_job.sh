#!/bin/bash
#BSUB -n 120                     # 24 cores
#BSUB -gpu "num=4:mode=exclusive_process:mps=no:j_exclusive=yes"
#BSUB -W 80:00                   # 80-hour run-time
#BSUB -R "rusage[mem=240GB]"     # 4000 MB per core
#BSUS -q research
#BSUB -J phantom
#BSUB -o model.out
#BSUB -e model.err
#BSUB -N

#module purge
#module load cuda

#echo ">>> Installing Requirements";
#conda run -n xcat pip install -r /cluster/home/quintep/deformation/requirements.txt
echo ">>> Running Code";
python /cluster/home/quintep/deformation/xcat_code.py
echo ">>> End";