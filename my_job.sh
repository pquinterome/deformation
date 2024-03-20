#!/bin/bash
#BSUB -m "pllimphsing5"
#BSUB -n 12                     # 24 cores

#BSUB -W 80:00                   # 80-hour run-time
#BSUB -R "rusage[mem=120GB]"     # 4000 MB per core
#BSUS -q research
#BSUB -J phantom
#BSUB -o model.out
#BSUB -e model.err
#BSUB -N


#module purge
#module avail
#module load cuda

#echo ">>> Installing Requirements";
#conda run -n xcat pip install -r /cluster/home/quintep/deformation/requirements.txt
echo ">>> Running Code";
python /cluster/home/quintep/deformation/model_zeus.py
echo ">>> End";