#!/bin/bash
#BSUB -n 24                     # 24 cores
#BSUB -W 8:00                   # 8-hour run-time
#BSUB -R "rusage[mem=4000]"     # 4000 MB per core
#BSUB -J phantom
#BSUB -o model.out
#BSUB -e model.err
#BSUB -N

module purge

echo ">>> Open environment";
#source /cluster/home/quintep/myproject_env/bin/activate
#echo ">>> Installing Requirements";
#pip install -r requirements.txt
#echo ">>> Running Code";
python code.py
deactivate