#!/bin/bash
#BSUB -n 24                     # 24 cores
#BSUB -W 8:00                   # 8-hour run-time
#BSUB -R "rusage[mem=4000]"     # 4000 MB per core
#BSUB -J analysis1
#BSUB -o model.out
#BSUB -e model.err
#BSUB -N

module purge

module load utilities/multi
module load readline/7.0
module load gcc/10.2.0
module load cuda/11.5.0
module load python/anaconda/4.6/miniconda/3.7


source /cluster/home/quintep/myproject_env/bin/activate
echo ">>> Installing Requirements";
pip install -r requirements.txt
echo ">>> Running Code";
python /cluster/home/quintep/deformation/code.py
deactivate