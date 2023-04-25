#!/bin/bash
#BSUB -n 1
#BSUB -W 
#BSUB -R 
#BSUB -J MRI_xcat_phantom
#BSUB -o model.out
#BSUB -e model.err
#BSUB -N 4
#BSUB -p gpu05,gpu
#BSUB --gres=gpu:1
#BSUB --time=20:00:00

module purge

pip install -r requirements.txt
echo ">>> Open environment";
source /cluster/home/quintep/myproject_env/bin/activate
echo ">>> Installing Requirements";
pip install -r requirements.txt
echo ">>> Running Code";
python /cluster/home/quintep/deformation/code.py
deactivate