#!/bin/bash
#BSUB -n 1
#BSUB -W 
#BSUB -R 
#BSUB -J MRI_xcat_phantom
#BSUB -o model.out
#BSUB -e model.err
#BSUB -N 4


module purge
echo ">>> Open environment";
source /cluster/home/quintep/myproject_env/bin/activate
echo ">>> Installing Requirements";
pip install -r requirements.txt
echo ">>> Running Code";
python /cluster/home/quintep/deformation/code.py
deactivate