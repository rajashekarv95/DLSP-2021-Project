#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=torch
#SBATCH --output=./out/train_fm_%j.out                                                                                                                                                                     
#SBATCH --error=./err/train_fm_%j.err 

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x 

singularity exec --nv \
--overlay /scratch/rv2138/env/conda_new.ext3:ro \
--overlay /scratch/rv2138/student_dataset.sqsh:ro \
/scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
bash -c "
source /ext3/env.sh
python train_fixmatch.py --checkpoint-path /scratch/rv2138/checkpoints/model.pth.tar \ 
--num-epochs 1
"