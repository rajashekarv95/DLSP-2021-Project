#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=torch
#SBATCH --output=./out/train_simsiam_%j.out                                                                                                                                                                     
#SBATCH --error=./err/train_simsiam_%j.err 

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x 


singularity exec --nv \
--overlay /scratch/sr5768/env/overlay-50G-10M.ext3:ro \
--overlay /scratch/sr5768/checkpoints/student_dataset.sqsh:ro \
/scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif \
bash -c "source /ext3/env.sh
python3 train_simsiam.py --checkpoint-path /scratch/sr5768/checkpoints/model.pth.tar \
--num-epochs 20 \
--batch-size 64 \
--dataset-folder /dataset \
" 
