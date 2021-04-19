#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --time=10:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:3
#SBATCH --job-name=torch
#SBATCH --output=./out/train_simsiam_%j.out                                                                                                                                                                     
#SBATCH --error=./err/train_simsiam_%j.err 

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x 


singularity exec --nv \
--overlay /scratch/rv2138/env/conda_new.ext3:ro \
--overlay /scratch/rv2138/student_dataset.sqsh:ro \
/scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
bash -c "source /ext3/env.sh
python3 train_simsiam.py --checkpoint-path /scratch/sr5768/checkpoints/model.pth.tar \
--num-epochs 1 \
--batch-size 64 \
--num-epochs 120 \
--num-steps 400 \
--train-from-start 1 \
--dataset-folder /dataset \
--learning-rate 0.03 \
--threshold 0.7 \
--lambd 10 \
--momentum 0.9 \
--weight-decay 0.0001
" 