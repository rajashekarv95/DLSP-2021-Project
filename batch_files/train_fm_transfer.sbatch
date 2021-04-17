#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=5:00:00
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
bash -c "source /ext3/env.sh
python3 train_fixmatch_transfer.py --checkpoint-path /scratch/rv2138/checkpoints/model_fixmatch_transfer.pth.tar \
--transfer-path /scratch/rv2138/checkpoints/model_transfer.pth.tar \
--num-epochs 600 \
--num-steps 400 \
--train-from-start 1 \
--batch-size 64 \
--fine-tune 1 \
--dataset-folder /dataset \
--learning-rate 0.03 \
--mu 7 \
--threshold 0.7 \
--lambd 10 \
--momentum 0.9 \
--weight-decay 0.0001
" 