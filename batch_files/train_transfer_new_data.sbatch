#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --mem=48GB
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
python3 train_transfer.py --checkpoint-path /scratch/rv2138/checkpoints/model_transfer_barlow_wide_new_data.pth.tar \
--transfer-path /scratch/rv2138/checkpoints/model_barlow_wide_fixed.pth.tar \
--best-path /scratch/rv2138/checkpoints/model_barlow_wide_best_new_data.pth.tar \
--num-epochs 160 \
--batch-size 128 \
--fine-tune 1 \
--dataset-folder /scratch/rv2138/dataset \
--learning-rate-classifier 0.00001 \
--learning-rate-barlow 0.00001 \
--weight-decay 0.01 \
--wide 1 \
--model-name barlow \
--dropout 0 \
--new-data 1
" 