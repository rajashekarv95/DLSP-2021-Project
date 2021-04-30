#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=n1s8-t4-1
#SBATCH --account=dl12
#SBATCH --time=1:00:00
#SBATCH --output=./out/train_fm_%j.out
#SBATCH --error=./err/train_fm_%j.err
#SBATCH --exclusive
#SBATCH --requeue

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x

cd $HOME/DLSP-2021-PROJECT

singularity exec --nv \
--overlay /scratch/DL21SP/conda_fixed.sqsh:ro \
--overlay /scratch/rv2138/student_dataset.sqsh:ro \
/share/apps/images/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
bash -c "source /ext3/env.sh
conda activate dev
python train_fixmatch_transfer.py --checkpoint-path $SCRATCH/checkpoints/model_fixmatch_transfer.pth.tar \
--transfer-path $SCRATCH/model_transfer_barlow_wide_best_1.pth.tar \
--best-path $SCRATCH/checkpoints/model_fixmatch_transfer_best.pth.tar \
--num-epochs 80 \
--num-steps 400 \
--train-from-start 1 \
--batch-size 64 \
--fine-tune 1 \
--dataset-folder /dataset \
--learning-rate 0.00001 \
--mu 7 \
--threshold 0.9 \
--lambd 2 \
--momentum 0.9 \
--weight-decay 0.01
"
