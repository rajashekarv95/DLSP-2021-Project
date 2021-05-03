#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=n1s8-t4-1
#SBATCH --account=dl12
#SBATCH --time=10:00:00
#SBATCH --output=./out/train_fm_%j.out
#SBATCH --error=./err/train_fm_%j.err
#SBATCH --exclusive
#SBATCH --requeue

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x

cd $HOME/DLSP-2021-PROJECT

singularity exec --nv \
--bind /scratch \
--overlay /scratch/DL21SP/conda_fixed.sqsh:ro \
--overlay /scratch/rv2138/student_dataset.sqsh:ro \
/share/apps/images/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate dev
python train_transfer.py \
--checkpoint-path $SCRATCH/checkpoints/model_transfer.pth.tar \
--transfer-path $SCRATCH/checkpoints/model_barlow.pth.tar \
--best-path $SCRATCH/checkpoints/model_transfer_best.pth.tar \
--num-epochs 160 \
--batch-size 32 \
--fine-tune 1 \
--dataset-folder /dataset \
--learning-rate-classifier 0.00001 \
--learning-rate-barlow 0.00001 \
--weight-decay 0.001 \
--wide 1 \
--model-name barlow \
--dropout 0.1 \
--seed 10
"
