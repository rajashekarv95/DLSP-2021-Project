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
python train_barlow.py \
--checkpoint-path $SCRATCH/checkpoints/model_barlow.pth.tar \
--num-epochs 500 \
--batch-size 512 \
--train-from-start 1 \
--dataset-folder /dataset \
--learning-rate 0.05 \
--lambd 0.05 \
--weight-decay 0.000015 \
--wide 1
"
