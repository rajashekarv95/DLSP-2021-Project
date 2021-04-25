#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:3
#SBATCH --job-name=torch
#SBATCH --output=./out/train_fm_%j.out                                                                                                                                                                     
#SBATCH --error=./err/train_fm_%j.err 

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x 

singularity exec --nv \
--overlay /scratch/sr5768/env/overlay-50G-10M.ext3:ro \
--overlay /scratch/sr5768/student_dataset.sqsh:ro \
/scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif \
bash -c "source /ext3/env.sh
python3 train_simsiam_labeled.py --checkpoint-path /scratch/sr5768/checkpoints/model_train_simsiam_labeled.pth.tar \
--transfer-path /scratch/sr5768/checkpoints/model.pth.tar \
--num-epochs 100 \
--batch-size 128 \
--dataset-folder /dataset \
--learning-rate-classifier 0.00001 \
--learning-rate-model 0.00001 \
--weight-decay 0.01
" 
