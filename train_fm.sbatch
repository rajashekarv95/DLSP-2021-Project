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

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL21SP/student_dataset.sqsh /tmp
echo "Dataset is copied to /tmp"

cd $HOME/DLSP-2021-PROJECT

singularity exec --nv \
--bind /scratch \
--overlay /scratch/DL21SP/conda.sqsh:ro \
--overlay /tmp/student_dataset.sqsh:ro \
/share/apps/images/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate dev
python train_fixmatch.py --checkpoint-path $SCRATCH/checkpoints/model.pth \
--batch-size 1 \
--num-epochs 1 \
--num-steps 1 \
--train-from-start 1 \
--dataset-folder /dataset
"
