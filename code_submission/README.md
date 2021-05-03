# BarlowMatch: Combining Redundancy Reduction Self-Supervision and Consistency for Semi-Supervised Learning

## Model Training Instructions
Follow the below instructions to replicate the results obtained by us.

Install the libraries and dependencies from requirements.txt by running the following command.
```
pip install -r requirements.txt
```
Run the below commands to replicate the results obtained by us.
### Barlow Twins training
Run the below command to train a Wide Resnet to learn image representations. This makes use of image augmentations and loss as described in the [Barlow Twins Paper](https://arxiv.org/pdf/2103.03230.pdf). To resume training from a checkpoint, set ```train-from-start = 1``` and provide a valid path for the parameter ```checkpoint-path```. Specify ```wide = 1``` to train it on Resnet-18 instead of Wide Resnet-50.
```
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
```
### Supervised training of the classifier
Run the below command to train a classifier on top of the barlow twins backbone. Provide the path of the previously trained model to the parameter ```transfer-path```. The output of this model will be saved in the path provided in ```checkpoint-path``` and the model with best accuracy on validation set will be saved in the path provided in ```best-path```.
```
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
```
### Fine tuning of the model using consistency regularization and pseudo-labelling.
Run the below command to fine tune the model trained by the previous method using consistency regularization. As done previously, provide the path of the previously trained model to the parameter ```transfer-path```. The output of this model will be saved in the path provided in ```checkpoint-path``` and the model with best accuracy on validation set will be saved in the path provided in ```best-path```.
```
python train_fixmatch_transfer.py \
--checkpoint-path $SCRATCH/checkpoints/model_fm_transfer.pth.tar \
--transfer-path $SCRATCH/checkpoints/model_transfer_best.pth.tar \
--best-path $SCRATCH/checkpoints/model_transfer_fm_best.pth.tar \
--num-epochs 400 \
--num-steps 200 \
--train-from-start 1 \
--batch-size 64 \
--fine-tune 1 \
--dataset-folder /dataset \
--learning-rate 0.00001 \
--mu 7 \
--threshold 0.9 \
--lambd 1 \
--momentum 0.9 \
--weight-decay 0.001
```
The model saved at the ```best-path``` will be the final model.

## Generate image IDs to get labels for
Run the below code to get file names of the images we need to request new labels for. For this, the following steps have to followed.
### Get image representations
Run the below command to get image representations for both labelled and unlabeled dataset.
```
python get_img_representations.py \
--checkpoint-path /scratch/rv2138/checkpoints/model_transfer_barlow_best.pth.tar \
--out-path /scratch/rv2138/representations/ \
--batch-size 512 \
--dataset-folder /dataset \
--wide 1 
```
Run the below command to get the image ids using the processes defined in the paper.
```
python get_label_samples.py
```

Run the below command to generate the new dataset.
```
python create_new_dataset.py
```