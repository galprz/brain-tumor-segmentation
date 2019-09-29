# Brain tumor segmentation using deep learning
<p align="center">
  <img src="https://raw.githubusercontent.com/galprz/brain-tumor-segemntation/master/images/segmentations.png" width="600"/>
</p>

## Intro
This repository contains implementation of the UNet[1], Deep ResUnet[2] Hybrid ResUnet and ONet[3] models.
We tested those models` performance with the dice metric on the brain_tumor_dataset (https://figshare.com/articles/brain_tumor_dataset/1512427).

## Setup
First download the code by git clone this repo:
```bash
git clone https://github.com/galprz/brain-tumor-segemntation
```
Then use conda to install dependencies and setup the environment 
```bash
conda end update -f environment.yml
conda activate brain-tumor-segmentation
```
## Code structure
The src folder contains all of the implementation code.
+ utils/ contains helpers to download unzip and visualize the data.
+ datasets.py file encapsulate the brain_tumor_dataset into pytorch datasets.
+ loss.py and metrics.py contains the loss function and the dice evaluation metric correspondingly.
+ models.py contains all the model implementation
+ trainers.py contains the pytorch trainer implementation for the experiments.

## Experiments
The jupyter ipynb files contain all the experiments that we ran to evaluate the models` performance on our dataset.
To reproduce the results we set the seed to constant value and the dataset will be downloaded automatically if you 
do not have the data folder so you can just rerun those files the reproduce the results.
## Referances
1. U-Net: Convolutional Networks for Biomedical Image Segmentation(https://arxiv.org/abs/1505.04597)
2. Road Extraction by Deep Residual U-Net (https://arxiv.org/abs/1711.10684)
3. Hybrid ResUnet and ONet (https://github.com/galprz/brain-tumor-segemntation/blob/master/report.pdf)