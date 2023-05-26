# COMP4901V Final Project
This is the github page for the final project COMP4901V.
For grading, please go to `bev` branch.

# Our works:
 1. Add the segmentation head based on the official BEVSegmentation code 
 2. Refractore the official original code to reduce 3d party library dependency, which is an open-mmlab
 3. Fix the dataloader to load the ground truth bird-eye-view segmentation map

# Prerequisites
Please run environment.yml to install all required prerequisites

# Dataset
Please follow the instruction [here](https://github.com/fundamentalvision/BEVFormer/blob/master/docs/prepare_dataset.md)

# Training
To run the code
./tools/dist_train.sh ./projects/BEVSegFormer/configs/bevformer_base.py 4

# Acknowledgement
Our project is completely based on [BEVFormer](https://github.com/fundamentalvision/BEVFormer). 
And the segmentation head is inspired by [BEVFusion](https://github.com/mit-han-lab/bevfusion).
