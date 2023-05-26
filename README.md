# COMP4901V Final Project
This is the github page for the final project COMP4901V.
This branch is used for submitting the code for COMP4901V Final project.

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

# Code Structure
## 1. Config files
- projects/BEVSegFormer/configs

## 2. Files modified by us
- tools/
- mmsegBEV

## 3. Model Definition
In `mmsegBEV/models`
- `BEVFormer` is defined in `detectors/bevformer.py`
- It get BEV Features and do Segmantation using `BEVSegmentationHead` in `heads/segm.py`
- Other is basically the same as the offical BEVFORMER implementation https://github.com/fundamentalvision/BEVFormer

For more details, please refence to the commit changes.

# Acknowledgement
Our project is completely based on [BEVFormer](https://github.com/fundamentalvision/BEVFormer). 
And the segmentation head is inspired by [BEVFusion](https://github.com/mit-han-lab/bevfusion).
