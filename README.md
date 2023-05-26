# BEVSegFormer 

This branch is used for submitting the code for COMP4901V Final project.

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
