# Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training
Official implementation of ['Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training'](https://arxiv.org/pdf/2205.14401.pdf).

The paper has been accepted by **NeurIPS 2022**.

## Introduction
Point-M2AE is a strong **M**ulti-scale **M**AE pre-training framework for hierarchical self-supervised learning of 3D point clouds. Unlike the standard transformer in MAE, we modify the encoder and decoder into pyramid architectures to progressively model spatial geometries and capture both fine-grained and high-level semantics of 3D shapes. We design a multi-scale masking strategy to generate consistent visible regions across scales, and reconstruct the masked coordinates from a global-to-local perspective.

<div align="center">
  <img src="pipeline.jpg"/>
</div>

## Requirements

### Installation
Create a conda environment and install basic dependencies:
```bash
git clone https://github.com/ZrrSkywalker/Point-M2AE.git
cd Point-M2AE

conda create -n pointm2ae python=3.7
conda activate pointm2ae

pip install -r requirements.txt

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
```
Install GPU-related packages:
```bash
# Chamfer Distance and EMD
cd ./extensions/chamfer_dist
python setup.py install --user
cd ../emd
python setup.py install --user

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
### Datasets
Follow [DATASET.md](https://github.com/lulutang0608/Point-BERT/blob/master/DATASET.md) to install ShapeNet, ModelNet40, ScanObjectNN, and ShapeNetPart datasets, referring to Point-BERT.

## Get Started

### Pre-training
Point-M2AE is pre-trained on ShapeNet dataset with the config file `cfgs/pre-training/point-m2ae.yaml`. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pre-training/point-m2ae.yaml --exp_name pre-train
```
To evaluate the pre-trained Point-M2AE by Linear SVM on ModelNet40, create the folder `ckpts/` and download the `ckpt-best.pth` into it. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pre-training/point-m2ae.yaml --exp_name test_svm --test_svm modelnet40 --ckpts ./ckpts/ckpt-best.pth
```
### Fine-tuning
Coming in a few days.

## Contact
If you have any question about this project, please feel free to contact zhangrenrui@pjlab.org.cn.
