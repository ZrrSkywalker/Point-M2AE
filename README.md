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
### Datasets
Coming in a few days.

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
