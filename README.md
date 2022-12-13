# Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point-m2ae-multi-scale-masked-autoencoders/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=point-m2ae-multi-scale-masked-autoencoders)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point-m2ae-multi-scale-masked-autoencoders/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=point-m2ae-multi-scale-masked-autoencoders)

Official implementation of ['Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training'](https://arxiv.org/pdf/2205.14401.pdf).

The paper has been accepted by **NeurIPS 2022**.

## Introduction
Point-M2AE is a strong **M**ulti-scale **M**AE pre-training framework for hierarchical self-supervised learning of 3D point clouds. Unlike the standard transformer in MAE, we modify the encoder and decoder into pyramid architectures to progressively model spatial geometries and capture both fine-grained and high-level semantics of 3D shapes. We design a multi-scale masking strategy to generate consistent visible regions across scales, and reconstruct the masked coordinates from a global-to-local perspective.

<div align="center">
  <img src="pipeline.jpg"/>
</div>

## Point-M2AE Models
The pre-trained Point-M2AE is evaluated by **Linear SVM** on both ModelNet40 and ScanObjectNN datasets, without downstream fine-tuning:
| Task | Dataset | Config | MN40 Acc.| SONN Acc.| Ckpts | Logs |   
| :-----: | :-----: |:-----:| :-----: | :-----:| :-----:|:-----:|
| Pre-training | ShapeNet |[point-m2ae.yaml](./cfgs/pre-training/point-m2ae.yaml)| 92.87% | 72.07% | [best-ckpt.pth](https://drive.google.com/file/d/1mkfoGSp01th9Pctlk_mE0o-5sOb3vQpD/view?usp=sharing) | [log](https://drive.google.com/file/d/1svx_CQ2x8dRDrf9C_jSDIXYYyJO8KG4m/view?usp=sharing) |

After pre-training, we fine-tune Point-M2AE on three downstream tasks:

Coming in a few days.

| Task | Dataset | Config | Acc.| Vote| Ckpts | Logs |   
| :-----: | :-----: |:-----:| :-----:| :-----: | :-----:|:-----:|
| Classification | ModelNet40 (1k)| -|93.43%| 93.96% | - | - |
| Classification | ScanObjectNN |-| 86.43%| -| - | - |
| Segmentation | ShapeNetPart |-| 86.51% | -| - | - |


|  Task | Dataset | Config | 5w10s | 5w20s | 10w10s| 10w20s|     
| :-----: | :-----: |:-----:| :-----: | :-----:|:-----:|:-----:|
|  Few-shot Cls. | ModelNet40 |-| 96.8%|98.3%|92.3%|95.0%|


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
For pre-training and fine-tuning, please follow [DATASET.md](https://github.com/lulutang0608/Point-BERT/blob/master/DATASET.md) to install ShapeNet, ModelNet40, ScanObjectNN, and ShapeNetPart datasets, referring to Point-BERT. Specially for Linear SVM evaluation, download the official [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) dataset and put the unzip folder under `data/`.

The final directory structure should be:
```
│Point-M2AE/
├──cfgs/
├──datasets/
├──data/
│   ├──ModelNet/
│   ├──ModelNetFewshot/
│   ├──modelnet40_ply_hdf5_2048/
│   ├──ScanObjectNN/
│   ├──ShapeNet55-34/
│   ├──shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──...
```

## Get Started

### Pre-training
Point-M2AE is pre-trained on ShapeNet dataset with the config file `cfgs/pre-training/point-m2ae.yaml`. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pre-training/point-m2ae.yaml --exp_name pre-train
```

To evaluate the pre-trained Point-M2AE by **Linear SVM on ModelNet40**, create the folder `ckpts/` and download the [ckpt-best.pth](https://drive.google.com/file/d/1mkfoGSp01th9Pctlk_mE0o-5sOb3vQpD/view?usp=sharing) into it. You will get 92.87% by running:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pre-training/point-m2ae.yaml --exp_name test_svm --test_svm modelnet40 --ckpts ./ckpts/ckpt-best.pth
```
### Fine-tuning
Coming in a few days.

## Acknowledgement
This repo benefits from [Point-BERT](https://github.com/lulutang0608/Point-BERT) and [Point-MAE](https://github.com/Pang-Yatian/Point-MAE). Thanks for their wonderful works.

## Citation
```bash
@article{zhang2022point,
  title={Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training},
  author={Zhang, Renrui and Guo, Ziyu and Gao, Peng and Fang, Rongyao and Zhao, Bin and Wang, Dong and Qiao, Yu and Li, Hongsheng},
  journal={arXiv preprint arXiv:2205.14401},
  year={2022}
}
```

## Contact
If you have any question about this project, please feel free to contact zhangrenrui@pjlab.org.cn.
