# ACN
> This repo holds code for [ACN: Adversarial Co-training Network for Brain Tumor Segmentation with Missing Modalities](https://arxiv.org/abs/2106.14591). (MICCAI 2021 Accepted)

<!--![](https://github.com/dbader/readme-template/raw/master/header.png)-->

## Usage

### Dataset
> You need to download the [BraTS2018](https://www.med.upenn.edu/sbia/brats2018/registration.html) or other multi-modality datasets into ```<root_dir>/ACN/data```
> The dataset directory should have this basic structure (BraTS as an example):
```
<root_dir>/ACN/data/<DATA_NAME>/*/case_name/*_flair.nii.gz      
<root_dir>/ACN/data/<DATA_NAME>/*/case_name/*_t1.nii.gz   
<root_dir>/ACN/data/<DATA_NAME>/*/case_name/*_t1ce.nii.gz   
<root_dir>/ACN/data/<DATA_NAME>/*/case_name/*_flair.nii.gz
<root_dir>/ACN/data/<DATA_NAME>/*/case_name/*_seg.nii.gz     # groundtruth 
```
### Pre-requsites
```
Python 3.6
Pytorch >= 0.4.1
CUDA 9.0 or higher
```
Please use the command ```pip install -r requirements.txt``` for the dependencies.

### Train/Val
> Run the code for both train and validation on a multi-modality dataset. 
> Note: This is an example for training a model when only T1ce modality is available. 
```
python train_val_ACN.py
```
### Citation
If you find this paper or code useful for your research, please cite our paper:
```
@misc{wang2021acn,
      title={ACN: Adversarial Co-training Network for Brain Tumor Segmentation with Missing Modalities}, 
      author={Yixin Wang and Yang Zhang and Yang Liu and Zihao Lin and Jiang Tian and Cheng Zhong and Zhongchao Shi and Jianping Fan and Zhiqiang He},
      year={2021},
      eprint={2106.14591},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
### Acknowledgement
> This repo is borrowed from the [Reproduction](https://github.com/doublechenching/brats_segmentation-pytorch) of BraTS18 top1's solution and [ADVENT](https://github.com/valeoai/ADVENT)

### TO DO
> This is an initial version, we will re-organize it after the final publication. 
