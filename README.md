# Separate to Adapt: Open Set Domain Adaptation via Progressive Separation
 (CVPR 2019)
Code release for Separate to Adapt: Open Set Domain Adaptation via Progressive Separation (CVPR 2019)

## Dataset
### Office-31 

## Requirements

- python 2.7
- PyTorch 0.4
- Tensorflow >= 1.0
- Tensorlayer >= 1.11
- Tensorboard
- torchvision

## Training

- Download datasets
- Step 1: `python step_1.py`, the known\unknown discriminator is saved as `discriminator_a.pkl`
- Step 2: `python step_2.py`
- Optional: iterate between step 1&2 to achieve better results
- Monitor 
  `tensorboard --logdir .`

## Citation
please cite:
```
@InProceedings{Liu_2019_CVPR,
author = {Liu, Hong and Cao, Zhangjie and Long, Mingsheng and Wang, Jianmin and Yang, Qiang},
title = {Separate to Adapt: Open Set Domain Adaptation via Progressive Separation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
} 
```

## Reference codes
**https://github.com/thuml/easydl**

## Contact
- h-l17@mails.tsinghua.edu.cn
- mingsheng@tsinghua.edu.cn
