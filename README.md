## Overview
This is authors' re-implementation of the CNN filter similarity described in Fig.1 of:  
"[Orthogonal Convolutional Neural Networks](https://arxiv.org/abs/1911.12207)"   
[Jiayun Wang](http://pwang.pw/),&nbsp; [Yubei Chen](https://redwood.berkeley.edu/people/yubei-chen/),&nbsp;  [Rudrasis Chakraborty](https://rudra1988.github.io/),&nbsp; [Stella X. Yu](https://www1.icsi.berkeley.edu/~stellayu/)&nbsp; (UC Berkeley/ICSI)&nbsp; in IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2020

## Important Notes
- The idea of using grad-cam to measure filter similarity was originally from Xudong Wang, as described in [this paper](https://arxiv.org/abs/2009.12021). The idea is implemented by Jiayun Wang and documented in this repo.
- The code is heavily adapted from [grad-cam-pytorch](https://github.com/kazuto1011/grad-cam-pytorch).

## Requirements
* [PyTorch](https://pytorch.org/) (version >= 0.4.1)


## Run the demo
We use ResNet18 as an example:
- Step 1: get the correlation matrix 
```
python main.py -p YOUR_PATH_TO_DATA_FOLDER -a resnet18 -m YOUR_MODEL_PATH
```
YOUR_PATH_TO_DATA_FOLDER refers to the data folder you want to run the evaluation on. For example, it could be ./data/ILSVRC2012/val_sub/.

- Step 2: plot the histogram for visualization
Please refer to [hist_vis.ipynb](hist_vis.ipynb).

## License and Citation
The use of this software is released under [BSD-3](LICENSE).

If you find this repo useful, please consider citing our paper.
```
@inproceedings{wang2019orthogonal,
  title={Orthogonal Convolutional Neural Networks},
  author={Wang, Jiayun and Chen, Yubei and Chakraborty, Rudrasis and Yu, Stella X},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```