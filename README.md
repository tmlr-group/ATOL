<h1 align="center">Out-of-distribution Detection Learning with Unreliable Out-of-distribution Sources</h1>

This repo contains the sample code of our proposed ```ATOL``` in our paper: [Out-of-distribution Detection Learning with Unreliable Out-of-distribution Sources](https://github.com/tmlr-group/ATOL) (NeurIPS 2023).

## Required Packages

The following packages are required to be installed:

- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/)
- [Scipy](https://github.com/scipy/scipy)
- [Numpy](http://www.numpy.org/)
- [Sklearn](https://scikit-learn.org/stable/)

All of our experiments are conducted on NVIDIA Tesla A100 GPUs with Python 3.8, PyTorch 1.11, CUDA 12.0 and Torchvision 0.13.

## Pretrained Models

For CIFAR-10/CIFAR-100, pretrained WRN models and data-generative models are provided in folder

```
./ckpt/
```

## Datasets

Please download the datasets in folder

```
./../data/
```

### CIFAR-10/100 as ID dataset

#### Test OOD Datasets 

- [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

- [Places365](http://places2.csail.mit.edu/download.html)

- [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)

- [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)

- [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz)

- [SVHN](http://ufldl.stanford.edu/housenumbers/)

## Fine-tuning and Testing

To train the ATOL model on CIFAR and ImageNet benckmarks, simply run:

- CIFAR-10
```train cifar10
python atol.py --dataset=cifar10 -b=256 -lr=0.005 --mean=5 --std=0.1 --ood_space_size=4 --trade_off=1
```

- CIFAR-100
```train cifar100
python atol.py --dataset=cifar100 -b=256 -lr=0.04 --mean=1.2 --std=0.5 --ood_space_size=4 --trade_off=5
```

## Results
The key results on CIFAR benchmarks are listed in the following table. 
| Methods     | CIFAR-10    | CIFAR-10    | CIFAR-100   | CIFAR-100   |
| ----------- | ----------- | ----------- | ----------- | ----------- |
|             | FPR95       | AUROC       | FRP95       | AUROC       |
| BoundaryGAN | 55.60       | 86.46       | 76.72       | 75.79       |
| ConfGAN     | 31.57       | 93.01       | 74.86       | 77.67       |
| ManifoldGAN | 26.68       | 94.09       | 73.54       | 77.40       |
| G2D         | 31.83       | 91.74       | 70.73       | 79.03       |
| CMG         | 39.83       | 92.83       | 79.60       | 77.51       |
| ***ATOL***  | ***14.66*** | ***97.05*** | ***55.22*** | ***87.24*** |


## Citation

If you find our work useful, please kindly cite our paper.

```bibtex
@inproceedings{
zheng2023atol,
title={Out-of-distribution Detection Learning with Unreliable Out-of-distribution Sources},
author={Haotian Zheng and Qizhou Wang and Zhen Fang and Xiaobo Xia and Feng Liu and Tongliang Liu and Bo Han},
booktitle={NeurIPS},
year={2023},
url={https://openreview.net/forum?id=87Qnneer8l}
}
```





