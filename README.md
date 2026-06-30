## Inter-Instance Similarity Modeling for Contrastive Learning

### 1. Introduction

This is the official implementation of paper: "Inter-Instance Similarity Modeling for Contrastive Learning".

![Framework](./images/framework.png)

PatchMix is a novel image mix strategy, which mixes multiple images in patch level. The mixed image contains massive local components from multiple images and efficiently simulates rich similarities among natural images in an unsupervised manner. To model rich inter-instance similarities among images, the contrasts between mixed images and original ones, mixed images to mixed ones, and original images to original ones are conducted to optimize the ViT model. Experimental results demonstrate that our proposed method significantly outperforms the previous state-of-the-art on both ImageNet-1K and CIFAR datasets, e.g., 3.0% linear accuracy improvement on ImageNet-1K and 8.7% kNN accuracy improvement on CIFAR100.

[[Paper](https://arxiv.org/abs/2306.12243)]    [[BibTex](#Citation)]    [[Blog(CN)](https://zhuanlan.zhihu.com/p/639240952)]

<img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fvisresearch%2Fpatchmix&count_bg=%23126DE4&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Hits&edge_flat=false" style="text-align:center;vertical-align:middle"/>

### Requirements

```bash
conda create -n patchmix python=3.8
pip install -r requirements.txt
```



### Datasets

Please set the root paths of dataset in the `*.py` configuration file under the directory: `./config/`.
 `CIFAR10`, `CIFAR100` datasets provided by `torchvision`. The root paths of data are set to `/path/to/dataset` . The root path of  `ImageNet-1K (ILSVRC2012)` is `/path/to/ILSVRC2012`



### Self-Supervised Pretraining

#### ViT-Small with 2-node (8-GPU) training

Set hyperparameters, dataset and GPU IDs in `./config/pretrain/vit_small_pretrain.py` and run the following command

```bash
python main_pretrain.py --arch vit-small
```



### kNN Evaluation

Set hyperparameters, dataset and GPU IDs in `./config/knn/knn.py` and run the following command

```bash
python main_knn.py --arch vit-small --pretrained-weights /path/to/pretrained-weights.pth
```



### Linear Evaluation

Set hyperparameters, dataset and GPU IDs in `./config/linear/vit_small_linear.py` and run the following command:

```bash
python main_linear.py --arch vit-small --pretrained-weights /path/to/pretrained-weights.pth
```



### Fine-tuning Evaluation

Set hyperparameters, dataset and GPUs in `./config/finetuning/vit_small_finetuning.py` and run the following command

```bash
python python main_finetune.py --arch vit-small --pretrained-weights /path/to/pretrained-weights.pth
```



### Main Results and Model Weights

#### ImageNet-1K

|     Arch     | Batch size | #Pre-Epoch | Finetuning Accuracy | Linear Probing Accuracy | kNN Accuracy |
|:------------:|:------:|:-----:|:------:|:--------:|:----------------------------------------------------------------------:|
|   ViT-S/16   |  1024  |  300  | 82.8% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/finetune/imagenet-1k/vit-small-300-82.8.pth)) |  77.4% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/linear/imagenet1k/vit-small-300-77.4.pth))  |   73.3% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/pretrain/imagenet1k/vit-small-300-73.3.pth))   |
|   ViT-B/16   |  1024  |  300  | 84.1% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/finetune/imagenet-1k/vit-base-300-84.1.pth)) |  80.2% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/linear/imagenet1k/vit-base-300-80.2.pth))  | 76.2% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/pretrain/imagenet1k/vit-base-300-76.2.pth)) |



#### CIFAR10

|  Arch   | Batch size | #Pre-Epoch |                     Finetuning Accuracy                      |                   Linear Probing Accuracy                    |                         kNN Accuracy                         |
| :-----: | :--------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ViT-T/2 |    512     |    800     | 97.5% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/finetune/cifar10/vit-tiny-800-97.5.pth)) | 94.4% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/linear/cifar10/vit-tiny-800-94.4.pth)) | 92.9% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/pretrain/cifar10/vit-tiny-800-92.9.pth)) |
| ViT-S/2 |    512     |    800     | 98.1% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/finetune/cifar10/vit-small-800-98.1.pth)) | 96.0% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/linear/cifar10/vit-small-800-96.0.pth)) | 94.6% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/pretrain/cifar10/vit-small-800-94.6.pth)) |
| ViT-B/2 |    512     |    800     | 98.3% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/finetune/cifar10/vit-base-800-98.3.pth)) | 96.6% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/linear/cifar10/vit-base-800-96.6.pth)) | 95.8% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/pretrain/cifar10/vit-base-800-95.8.pth)) |



#### CIFAR100

|  Arch   | Batch size | #Pre-Epoch |                     Finetuning Accuracy                      |                   Linear  Probing Accuracy                   |                         kNN Accuracy                         |
| :-----: | :--------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ViT-T/2 |    512     |    800     | 84.9% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/finetune/cifar100/vit-tiny-800-84.6.pth)) | 74.7% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/linear/cifar100/vit-tiny-800-74.7.pth)) | 68.8% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/pretrain/cifar100/vit-tiny-800-68.8.pth)) |
| ViT-S/2 |    512     |    800     | 86.0% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/finetune/cifar100/vit-small-800-86.0.pth)) | 78.7% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/linear/cifar100/vit-small-800-78.7.pth)) | 75.4% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/pretrain/cifar100/vit-small-800-75.4.pth)) |
| ViT-B/2 |    512     |    800     | 86.0% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/finetune/cifar100/vit-tiny-800-84.6.pth)) | 79.7% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/linear/cifar100/vit-base-800-79.7.pth)) | 75.7% ([link](https://huggingface.co/visresearch/PatchMix/blob/main/pretrain/cifar100/vit-base-800-75.7.pth)) |



### The Visualization of Inter-Instance Similarities

![visualization](./images/visualization.png)

The query sample and the image with id 4 in key samples are from the same category. The images with id 3 and 5 come from category similar to query sample.

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### Citation

```bibtex
@article{shen2023inter,
  author  = {Shen, Chengchao and Liu, Dawei and Tang, Hao and Qu, Zhe and Wang, Jianxin},
  title   = {Inter-Instance Similarity Modeling for Contrastive Learning},
  journal = {arXiv preprint arXiv:2306.12243},
  year    = {2023},
}
```

