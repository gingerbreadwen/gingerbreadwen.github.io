# Pytorch深度学习框架入门-数据输入


简要介绍pytorch的数据读取与加载

<!--more-->

# 数据输入

PyTorch提供的原始数据输入方法: `torch.utils.data.DataLoader` 和 `torch.utils.data.Dataset`。

- torch.utils.data.DataLoader
- torch.utils.data.Dataset

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",  # 路径
    train=True,		# 加载的训练集
    download=True,	# 如果路径不包含数据集，则从网络下载数据集
    transform=ToTensor() # 数据集内容转化为张量
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```


