# Pytorch深度学习框架入门-数据输入


简要介绍pytorch的数据读取与加载

<!--more-->

# 数据输入

## 数据集加载

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

## 自定义数据集

自定义的数据集包含整个数据

- len返回数据的长度
- getitem索引数据
- init读取数据

## 数据加载器

- minibatches
- reshuffle
- multiproccessing 加速数据检索

```python
# Display image and label.
train_features, train_labels = next(iter(train_dataloader)) #使得dataloader由可迭代变为迭代器，并获取特征与标签
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```


