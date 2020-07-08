import os
from torch.utils import data
from torch.utils.data import sampler
from torchvision import datasets
from hw14_life_long_learning.preprocess import get_transform


# 準備 資料集
# MNIST : 一張圖片資料大小:  28∗28∗1 , 灰階 , 10 個種類
# SVHN : 一張圖片資料大小:  32∗32∗3 , RGB , 10 個種類
# USPS : 一張圖片資料大小:  16∗16∗1 , 灰階 , 10 個種類
class Data():
    def __init__(self, path):
        transform = get_transform()

        self.MNIST_dataset = datasets.MNIST(root=os.path.join(path, "MNIST"),
                                            transform=transform,
                                            train=True,
                                            download=True)

        self.SVHN_dataset = datasets.SVHN(root=os.path.join(path, "SVHN"),
                                          transform=transform,
                                          split='train',
                                          download=True)

        self.USPS_dataset = datasets.USPS(root=os.path.join(path, "USPS"),
                                          transform=transform,
                                          train=True,
                                          download=True)

    def get_datasets(self):
        a = [(self.SVHN_dataset, "SVHN"), (self.MNIST_dataset, "MNIST"), (self.USPS_dataset, "USPS")]
        return a


# 建立 Dataloader
# *.train_loader: 拿取訓練集並訓練 \
# *.val_loader: 拿取驗證集並驗測結果 \
class Dataloader():

    def __init__(self, dataset, batch_size, split_ratio=0.1):
        self.dataset = dataset[0]
        self.name = dataset[1]
        train_sampler, val_sampler = self.split_dataset(split_ratio)

        self.train_dataset_size = len(train_sampler)
        self.val_dataset_size = len(val_sampler)

        self.train_loader = data.DataLoader(self.dataset, batch_size=batch_size, sampler=train_sampler)
        self.val_loader = data.DataLoader(self.dataset, batch_size=batch_size, sampler=val_sampler)
        self.train_iter = self.infinite_iter()

    def split_dataset(self, split_ratio):
        data_size = len(self.dataset)
        split = int(data_size * split_ratio)
        indices = list(range(data_size))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = sampler.SubsetRandomSampler(train_idx)
        val_sampler = sampler.SubsetRandomSampler(valid_idx)
        return train_sampler, val_sampler

    def infinite_iter(self):
        it = iter(self.train_loader)
        while True:
            try:
                ret = next(it)
                yield ret
            except StopIteration:
                it = iter(self.train_loader)
