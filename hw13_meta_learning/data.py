import glob
import numpy as np
import torch
from PIL.Image import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class Omniglot(Dataset):
  def __init__(self, data_dir, k_way, q_query):
    self.file_list = [f for f in glob.glob(data_dir + "**/character*", recursive=True)]
    self.transform = transforms.Compose([transforms.ToTensor()])
    self.n = k_way + q_query

  def __getitem__(self, idx):
    sample = np.arange(20)
    np.random.shuffle(sample) # 這裡是為了等一下要 random sample 出我們要的 character
    img_path = self.file_list[idx]
    img_list = [f for f in glob.glob(img_path + "**/*.png", recursive=True)]
    img_list.sort()
    imgs = [self.transform(Image.open(img_file)) for img_file in img_list]
    imgs = torch.stack(imgs)[sample[:self.n]] # 每個 character，取出 k_way + q_query 個
    return imgs

  def __len__(self):
    return len(self.file_list)
