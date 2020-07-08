import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import torch
import torch.nn as nn
from torch import optim


def preprocess(image_list):  # 將圖片的數值介於 0~255 的 int 線性轉為 0～1 的 float。
    """ Normalize Image and Permute (N,H,W,C) to (N,C,H,W)
    Args:
      image_list: List of images (9000, 32, 32, 3)
    Returns:
      image_list: List of images (9000, 3, 32, 32)
    """
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    image_list = (image_list / 255.0) * 2 - 1
    image_list = image_list.astype(np.float32)
    return image_list


class Image_Dataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        images = self.image_list[idx]
        return images


def count_parameters(model, only_trainable=False):  # 計算 model 參數量的（report 會用到）
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def same_seeds(seed):  # 固定訓練的隨機種子（以便 reproduce）
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True


class AE(nn.Module):  # 定義我們的 baseline autoeocoder
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 17, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x = self.decoder(x1)
        return x1, x


if __name__ == '__main__':
    trainX = np.load('./trainX_new.npy')
    trainX_preprocessed = preprocess(trainX)
    img_dataset = Image_Dataset(trainX_preprocessed)
    """
    這個部分就是主要的訓練階段。 我們先將準備好的 dataset 當作參數餵給 dataloader。 
    將 dataloader、model、loss criterion、optimizer 都準備好之後，就可以開始訓練。 
    訓練完成後，我們會將 model 存下來。
    """
    model = AE()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    model.train()
    n_epoch = 10

    same_seeds(0)
    # 準備 dataloader, model, loss criterion 和 optimizer
    img_dataloader = DataLoader(img_dataset, batch_size=8, shuffle=True)

    # 主要的訓練過程
    for epoch in range(n_epoch):
        for data in img_dataloader:
            img = data
            img = img

            output1, output = model(img)
            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), './checkpoints/checkpoint_{}.pth'.format(epoch + 1))

        print('epoch [{}/{}], loss:{:.5f}'.format(epoch + 1, n_epoch, loss.data))

    # 訓練完成後儲存 model
    torch.save(model.state_dict(), './checkpoints/last_checkpoint.pth')
