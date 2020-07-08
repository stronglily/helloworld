import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time

from hw7_Architecture_Design.model import StudentNet


def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x


class ImgDataset(Dataset):
    """
    在 Pytorch 中，我們可以利用 torch.utils.data 的 Dataset 及 DataLoader 來"包裝" data，使後續的 training 及 testing 更為方便。
    Dataset 需要 overload 兩個函數：__len__ 及 __getitem__
    __len__ 必須要回傳 dataset 的大小，而 __getitem__ 則定義了當程式利用 [ ] 取值時，dataset 應該要怎麼回傳資料。
    實際上我們並不會直接使用到這兩個函數，但是使用 DataLoader 在 enumerate Dataset 時會使用到，沒有實做的話會在程式運行階段出現 error。
    """
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):  # 回传数据集的大小
        return len(self.x)

    def __getitem__(self, index):  # 定义了用[]取值时，数据集应该怎么回传数据
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


def onetrain():
    model = StudentNet()
    loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # optimizer 使用 Adam
    num_epoch = 10

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零
            train_pred = model(data[0])  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
            batch_loss = loss(train_pred, data[1])  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
            batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
            optimizer.step()  # 以 optimizer 用 gradient 更新參數值

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred = model(data[0])
                batch_loss = loss(val_pred, data[1])

                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                val_loss += batch_loss.item()

            # 將結果 print 出來
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                  (epoch + 1, num_epoch, time.time() - epoch_start_time,
                   train_acc / train_set.__len__(),
                   train_loss / train_set.__len__(),
                   val_acc / val_set.__len__(),
                   val_loss / val_set.__len__()))


def twotrain():
    model_best = StudentNet()
    loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)  # optimizer 使用 Adam
    num_epoch = 10

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0

        model_best.train()
        for i, data in enumerate(train_val_loader):
            optimizer.zero_grad()
            train_pred = model_best(data[0])
            batch_loss = loss(train_pred, data[1])
            batch_loss.backward()
            optimizer.step()

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

            # 將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
              (epoch + 1, num_epoch, time.time() - epoch_start_time,
               train_acc / train_val_set.__len__(), train_loss / train_val_set.__len__()))
    return model_best


if __name__ == '__main__':
    workspace_dir = 'C:\\Users\\Alice\\jupyters\\food-11'
    print("Reading data")
    train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
    print("Size of training data = {}".format(len(train_x)))
    val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
    print("Size of validation data = {}".format(len(val_x)))
    test_x = readfile(os.path.join(workspace_dir, "testing"), False)
    print("Size of Testing data = {}".format(len(test_x)))

    # training 時做 data augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),  # 隨機將圖片水平翻轉
        transforms.RandomRotation(15),  # 隨機旋轉圖片
        transforms.ToTensor(),  # 將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)
    ])
    # testing 時不需做 data augmentation
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    batch_size = 8
    train_set = ImgDataset(train_x, train_y, train_transform)
    val_set = ImgDataset(val_x, val_y, test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 使用training set訓練，並使用validation set尋找好的參數
    onetrain()

    # 得到好的參數後，使用training set和validation set共同訓練（資料量變多，模型效果較好）
    train_val_x = np.concatenate((train_x, val_x), axis=0)
    train_val_y = np.concatenate((train_y, val_y), axis=0)
    train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
    train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)
    # 二次训练
    model_best = twotrain()

    # 利用剛剛 train 好的 model 進行 prediction
    test_set = ImgDataset(test_x, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model_best.eval()
    prediction = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = model_best(data)
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            for y in test_label:
                prediction.append(y)

    # 將結果寫入 csv 檔
    with open("predict.csv", 'w') as f:
        f.write('Id,Category\n')
        for i, y in enumerate(prediction):
            f.write('{},{}\n'.format(i, y))
