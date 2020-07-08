from torch.utils.data import Dataset, DataLoader
import cv2
import os
import glob
import torchvision.transforms as transforms
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt


'''准备数据'''
class FaceDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(fname)
        img = self.BGR2RGB(img)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


'''数据预处理 将input形状resize到（64,64），并且其value由0~1线性转换到-1~1'''
def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))
    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)]
    )
    dataset = FaceDataset(fnames, transform)
    return dataset


'''提供固定random seed的函數，以便reproduce'''
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


'''模型 使用DCGAN作为baseline model'''
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    """
    input:(N,in_dim)
    output:(N,3,64,64)
    """
    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim*8*4*4, bias=False),
            nn.BatchNorm1d(dim*8*4*4),
            nn.ReLU()
        )
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh()
        )
        self.apply(weights_init)
    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class Discriminator(nn.Module):
    """
    input (N, 3, 64, 64)
    output (N, )
    """
    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2))
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4),
            nn.Sigmoid())
        self.apply(weights_init)
    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y


if __name__ == '__main__':

    '''训练 设定好hyperparameters。准备好dataloader, model, loss criterion, optimizer。'''
    workspace_dir = './'
    # hyperparameters
    batch_size = 64
    z_dim = 100
    lr = 1e-4
    n_epoch = 10
    save_dir = os.path.join(workspace_dir, 'logs')
    os.makedirs(save_dir, exist_ok=True)

    # model
    G = Generator(in_dim=z_dim)
    D = Discriminator(3)
    G.train()
    D.train()

    # loss criterion
    criterion = nn.BCELoss()

    # optimizer
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    same_seeds(0)
    # dataloader (You might need to edit the dataset path if you use extra dataset.)
    dataset = get_dataset(os.path.join(workspace_dir, 'faces'))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)


    # plt.imshow(dataset[10].numpy().transpose(1,2,0))  # 打印一张图片看看


    '''开始训练'''
    # for logging
    z_sample = Variable(torch.randn(100, z_dim))

    for e, epoch in enumerate(range(n_epoch)):
        for i, data in enumerate(dataloader):
            imgs = data
            imgs = imgs

            bs = imgs.size(0)

            """ Train D """
            z = Variable(torch.randn(bs, z_dim))
            r_imgs = Variable(imgs)
            f_imgs = G(z)

            # label
            r_label = torch.ones((bs))
            f_label = torch.zeros((bs))

            # dis
            r_logit = D(r_imgs.detach())
            f_logit = D(f_imgs.detach())

            # compute loss
            r_loss = criterion(r_logit, r_label)
            f_loss = criterion(f_logit, f_label)
            loss_D = (r_loss + f_loss) / 2

            # update model
            D.zero_grad()
            loss_D.backward()
            opt_D.step()

            """ train G """
            # leaf
            z = Variable(torch.randn(bs, z_dim))
            f_imgs = G(z)

            # dis
            f_logit = D(f_imgs)

            # compute loss
            loss_G = criterion(f_logit, r_label)

            # update model
            G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # log
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, loss_D.item(), 100 * loss_D.item(), loss_G.item()),end='')
            # print('\rEpoch [{epoch + 1}/{n_epoch}] {i + 1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}',end='')

        G.eval()
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        filename = os.path.join(save_dir, 'Epoch_{epoch + 1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(' | Save some samples to {filename}.')

        # show generated image
        grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
        G.train()
        if (e + 1) % 5 == 0:
            torch.save(G.state_dict(), os.path.join(workspace_dir, 'dcgan_g.pth'))
            torch.save(D.state_dict(), os.path.join(workspace_dir, 'dcgan_d.pth'))

