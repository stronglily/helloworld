import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import glob
from tqdm import tqdm
import numpy as np
from collections import OrderedDict


def ConvBlock(in_ch, out_ch):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU(),
                         nn.MaxPool2d(kernel_size=2, stride=2))  # 原作者在 paper 裡是說她在 omniglot 用的是 strided convolution
    # 不過這裡我改成 max pool (mini imagenet 才是 max pool)
    # 這並不是你們在 report 第三題要找的 tip


def ConvBlockFunction(x, w, b, w_bn, b_bn):
    x = F.conv2d(x, w, b, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x


class Classifier(nn.Module):
    def __init__(self, in_ch, k_way):
        super(Classifier, self).__init__()
        self.conv1 = ConvBlock(in_ch, 64)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = ConvBlock(64, 64)
        self.logits = nn.Linear(64, k_way)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = nn.Flatten(x)
        x = self.logits(x)
        return x

    def functional_forward(self, x, params):
        '''
        Arguments:
        x: input images [batch, 1, 28, 28]
        params: 模型的參數，也就是 convolution 的 weight 跟 bias，以及 batchnormalization 的  weight 跟 bias
                這是一個 OrderedDict
        '''
        for block in [1, 2, 3, 4]:
            x = ConvBlockFunction(x, params['conv{block}.0.weight'], params[f'conv{block}.0.bias'],
                                  params.get('conv{block}.1.weight'), params.get(f'conv{block}.1.bias'))
        x = x.view(x.shape[0], -1)
        x = F.linear(x, params['logits.weight'], params['logits.bias'])
        return x


def create_label(n_way, k_shot):
    return torch.arange(n_way).repeat_interleave(k_shot).long()


# 我們試著產生 5 way 2 shot 的 label 看看
# create_label(5, 2)

def MAML(model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_step=1, inner_lr=0.4, train=True):
    """
    Args:
    x is the input omniglot images for a meta_step, shape = [batch_size, n_way * (k_shot + q_query), 1, 28, 28]
    n_way: 每個分類的 task 要有幾個 class
    k_shot: 每個類別在 training 的時候會有多少張照片
    q_query: 在 testing 時，每個類別會用多少張照片 update
    """
    criterion = loss_fn
    task_loss = []  # 這裡面之後會放入每個 task 的 loss
    task_acc = []  # 這裡面之後會放入每個 task 的 loss
    for meta_batch in x:
        train_set = meta_batch[:n_way * k_shot]  # train_set 是我們拿來 update inner loop 參數的 data
        val_set = meta_batch[n_way * k_shot:]  # val_set 是我們拿來 update outer loop 參數的 data

        fast_weights = OrderedDict(
            model.named_parameters())  # 在 inner loop update 參數時，我們不能動到實際參數，因此用 fast_weights 來儲存新的參數 θ'

        for inner_step in range(inner_train_steps):  # 這個 for loop 是 Algorithm2 的 line 7~8
            # 實際上我們 inner loop 只有 update 一次 gradients，不過某些 task 可能會需要多次 update inner loop 的 θ'，
            # 所以我們還是用 for loop 來寫
            train_label = create_label(n_way, k_shot).cuda()
            logits = model.functional_forward(train_set, fast_weights)
            loss = criterion(logits, train_label)
            grads = torch.autograd.grad(loss, fast_weights.values(),
                                        create_graph=True)  # 這裡是要計算出 loss 對 θ 的微分 (∇loss)
            fast_weights = OrderedDict((name, param - inner_lr * grad)
                                       for ((name, param), grad) in
                                       zip(fast_weights.items(), grads))  # 這裡是用剛剛算出的 ∇loss 來 update θ 變成 θ'

        val_label = create_label(n_way, q_query).cuda()
        logits = model.functional_forward(val_set, fast_weights)  # 這裡用 val_set 和 θ' 算 logit
        loss = criterion(logits, val_label)  # 這裡用 val_set 和 θ' 算 loss
        task_loss.append(loss)  # 把這個 task 的 loss 丟進 task_loss 裡面
        acc = np.asarray([torch.argmax(logits, -1).cpu().numpy() == val_label.cpu().numpy()]).mean()  # 算 accuracy
        task_acc.append(acc)

    model.train()
    optimizer.zero_grad()
    meta_batch_loss = torch.stack(task_loss).mean()  # 我們要用一整個 batch 的 loss 來 update θ (不是 θ')
    if train:
        meta_batch_loss.backward()
        optimizer.step()
    task_acc = np.mean(task_acc)
    return meta_batch_loss, task_acc
