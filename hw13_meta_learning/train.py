import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from hw13_meta.data import Omniglot
from hw13_meta.model import Classifier, MAML
import numpy as np


def get_meta_batch(meta_batch_size, k_shot, q_query, data_loader, iterator):
  data = []
  for _ in range(meta_batch_size):
    try:
      task_data = iterator.next()  # 一筆 task_data 就是一個 task 裡面的 data，大小是 [n_way, k_shot+q_query, 1, 28, 28]
    except StopIteration:
      iterator = iter(data_loader)
      task_data = iterator.next()
    train_data = task_data[:, :k_shot].reshape(-1, 1, 28, 28)
    val_data = task_data[:, k_shot:].reshape(-1, 1, 28, 28)
    task_data = torch.cat((train_data, val_data), 0)
    data.append(task_data)
  return torch.stack(data).cuda(), iterator


if __name__ == '__main__':
    n_way = 5
    k_shot = 1
    q_query = 1
    inner_train_steps = 1
    inner_lr = 0.4
    meta_lr = 0.001
    meta_batch_size = 32
    max_epoch = 40
    eval_batches = test_batches = 20
    train_data_path = './Omniglot/images_background/'
    test_data_path = './Omniglot/images_evaluation/'

    dataset = Omniglot(train_data_path, k_shot, q_query)
    train_set, val_set = torch.utils.data.random_split(Omniglot(train_data_path, k_shot, q_query), [3200, 656])
    train_loader = DataLoader(train_set,
                              batch_size=n_way,  # 這裡的 batch size 並不是 meta batch size, 而是一個 task裡面會有多少不同的
                              # characters，也就是 few-shot classifiecation 的 n_way
                              num_workers=8,
                              shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_set,
                            batch_size=n_way,
                            num_workers=8,
                            shuffle=True,
                            drop_last=True)
    test_loader = DataLoader(Omniglot(test_data_path, k_shot, q_query),
                             batch_size=n_way,
                             num_workers=8,
                             shuffle=True,
                             drop_last=True)
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    test_iter = iter(test_loader)

    meta_model = Classifier(1, n_way).cuda()
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)
    loss_fn = nn.CrossEntropyLoss().cuda()

    # 开始训练
    for epoch in range(max_epoch):
        print("Epoch %d" % (epoch))
        train_meta_loss = []
        train_acc = []
        for step in tqdm(range(len(train_loader) // (meta_batch_size))):  # 這裡的 step 是一次 meta-gradinet update step
            x, train_iter = get_meta_batch(meta_batch_size, k_shot, q_query, train_loader, train_iter)
            meta_loss, acc = MAML(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn)
            train_meta_loss.append(meta_loss.item())
            train_acc.append(acc)
        print("  Loss    : ", np.mean(train_meta_loss))
        print("  Accuracy: ", np.mean(train_acc))

        # 每個 epoch 結束後，看看 validation accuracy 如何
        # 助教並沒有做 early stopping，同學如果覺得有需要是可以做的
        val_acc = []
        for eval_step in tqdm(range(len(val_loader) // (eval_batches))):
            x, val_iter = get_meta_batch(eval_batches, k_shot, q_query, val_loader, val_iter)
            _, acc = MAML(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_step=3,
                          train=False)  # testing時，我們更新三次 inner-step
            val_acc.append(acc)
        print("  Validation accuracy: ", np.mean(val_acc))

        # 测试
        test_acc = []
        for test_step in tqdm(range(len(test_loader) // (test_batches))):
            x, val_iter = get_meta_batch(test_batches, k_shot, q_query, test_loader, test_iter)
            _, acc = MAML(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_step=3,
                          train=False)  # testing時，我們更新三次 inner-step
            test_acc.append(acc)
        print("  Testing accuracy: ", np.mean(test_acc))

