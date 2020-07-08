import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.data.sampler as sampler
import torchvision
from torchvision import datasets, transforms
import numpy as np
import sys
import os
import random
import json
import matplotlib.pyplot as plt
from hw8_seq2seq.config import configurations
from hw8_seq2seq.data import EN2CNDataset
from hw8_seq2seq.utils import build_model
from hw8_seq2seq.utils import save_model
from hw8_seq2seq.utils import schedule_sampling
from hw8_seq2seq.utils import tokens2sentence, computebleu, infinite_iter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判斷是用 CPU 還是 GPU 執行運算


def train(model, optimizer, train_iter, loss_function, total_steps, summary_steps, train_dataset):
    model.train()
    model.zero_grad()
    losses = []
    loss_sum = 0.0
    for step in range(summary_steps):
        sources, targets = next(train_iter)
        sources, targets = sources.to(device), targets.to(device)
        outputs, preds = model(sources, targets, schedule_sampling())
        # targets 的第一個 token 是 <BOS> 所以忽略
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)
        loss = loss_function(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        loss_sum += loss.item()
        if (step + 1) % 5 == 0:
            loss_sum = loss_sum / 5
            print("\r", "train [{}] loss: {:.3f}, Perplexity: {:.3f}      ".format(total_steps + step + 1, loss_sum, np.exp(loss_sum)), end=" ")
            losses.append(loss_sum)
            loss_sum = 0.0

    return model, optimizer, losses


def test(model, dataloader, loss_function):
  model.eval()
  loss_sum, bleu_score = 0.0, 0.0
  n = 0
  result = []
  for sources, targets in dataloader:
    sources, targets = sources.to(device), targets.to(device)
    batch_size = sources.size(0)
    outputs, preds = model.inference(sources, targets)
    # targets 的第一個 token 是 <BOS> 所以忽略
    outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
    targets = targets[:, 1:].reshape(-1)

    loss = loss_function(outputs, targets)
    loss_sum += loss.item()

    # 將預測結果轉為文字
    targets = targets.view(sources.size(0), -1)
    preds = tokens2sentence(preds, dataloader.dataset.int2word_cn)
    sources = tokens2sentence(sources, dataloader.dataset.int2word_en)
    targets = tokens2sentence(targets, dataloader.dataset.int2word_cn)
    for source, pred, target in zip(sources, preds, targets):
      result.append((source, pred, target))
    # 計算 Bleu Score
    bleu_score += computebleu(preds, targets)

    n += batch_size

  return loss_sum / len(dataloader), bleu_score / n, result


def train_process(config):
    # 準備訓練資料
    train_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'training')
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_iter = infinite_iter(train_loader)
    # 準備檢驗資料
    val_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'validation')
    val_loader = data.DataLoader(val_dataset, batch_size=1)
    # 建構模型
    model, optimizer = build_model(config, train_dataset.en_vocab_size, train_dataset.cn_vocab_size)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)

    train_losses, val_losses, bleu_scores = [], [], []
    total_steps = 0
    while (total_steps < config.num_steps):
        # 訓練模型
        model, optimizer, loss = train(model, optimizer, train_iter, loss_function, total_steps, config.summary_steps,
                                       train_dataset)
        train_losses += loss
        # 檢驗模型
        val_loss, bleu_score, result = test(model, val_loader, loss_function)
        val_losses.append(val_loss)
        bleu_scores.append(bleu_score)

        total_steps += config.summary_steps
        print("\r", "val [{}] loss: {:.3f}, Perplexity: {:.3f}, blue score: {:.3f}       ".format(total_steps, val_loss,
                                                                                                  np.exp(val_loss),
                                                                                                  bleu_score))

        # 儲存模型和結果
        if total_steps % config.store_steps == 0 or total_steps >= config.num_steps:
            save_model(model, optimizer, config.store_model_path, total_steps)
            with open('{config.store_model_path}/output_{total_steps}.txt', 'w') as f:
                for line in result:
                    print(line, file=f)

    return train_losses, val_losses, bleu_scores


if __name__ == '__main__':
    config = configurations()
    print('config:\n', vars(config))
    train_losses, val_losses, bleu_scores = train_process(config)

    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('次數')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.show()

    plt.figure()
    plt.plot(val_losses)
    plt.xlabel('次數')
    plt.ylabel('loss')
    plt.title('validation loss')
    plt.show()

    plt.figure()
    plt.plot(bleu_scores)
    plt.xlabel('次數')
    plt.ylabel('BLEU score')
    plt.title('BLEU score')
    plt.show()

