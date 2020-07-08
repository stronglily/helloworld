import json

import torch
import torch.nn as nn
from hw14_life_long_learning.EWC import EWC
from hw14_life_long_learning.MAS import MAS
from hw14_life_long_learning.config import configurations
from hw14_life_long_learning.utils import save_model, build_model, load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normal_train(model, optimizer, task, total_epochs, summary_epochs):
    model.train()
    model.zero_grad()
    ceriation = nn.CrossEntropyLoss()
    losses = []
    loss = 0.0
    for epoch in range(summary_epochs):
        imgs, labels = next(task.train_iter)
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        ce_loss = ceriation(outputs, labels)

        optimizer.zero_grad()
        ce_loss.backward()
        optimizer.step()

        loss += ce_loss.item()
        if (epoch + 1) % 50 == 0:
            loss = loss / 50
            print("\r", "train task {} [{}] loss: {:.3f}      ".format(task.name, (total_epochs + epoch + 1), loss),
                  end=" ")
            losses.append(loss)
            loss = 0.0

    return model, optimizer, losses


def ewc_train(model, optimizer, task, total_epochs, summary_epochs, ewc, lambda_ewc):
    model.train()
    model.zero_grad()
    ceriation = nn.CrossEntropyLoss()
    losses = []
    loss = 0.0
    for epoch in range(summary_epochs):
        imgs, labels = next(task.train_iter)
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        ce_loss = ceriation(outputs, labels)
        total_loss = ce_loss
        ewc_loss = ewc.penalty(model)
        total_loss += lambda_ewc * ewc_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss += total_loss.item()
        if (epoch + 1) % 50 == 0:
            loss = loss / 50
            print("\r", "train task {} [{}] loss: {:.3f}      ".format(task.name, (total_epochs + epoch + 1), loss),
                  end=" ")
            losses.append(loss)
            loss = 0.0

    return model, optimizer, losses


def mas_train(model, optimizer, task, total_epochs, summary_epochs, mas_tasks, lambda_mas, alpha=0.8):
    model.train()
    model.zero_grad()
    ceriation = nn.CrossEntropyLoss()
    losses = []
    loss = 0.0
    for epoch in range(summary_epochs):
        imgs, labels = next(task.train_iter)
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        ce_loss = ceriation(outputs, labels)
        total_loss = ce_loss
        mas_tasks.reverse()
        if len(mas_tasks) > 1:
            preprevious = 1 - alpha
            scalars = [alpha, preprevious]
            for mas, scalar in zip(mas_tasks[:2], scalars):
                mas_loss = mas.penalty(model)
                total_loss += lambda_mas * mas_loss * scalar
        elif len(mas_tasks) == 1:
            mas_loss = mas_tasks[0].penalty(model)
            total_loss += lambda_mas * mas_loss
        else:
            pass

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss += total_loss.item()
        if (epoch + 1) % 50 == 0:
            loss = loss / 50
            print("\r", "train task {} [{}] loss: {:.3f}      ".format(task.name, (total_epochs + epoch + 1), loss),
                  end=" ")
            losses.append(loss)
            loss = 0.0

    return model, optimizer, losses


def val(model, task):
    model.eval()
    correct_cnt = 0
    for imgs, labels in task.val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, pred_label = torch.max(outputs.data, 1)

        correct_cnt += (pred_label == labels.data).sum().item()

    return correct_cnt / task.val_dataset_size


def train_process(model, optimizer, tasks, config):
    task_loss, acc = {}, {}
    for task_id, task in enumerate(tasks):
        print('\n')
        total_epochs = 0
        task_loss[task.name] = []
        acc[task.name] = []
        if config.mode == 'basic' or task_id == 0:
            while (total_epochs < config.num_epochs):
                model, optimizer, losses = normal_train(model, optimizer, task, total_epochs, config.summary_epochs)
                task_loss[task.name] += losses

                for subtask in range(task_id + 1):
                    acc[tasks[subtask].name].append(val(model, tasks[subtask]))

                total_epochs += config.summary_epochs
                if total_epochs % config.store_epochs == 0 or total_epochs >= config.num_epochs:
                    save_model(model, optimizer, config.store_model_path)

        if config.mode == 'ewc' and task_id > 0:
            old_dataloaders = []
            for old_task in range(task_id):
                old_dataloaders += [tasks[old_task].val_loader]
            ewc = EWC(model, old_dataloaders, device)
            while (total_epochs < config.num_epochs):
                model, optimizer, losses = ewc_train(model, optimizer, task, total_epochs, config.summary_epochs, ewc,
                                                     config.lifelong_coeff)
                task_loss[task.name] += losses

                for subtask in range(task_id + 1):
                    acc[tasks[subtask].name].append(val(model, tasks[subtask]))

                total_epochs += config.summary_epochs
                if total_epochs % config.store_epochs == 0 or total_epochs >= config.num_epochs:
                    save_model(model, optimizer, config.store_model_path)

        if config.mode == 'mas' and task_id > 0:
            old_dataloaders = []
            mas_tasks = []
            for old_task in range(task_id):
                old_dataloaders += [tasks[old_task].val_loader]
                mas = MAS(model, old_dataloaders, device)
                mas_tasks += [mas]
            while (total_epochs < config.num_epochs):
                model, optimizer, losses = mas_train(model, optimizer, task, total_epochs, config.summary_epochs,
                                                     mas_tasks, config.lifelong_coeff)
                task_loss[task.name] += losses

                for subtask in range(task_id + 1):
                    acc[tasks[subtask].name].append(val(model, tasks[subtask]))

                total_epochs += config.summary_epochs
                if total_epochs % config.store_epochs == 0 or total_epochs >= config.num_epochs:
                    save_model(model, optimizer, config.store_model_path)

        if config.mode == 'scp' and task_id > 0:
            pass
            ########################################
            ##       TODO 區塊 （ PART 2 ）         ##
            ########################################
            ##    PART 2  implementation 的部份    ##
            ##   你也可以寫別的 regularization 方法  ##
            ##    助教這裡有提供的是  scp    的 作法   ##
            ##     Slicer Cramer Preservation     ##
            ########################################
            ########################################
            ##       TODO 區塊 （ PART 2 ）         ##
            ########################################
    return task_loss, acc


"""
the order is svhn -> mnist -> usps
"""
if __name__ == '__main__':
    mode_list = ['mas', 'ewc', 'basic']

    ## hint: 謹慎的去選擇 lambda 超參數 / ewc: 80~400, mas: 0.1 - 10
    ############################################################################
    #####                           TODO 區塊 （ PART 1 ）                   #####
    ############################################################################
    coeff_list = [0, 0, 0]  ## 你需要在這 微調 lambda 參數, mas, ewc, baseline=0##
    ############################################################################
    #####                           TODO 區塊 （ PART 1 ）                   #####
    ############################################################################

    config = configurations()
    count = 0
    for mode in mode_list:
        config.mode = mode
        config.lifelong_coeff = coeff_list[count]
        print("{} training".format(config.mode))
        model, optimizer, tasks = build_model(config.load_model_path, config.batch_size, config.learning_rate)
        print("Finish build model")
        if config.load_model:
            model, optimizer = load_model(model, optimizer, config.load_model_path)
        task_loss, acc = train_process(model, optimizer, tasks, config)
        with open('./{config.mode}_acc.txt', 'w') as f:
            json.dump(acc, f)
        count += 1



