import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.sampler import SequentialSampler
import numpy as np


# 將testing的圖片輸入model後，可以得到其重建的圖片，並對兩者取平方差。
# 可以發現inlier的平方差應該與outlier的平方差形成差距明顯的兩群數據。

if task == 'ae':
    if model_type == 'fcn' or model_type == 'vae':
        y = test.reshape(len(test_tmp), -1)
    else:
        y = test

    data = torch.tensor(y, dtype=torch.float)
    test_dataset = TensorDataset(data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    model = torch.load('best_model_{}.pt'.format(model_type), map_location='cuda')

    model.eval()
    reconstructed = list()
    for i, data in enumerate(test_dataloader):
        if model_type == 'cnn':
            img = data[0].transpose(3, 1).cuda()
        else:
            img = data[0].cuda()
        output = model(img)
        if model_type == 'cnn':
            output = output.transpose(3, 1)
        elif model_type == 'vae':
            output = output[0]
        reconstructed.append(output.cpu().detach().numpy())

    reconstructed = np.concatenate(reconstructed, axis=0)
    anomality = np.sqrt(np.sum(np.square(reconstructed - y).reshape(len(y), -1), axis=1))
    y_pred = anomality
    with open('prediction.csv', 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(i + 1, y_pred[i]))
    # score = roc_auc_score(y_label, y_pred, average='micro')
    # score = f1_score(y_label, y_pred, average='micro')
    # print('auc score: {}'.format(score))
