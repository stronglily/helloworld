# Predict and Write to csv file
import os

import pandas as pd
import torch
from hw4RNN.data import TwitterDataset
from hw4RNN.preprocess import Preprocess
from hw4RNN.test import testing
from hw4RNN.utils import load_testing_data


if __name__ == '__main__':
    path_prefix = './'
    model_dir = './'
    # 開始測試模型並做預測
    print("loading testing data ...")
    test_x = load_testing_data(path='../hw4_data/testing_data.txt')
    preprocess = Preprocess(test_x, sen_len, w2v_path="./w2v.model")
    embedding = preprocess.make_embedding(load=True)
    test_x = preprocess.sentence_word2idx()
    test_dataset = TwitterDataset(X=test_x, y=None)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                num_workers = 8)
    print('\nload model ...')
    model = torch.load(os.path.join(model_dir, 'ckpt.model'))
    outputs = testing(batch_size, test_loader, model, device)

    # 寫到csv檔案供上傳kaggle
    tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
    print("save csv ...")
    tmp.to_csv(os.path.join(path_prefix, 'predict.csv'), index=False)
    print("Finish Predicting")
