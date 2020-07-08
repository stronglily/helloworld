import os
import torch
from hw4RNN.data import TwitterDataset
from hw4RNN.model import LSTM_Net
from hw4RNN.preprocess import Preprocess
from hw4RNN.train import training
from hw4RNN.utils import load_training_data

# 本次作業是NLP當中一個簡單的task——句子分類(文本分類)
# 給定一個句子，判斷他有沒有惡意(負面標1，正面標0)
# 数据集有三個檔案，分別是training_label.txt、training_nolabel.txt、testing_data.txt
# training_label.txt：有label的training data(句子配上0 or 1)
# training_nolabel.txt：沒有label的training data(只有句子)，用來做semi-supervise learning
# testing_data.txt：你要判斷testing data裡面的句子是0 or 1


if __name__ == '__main__':
    path_prefix = './'
    # 通過torch.cuda.is_available()的回傳值進行判斷是否有使用GPU的環境，如果有的話device就設為"cuda"，沒有的話就設為"cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 處理好各個data的路徑
    train_with_label = os.path.join(path_prefix, 'training_label.txt')
    train_no_label = os.path.join(path_prefix, 'training_nolabel.txt')
    testing_data = os.path.join(path_prefix, 'testing_data.txt')

    w2v_path = os.path.join(path_prefix, 'w2v_all.model')  # 處理word to vec model的路徑

    # 定義句子長度、要不要固定embedding、batch大小、要訓練幾個epoch、learning rate的值、model的資料夾路徑
    sen_len = 30
    fix_embedding = True  # fix embedding during training
    batch_size = 2
    epoch = 5
    lr = 0.001
    # model_dir = os.path.join(path_prefix, 'model/') # model directory for checkpoint model
    model_dir = path_prefix  # model directory for checkpoint model

    print("loading data ...")  # 把'training_label.txt'跟'training_nolabel.txt'讀進來
    train_x, y = load_training_data(train_with_label)
    train_x_no_label = load_training_data(train_no_label)

    # 對input跟labels做預處理
    preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x = preprocess.sentence_word2idx()
    y = preprocess.labels_to_tensor(y)

    # 製作一個model的對象
    model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=250, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
    model = model.to(device)  # device為"cuda"，model使用GPU來訓練(餵進去的inputs也需要是cuda tensor)

    # 把data分為training data跟validation data(將一部份training data拿去當作validation data)
    X_train, X_val, y_train, y_val = train_x[:190000], train_x[190000:], y[:190000], y[190000:]

    # 把data做成dataset供dataloader取用
    train_dataset = TwitterDataset(X=X_train, y=y_train)
    val_dataset = TwitterDataset(X=X_val, y=y_val)

    # 把data 轉成 batch of tensors
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True,
                                                num_workers = 0)

    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                num_workers = 0)

    # 開始訓練
    training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)
