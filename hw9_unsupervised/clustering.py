import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader

from hw9.unsupervised import preprocess, Image_Dataset, AE


def cal_acc(gt, pred):
    """ Computes categorization accuracy of our task.
    Args:
      gt: Ground truth labels (9000, )
      pred: Predicted labels (9000, )
    Returns:
      acc: Accuracy (0~1 scalar)
    """
    # Calculate Correct predictions
    correct = np.sum(gt == pred)
    acc = correct / gt.shape[0]
    # 因為是 binary unsupervised clustering，因此取 max(acc, 1-acc)
    return max(acc, 1-acc)


def plot_scatter(feat, label, savefig=None):
    """ Plot Scatter Image.
    Args:
      feat: the (x, y) coordinate of clustering result, shape: (9000, 2)
      label: ground truth label of image (0/1), shape: (9000,)
    Returns:
      None
    """
    X = feat[:, 0]
    Y = feat[:, 1]
    plt.scatter(X, Y, c = label)
    plt.legend(loc='best')
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    return


"""
接著我們使用訓練好的 model，來預測 testing data 的類別。
由於 testing data 與 training data 一樣，因此我們使用同樣的 dataset 來實作 dataloader。
與 training 不同的地方在於 shuffle 這個參數值在這邊是 False。
準備好 model 與 dataloader，我們就可以進行預測了。
我們只需要 encoder 的結果（latents），利用 latents 進行 clustering 之後，就可以分類了。
"""
def inference(X, model, batch_size=8):
    X = preprocess(X)
    dataset = Image_Dataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    for i, x in enumerate(dataloader):
        x = torch.FloatTensor(x)
        vec, img = model(x.cuda())
        if i == 0:
            latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
        else:
            latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis = 0)
    print('Latents Shape:', latents.shape)
    return latents

def predict(latents):
    # First Dimension Reduction
    transformer = KernelPCA(n_components=200, kernel='rbf', n_jobs=-1)
    kpca = transformer.fit_transform(latents)
    print('First Reduction Shape:', kpca.shape)

    # # Second Dimesnion Reduction
    X_embedded = TSNE(n_components=2).fit_transform(kpca)
    print('Second Reduction Shape:', X_embedded.shape)

    # Clustering
    pred = MiniBatchKMeans(n_clusters=2, random_state=0).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred, X_embedded

def invert(pred):
    return np.abs(1-pred)

def save_prediction(pred, out_csv='prediction.csv'):
    with open(out_csv, 'w') as f:
        f.write('id, label\n')
        for i, p in enumerate(pred):
            f.write('{i},{p}\n')
    print('Save prediction to {out_csv}.')


if __name__ == '__main__':
    # load model
    model = AE()
    model.load_state_dict(torch.load('./checkpoints/last_checkpoint.pth'))
    model.eval()

    # 準備 data
    trainX = np.load('./trainX_new.npy')

    # 預測答案
    latents = inference(X=trainX, model=model)
    pred, X_embedded = predict(latents)

    # 將預測結果存檔，上傳 kaggle
    save_prediction(pred, 'prediction.csv')

    # 由於是 unsupervised 的二分類問題，我們只在乎有沒有成功將圖片分成兩群
    # 如果上面的檔案上傳 kaggle 後正確率不足 0.5，只要將 label 反過來就行了
    save_prediction(invert(pred), 'prediction_invert.csv')

