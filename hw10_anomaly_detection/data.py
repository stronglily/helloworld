import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import f1_score, pairwise_distances, roc_auc_score
from scipy.cluster.vq import vq, kmeans
from sklearn.decomposition import PCA

"""
這份作業要執行的task是semi-supervised anomaly detection，
也就是說training set是乾淨的，testing的時候才會混進outlier data(anomaly)。 
我們以某個簡單的image dataset（image加上他們的label（分類））作為示範，
training data為原先training set中的某幾類，而testing data則是原先testing set的所有data，
要偵測的anomaly為training data中未出現的類別。label的部分，1為outlier data，而0為inlier data(相對於 outlier)。
正確率以AUC計算。 方法則列舉3種： KNN, PCA, Autoencoder
"""

if __name__ == '__main__':
    train = np.load('./train.npy', allow_pickle=True)
    test = np.load('./test.npy', allow_pickle=True)

    task = 'pca'

    if task == 'knn':
        x = train.reshape(len(train), -1)
        y = test.reshape(len(test), -1)
        scores = list()
        for n in range(1, 10):
            kmeans_x = MiniBatchKMeans(n_clusters=n, batch_size=100).fit(x)
            y_cluster = kmeans_x.predict(y)
            y_dist = np.sum(np.square(kmeans_x.cluster_centers_[y_cluster] - y), axis=1)

            y_pred = y_dist
            # score = f1_score(y_label, y_pred, average='micro')
            # score = roc_auc_score(y_label, y_pred, average='micro')
            # scores.append(score)
        # print(np.max(scores), np.argmax(scores))
        # print(scores)
        # print('auc score: {}'.format(np.max(scores)))

    if task == 'pca':
        x = train.reshape(len(train), -1)
        y = test.reshape(len(test), -1)
        pca = PCA(n_components=2).fit(x)

        y_projected = pca.transform(y)
        y_reconstructed = pca.inverse_transform(y_projected)
        dist = np.sqrt(np.sum(np.square(y_reconstructed - y).reshape(len(y), -1), axis=1))

        y_pred = dist
        # score = roc_auc_score(y_label, y_pred, average='micro')
        # score = f1_score(y_label, y_pred, average='micro')
        # print('auc score: {}'.format(score))
