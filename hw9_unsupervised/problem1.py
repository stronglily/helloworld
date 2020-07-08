import numpy as np
import torch
from hw9.clustering import inference, predict, cal_acc, plot_scatter
from hw9.unsupervised import AE

# 將 val data 的降維結果 (embedding) 與他們對應的 label 畫出來。

if __name__ == '__main__':
    valX = np.load('./valX.npy')
    valY = np.load('./valY.npy')

    # ==============================================
    #  我們示範 basline model 的作圖，
    #  report 請同學另外還要再畫一張 improved model 的圖。
    # ==============================================
    model = AE()
    model.load_state_dict(torch.load('./checkpoints/last_checkpoint.pth'))
    model.eval()
    latents = inference(valX, model)
    pred_from_latent, emb_from_latent = predict(latents)
    acc_latent = cal_acc(valY, pred_from_latent)
    print('The clustering accuracy is:', acc_latent)
    print('The clustering result:')
    plot_scatter(emb_from_latent, valY, savefig='p1_baseline.png')
