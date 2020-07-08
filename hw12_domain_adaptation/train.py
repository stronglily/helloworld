"""
如何實作DaNN?
理論上，在原始paper中是加上Gradient Reversal Layer，並將Feature Extractor / Label Predictor / Domain Classifier 一起train，
但其實我們也可以交換的train Domain Classfier & Feature Extractor(就像在train GAN的Generator & Discriminator一樣)，這也是可行的。
在code實現中，我們採取後者的方式，畢竟大家上個作業就是GAN，應該會比較熟悉:)。
小提醒:
原文中的lambda(控制Domain Adversarial Loss的係數)是有Adaptive的版本，如果有興趣可以參考原文。以下為了方便固定設置0.1。
因為我們完全沒有target的label，所以結果如何，只好丟kaggle看看囉:)?
"""
import torch
from hw12_domain_adaptation.model import FeatureExtractor, LabelPredictor, DomainClassifier
from torch import nn, optim
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def train_epoch(source_dataloader, target_dataloader, lamb):
    '''
      Args:
        source_dataloader: source data的dataloader
        target_dataloader: target data的dataloader
        lamb: 調控adversarial的loss係數。
    '''

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    # total_hit: 計算目前對了幾筆 total_num: 目前經過了幾筆
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0

    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = target_data.cuda()

        # 我們把source data和target data混在一起，否則batch_norm可能會算錯 (兩邊的data的mean/var不太一樣)
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        # 設定source data的label為1
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : 訓練Domain Classifier
        feature = feature_extractor(mixed_data)
        # 因為我們在Step 1不需要訓練Feature Extractor，所以把feature detach避免loss backprop上去。
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss += loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2 : 訓練Feature Extractor和Domain Classifier
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss為原本的class CE - lamb * domain BCE，相減的原因同GAN中的Discriminator中的G loss。
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss += loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i, end='\r')

    return running_D_loss / (i + 1), running_F_loss / (i + 1), total_hit / total_num


if __name__ == '__main__':
    """------数据预处理------"""
    source_transform = transforms.Compose([
        # 轉灰階: Canny 不吃 RGB。
        transforms.Grayscale(),
        # cv2 不吃 skimage.Image，因此轉成np.array後再做cv2.Canny
        transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
        # 重新將np.array 轉回 skimage.Image
        transforms.ToPILImage(),
        # 水平翻轉 (Augmentation)
        transforms.RandomHorizontalFlip(),
        # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
        transforms.RandomRotation(15, fill=(0,)),
        # 最後轉成Tensor供model使用。
        transforms.ToTensor(),
    ])
    target_transform = transforms.Compose([
        # 轉灰階: 將輸入3維壓成1維。
        transforms.Grayscale(),
        # 縮放: 因為source data是32x32，我們將target data的28x28放大成32x32。
        transforms.Resize((32, 32)),
        # 水平翻轉 (Augmentation)
        transforms.RandomHorizontalFlip(),
        # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
        transforms.RandomRotation(15, fill=(0,)),
        # 最後轉成Tensor供model使用。
        transforms.ToTensor(),
    ])

    source_dataset = ImageFolder('./real_or_drawing/train_data', transform=source_transform)
    target_dataset = ImageFolder('./real_or_drawing/test_data', transform=target_transform)

    source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
    target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)

    # Pre-processing
    # 這裡我們選用Adam來當Optimizer。
    feature_extractor = FeatureExtractor().cuda()
    label_predictor = LabelPredictor().cuda()
    domain_classifier = DomainClassifier().cuda()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer_F = optim.Adam(feature_extractor.parameters())
    optimizer_C = optim.Adam(label_predictor.parameters())
    optimizer_D = optim.Adam(domain_classifier.parameters())

    # 訓練200 epochs
    for epoch in range(200):
        train_D_loss, train_F_loss, train_acc = train_epoch(source_dataloader, target_dataloader, lamb=0.1)

        torch.save(feature_extractor.state_dict(), 'extractor_model.bin')
        torch.save(label_predictor.state_dict(), 'predictor_model.bin')

        print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss,
                                                                                               train_F_loss, train_acc))
