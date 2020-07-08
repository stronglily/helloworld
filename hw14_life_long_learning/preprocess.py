# 資料預處理
# 轉換 MNIST ( 1∗28∗28 ) 到 ( 3∗32∗32 )
# 轉換 USPS ( 1∗16∗16 ) 到 ( 3∗32∗32 )
# 正規化 圖片
import torch
from torchvision import transforms


class Convert2RGB(object):

    def __init__(self, num_channel):
        self.num_channel = num_channel

    def __call__(self, img):
        # If the channel of img is not equal to desired size,
        # then expand the channel of img to desired size.
        img_channel = img.size()[0]
        img = torch.cat([img] * (self.num_channel - img_channel + 1), 0)
        return img


class Pad(object):

    def __init__(self, size, fill=0, padding_mode='constant'):
        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        # If the H and W of img is not equal to desired size,
        # then pad the channel of img to desired size.
        img_size = img.size()[1]
        assert ((self.size - img_size) % 2 == 0)
        padding = (self.size - img_size) // 2
        padding = (padding, padding, padding, padding)
        return F.pad(img, padding, self.padding_mode, self.fill)


def get_transform():
    transform = transforms.Compose([transforms.ToTensor(),
                                    Pad(32),
                                    Convert2RGB(3),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    return transform
