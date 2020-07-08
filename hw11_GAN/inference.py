import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable
from hw11.hw11GAN import Generator, z_dim, workspace_dir

if __name__ == '__main__':
    '''inference'''
    # load pretrained model
    G = Generator(z_dim)
    G.load_state_dict(torch.load(os.path.join(workspace_dir, 'dcgan_g.pth')))
    G.eval()
    G.cuda()
    # generate images and save the result
    n_output = 20
    z_sample = Variable(torch.randn(n_output, z_dim))
    imgs_sample = (G(z_sample).data + 1) / 2.0
    save_dir = os.path.join(workspace_dir, 'logs')
    filename = os.path.join(save_dir, 'result.jpg')
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)
    # show image
    grid_img = torchvision.utils.make_grid(imgs_sample.cpu(), nrow=10)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
