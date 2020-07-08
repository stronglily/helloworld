# Memory Aware Synapses
import torch
from torch import nn


class MAS(object):
    """
    @article{aljundi2017memory,
      title={Memory Aware Synapses: Learning what (not) to forget},
      author={Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny, Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
      booktitle={ECCV},
      year={2018},
      url={https://eccv2018.org/openaccess/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf}
    }
    """

    def __init__(self, model: nn.Module, dataloaders: list, device):
        self.model = model
        self.dataloaders = dataloaders
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # 抓出模型的所有參數
        self._means = {}  # 初始化 平均參數
        self.device = device
        self._precision_matrices = self.calculate_importance()  # 產生 MAS 的 Omega(Ω) 矩陣

        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def calculate_importance(self):
        print('Computing MAS')

        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0)  # 初始化 Omega(Ω) 矩陣（都補零）

        self.model.eval()
        dataloader_num = len(self.dataloaders)
        num_data = sum([len(loader) for loader in self.dataloaders])
        for dataloader in self.dataloaders:
            for data in dataloader:
                self.model.zero_grad()
                output = self.model(data[0].to(self.device))

                #######################################################################################
                #####  產生 MAS 的 Omega(Ω) 矩陣 ( 對 output 向量 算他的 l2 norm 的平方) 再取 gradient  #####
                #######################################################################################
                output.pow_(2)
                loss = torch.sum(output, dim=1)
                loss = loss.mean()
                loss.backward()

                for n, p in self.model.named_parameters():
                    precision_matrices[n].data += p.grad.abs() / num_data  ## difference with EWC

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
