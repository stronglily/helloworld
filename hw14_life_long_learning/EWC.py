# Elastic Weight Consolidation
from torch import nn
import torch.nn.functional as F

class EWC(object):
    """
      @article{kirkpatrick2017overcoming,
          title={Overcoming catastrophic forgetting in neural networks},
          author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
          journal={Proceedings of the national academy of sciences},
          year={2017},
          url={https://arxiv.org/abs/1612.00796}
      }
    """

    def __init__(self, model: nn.Module, dataloaders: list, device):

        self.model = model
        self.dataloaders = dataloaders
        self.device = device

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # 抓出模型的所有參數
        self._means = {}  # 初始化 平均參數
        self._precision_matrices = self._calculate_importance()  # 產生 EWC 的 Fisher (F) 矩陣

        for n, p in self.params.items():
            self._means[n] = p.clone().detach()  # 算出每個參數的平均 （用之前任務的資料去算平均）

    def _calculate_importance(self):
        precision_matrices = {}
        for n, p in self.params.items():  # 初始化 Fisher (F) 的矩陣（都補零）
            precision_matrices[n] = p.clone().detach().fill_(0)

        self.model.eval()
        dataloader_num = len(self.dataloaders)
        number_data = sum([len(loader) for loader in self.dataloaders])
        for dataloader in self.dataloaders:
            for data in dataloader:
                self.model.zero_grad()
                input = data[0].to(self.device)
                output = self.model(input).view(1, -1)
                label = output.max(1)[1].view(-1)

                ############################################################################
                #####                      產生 EWC 的 Fisher(F) 矩陣                    #####
                ############################################################################
                loss = F.nll_loss(F.log_softmax(output, dim=1), label)
                loss.backward()

                for n, p in self.model.named_parameters():
                    precision_matrices[n].data += p.grad.data ** 2 / number_data

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
