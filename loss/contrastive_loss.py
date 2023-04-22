import torch
from torch import nn


class UnsupervisedContrastiveLoss(object):
    def __init__(self,
                 args,
                 **kwargs):
        self.args = args
        self.temperature = args.contrastive_temperature

    def __call__(self, src, tgt):
        N, D = src.shape

        src = nn.functional.normalize(src, dim=-1)
        tgt = nn.functional.normalize(tgt, dim=-1)
        assert src.shape == tgt.shape

        sim_matrix = torch.matmul(src, tgt.T)

        i_logsoftmax = nn.functional.log_softmax(sim_matrix / self.temperature, dim=1)
        j_logsoftmax = nn.functional.log_softmax(sim_matrix.T / self.temperature, dim=1)

        i_diag = torch.diag(i_logsoftmax)
        loss_i = i_diag.mean()

        j_diag = torch.diag(j_logsoftmax)
        loss_j = j_diag.mean()

        return - loss_i - loss_j


class SupervisedContrastiveLoss(object):
    def __init__(self,
                 args,
                 **kwargs):
        self.args = args
        self.temperature = args.contrastive_temperature

    def __call__(self, src, src_label, tgt, tgt_label):
        N, D = src.shape

        src = nn.functional.normalize(src, dim=-1)
        tgt = nn.functional.normalize(tgt, dim=-1)
        assert src.shape == tgt.shape
        sim_matrix = torch.matmul(src, tgt.T)

        src_label = src_label.reshape(-1, 1).expand(N, N)
        tgt_label = tgt_label.reshape(1, -1).expand(N, N)
        sim_mask = (src_label == tgt_label) & ~torch.eye(N, dtype=torch.bool, device=self.args.device)
        sim_matrix = sim_matrix.masked_fill(sim_mask, -1.0e+4)

        i_logsoftmax = nn.functional.log_softmax(sim_matrix / self.temperature, dim=1)
        j_logsoftmax = nn.functional.log_softmax(sim_matrix.T / self.temperature, dim=1)

        i_diag = torch.diag(i_logsoftmax)
        loss_i = i_diag.mean()

        j_diag = torch.diag(j_logsoftmax)
        loss_j = j_diag.mean()

        return - loss_i - loss_j


class TargetContrastiveLoss(object):
    def __init__(self,
                 args,
                 **kwargs):
        self.args = args
        self.temperature = args.contrastive_temperature

    def __call__(self, src, tgt, label):
        N, D = src.shape

        src = nn.functional.normalize(src, dim=-1)
        tgt = nn.functional.normalize(tgt, dim=-1)
        assert src.shape == tgt.shape
        sim_matrix = torch.matmul(src, tgt.T)

        label1 = label.reshape(-1, 1).expand(N, N)
        label2 = label.reshape(1, -1).expand(N, N)
        sim_mask = (label1 == label2) & ~torch.eye(N, dtype=torch.bool, device=self.args.device)
        sim_matrix = sim_matrix.masked_fill(sim_mask, -1.0e+4)

        i_logsoftmax = nn.functional.log_softmax(sim_matrix / self.temperature, dim=1)
        j_logsoftmax = nn.functional.log_softmax(sim_matrix.T / self.temperature, dim=1)

        i_diag = torch.diag(i_logsoftmax)
        loss_i = i_diag.mean()

        j_diag = torch.diag(j_logsoftmax)
        loss_j = j_diag.mean()

        return - loss_i - loss_j






