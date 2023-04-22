import torch
from torch import nn


class NormSoftmaxLoss(nn.Module):
    def __init__(self,
                 args,
                 **kwargs):
        super().__init__()
        self.args = args
        self.temperature = args.contrastive_temperature

    def forward(self, input1, mask1, input2, mask2):
        input1 = nn.functional.normalize(input1, dim=-1)
        input2 = nn.functional.normalize(input2, dim=-1)
        mask1 = mask1.reshape(-1, 1)
        mask2 = mask2.reshape(-1, 1)
        assert input1.shape == input2.shape
        assert mask1.shape == mask2.shape

        sim_matrix = torch.matmul(input1, input2.T)
        sim_mask = mask1 * mask2.T
        sim_matrix = sim_matrix.masked_fill(~sim_mask, -1.0e+4)

        i_logsoftmax = nn.functional.log_softmax(sim_matrix / self.temperature, dim=1)
        j_logsoftmax = nn.functional.log_softmax(sim_matrix.T / self.temperature, dim=1)

        i_diag = torch.masked_select(torch.diag(i_logsoftmax), torch.diag(sim_mask))
        loss_i = i_diag.mean()

        j_diag = torch.masked_select(torch.diag(j_logsoftmax), torch.diag(sim_mask.T))
        loss_j = j_diag.mean()

        return - loss_i - loss_j



