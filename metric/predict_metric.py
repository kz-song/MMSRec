import torch
from torch import nn


class PredictMetric(object):
    def __call__(self, predicts, labels):
        predicts_sort = torch.argsort(predicts, dim=-1, descending=True)
        diff = predicts_sort - labels.reshape(-1, 1)
        sort_index = torch.argmax((diff == 0).type_as(diff), dim=-1)
        return sort_index, predicts.shape[0]


class SequentialPredictMetric(object):
    def __call__(self, query, value, labels):
        N, D = query.shape

        query = nn.functional.normalize(query, dim=-1)
        value = nn.functional.normalize(value, dim=-1)
        labels = labels.reshape(-1, 1)

        sim_matrix = torch.matmul(query, value.T)  # [N, V]
        sort_index = self.calc_sort_index(sim_matrix, labels)
        return sort_index, N

    def calc_sort_index(self, sim_matrix, sim_id):
        sim_matrix_sort = torch.argsort(sim_matrix, dim=-1, descending=True)
        diff = sim_matrix_sort - sim_id
        sort_index = torch.argmax((diff == 0).type_as(diff), dim=1)
        return sort_index
