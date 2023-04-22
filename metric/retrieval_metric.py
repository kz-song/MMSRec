import torch
from torch import nn


class RetrievalMetric(object):
    def __call__(self, queries, query_mask, values, value_mask):
        queries = nn.functional.normalize(queries, dim=-1)
        values = nn.functional.normalize(values, dim=-1)
        query_mask = query_mask.reshape(-1, 1)
        value_mask = value_mask.reshape(-1, 1)

        sim_matrix = torch.matmul(queries, values.T)
        sim_mask = query_mask * value_mask.T
        sim_matrix = sim_matrix.masked_fill(~sim_mask, -1.0e+4)
        calc_mask = (query_mask | value_mask).reshape(-1)
        sort_index = self.calc_sort_index(sim_matrix, calc_mask)
        R1, R5, R10 = self.calc_matrics(sort_index, calc_mask)
        return R1, R5, R10

    def calc_sort_index(self, sim_matrix, mask):
        sim_matrix_sort = torch.argsort(sim_matrix, dim=-1, descending=True)
        sim_diag = torch.arange(0, sim_matrix.shape[0], device=sim_matrix.device).reshape(-1, 1)
        diff = sim_matrix_sort - sim_diag
        sort_index = torch.argmax((diff == 0).type_as(diff), dim=1)
        sort_index = sort_index.masked_fill(~mask, sort_index.shape[0] + 1)
        return sort_index

    def calc_matrics(self, sort_index, mask):
        batch = mask.sum()
        R1 = torch.sum(sort_index == 0) / batch
        R5 = torch.sum(sort_index < 5) / batch
        R10 = torch.sum(sort_index < 10) / batch
        return R1, R5, R10




