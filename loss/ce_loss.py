import torch
from torch import nn


class CELoss(object):
    def __init__(self, args, **kwargs):
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    def __call__(self, predicts, labels):
        L, V = predicts.shape

        predicts = predicts.reshape(-1, V)
        labels = labels.reshape(-1)

        loss = self.loss(predicts, labels)
        return loss









