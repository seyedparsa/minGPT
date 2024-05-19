import torch
from torch import nn


class TransformerClassifier(nn.Module):
    def __init__(self, n_classes):
        super(TransformerClassifier, self).__init__()
        self.n_classes = n_classes

    def forward(self, x):
        return torch.randn(x.size(0), self.n_classes)
