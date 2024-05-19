import torch
from torch import nn
from transformer import TransformerEncoder, FeedForward


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, context_size, embed_dim, n_layers, n_heads, hidden_dim, n_classes):
        super(TransformerClassifier, self).__init__()
        self.n_classes = n_classes
        self.encoder = TransformerEncoder(vocab_size, context_size, embed_dim, n_layers, n_heads)
        self.feedforward = FeedForward(embed_dim, hidden_dim, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.mean(x, dim=1)
        x = self.feedforward(x)
        return x

