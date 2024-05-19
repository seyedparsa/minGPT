import torch
from torch import nn
from transformer import TransformerEncoder, FeedForward
from torch.nn import functional as F


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


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_size, embed_dim, n_layers, n_heads):
        super(TransformerLM, self).__init__()
        self.encoder = TransformerEncoder(vocab_size, context_size, embed_dim, n_layers, n_heads, causal=True)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.n_classes = vocab_size

    def forward(self, x, y):
        x = self.encoder(x)
        logits = self.fc(x)
        loss = F.cross_entropy(logits.view(-1, self.n_classes), y.view(-1))
        return loss, logits
