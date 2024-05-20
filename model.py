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
        # mask = (x == cls_i) # Uncomment for part 3
        x = self.encoder(x)
        # x = x[mask] # Uncomment for part 3
        x = torch.mean(x, dim=1) # Comment for part 3
        x = self.feedforward(x)
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_size, embed_dim, n_layers, n_heads):
        super(TransformerLM, self).__init__()
        self.encoder = TransformerEncoder(vocab_size, context_size, embed_dim, n_layers, n_heads, causal=True)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.n_classes = vocab_size

    def forward(self, x, y=None):
        x = self.encoder(x)
        logits = self.fc(x)
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, self.n_classes), y.view(-1))
        else:
            loss = None
        return loss, logits

    def generate(self, x):
        x = self.encoder(x)
        logits = self.fc(x)
        return torch.argmax(logits, dim=-1)
