# add all  your Encoder and Decoder code here
import math

import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    """
    Feed-forward neural network with one hidden layer and relu activation function
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def attention(q, k, v):
    """
    Compute attention weights between query and key vectors and return value vectors weighted by the attention weights
    """
    attention_matrix = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1)), dim=-1)
    return torch.matmul(attention_matrix, v)


class Head(nn.Module):
    """
    Self-attention head
    """
    def __init__(self, embed_dim, head_dim):
        super(Head, self).__init__()
        self.Q = nn.Linear(embed_dim, head_dim, bias=False)
        self.K = nn.Linear(embed_dim, head_dim, bias=False)
        self.V = nn.Linear(embed_dim, head_dim, bias=False)

    def forward(self, x):
        return attention(self.Q(x), self.K(x), self.V(x))


class MultiHeadAttention(nn.Module):
    """
    Concatenation of multi self-attention heads with a linear layer on top of it
    """
    def __init__(self, embed_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        head_dim = embed_dim // n_heads
        self.heads = nn.ModuleList([Head(embed_dim, head_dim) for _ in range(n_heads)])
        self.linear = nn.Linear(n_heads * head_dim, embed_dim, bias=False)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.linear(x)
        return x


class Block(nn.Module):
    """
    Block of multi-head attention and feed-forward neural network with residual connections and layer normalizations
    """
    def __init__(self, embed_dim, n_head):
        super(Block, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, n_head)
        self.feed_forward = FeedForward(embed_dim, embed_dim*4, embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with multiple blocks
    """
    def __init__(self, vocab_size, context_size, embed_dim, n_layers, n_heads):
        super(TransformerEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_size, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, seq):
        device = seq.device
        batch_size, seq_len = seq.shape
        token_embed = self.token_embedding(seq)
        pos_embed = self.position_embedding(torch.arange(seq_len, dtype=torch.long, device=device))
        x = token_embed + pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        return x
