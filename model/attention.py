import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()

        torch.manual_seed(0)

        self.attention_dim = attention_dim

        # Order matters
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)

        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)

        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:

        # Q, K, V projections
        K = self.key(embedded)

        Q = self.query(embedded)

        V = self.value(embedded)

        # Attention scores
        scores = (Q @ K.transpose(1, 2)) / (self.attention_dim ** 0.5)

        # Causal mask
        mask = torch.tril(torch.ones(scores.shape[-2:]))

        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        scores = torch.softmax(scores, dim=2)

        # Weighted sum
        output = scores @ V

        # Round to 4 decimals
        return torch.round(output * 10000) / 10000

