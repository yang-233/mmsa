import torch
import numpy as np
from math import sqrt
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim:int, n_heads:int, dropout:float=0.) -> None:
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be div by n_heads"
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.fc_query = nn.Linear(hidden_dim, hidden_dim)
        self.fc_key = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, hidden_dim)

        self.fc_output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout, True)
        self.scale = torch.tensor(sqrt(n_heads), dtype=torch.float32)

    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor,
                    mask:torch.BoolTensor):
        """[summary]

        Parameters
        ----------
        query : torch.Tensor
            [shape : (batch_size, query_len, hidden_dim)]
        key : torch.Tensor
            [shape : (batch_size, key_len, hidden_dim)]
        value : torch.Tensor
            [shape : (batch_size, key_len, hidden_dim)]
        mask : torch.BoolTensor
            [shape : (batch_size, query_len, key_len)]
        """
        batch_size = query.shape[0]
        Q, K, V = self.fc_query(query), self.fc_key(key), self.fc_value(value)

        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, key len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        #energy = [batch size, n heads, query len, key len]
        if mask is not None:
            # mask (batch_size, query_len, key_len)
            mask = mask.unsqueeze(1).expand(batch_size, self.n_heads, -1, -1)
            energy = energy.masked_fill(mask, float("-inf"))

        score = torch.softmax(energy, dim=-1)
        #score = [batch size, n heads, query len, key len]

        res = torch.matmul(self.dropout(score), V)
        #res = [batch size, n heads, query len, head dim]
        
        res = res.permute(0, 2, 1, 3).contiguous()
        #res = [batch size, query len, n heads, head dim]

        res = res.view(batch_size, -1, self.hidden_dim)
        #res = [batch size, query len, hiddden dim]

        res = self.fc_output(res)
        #res = [batch size, query len, hiddden dim]
        return res, score.mean(dim=1) # score [batch_size, query len, key len]

class MultiHeadAttentionScore(nn.Module):
    def __init__(self, hidden_dim:int, n_heads:int, uniform_bound:float) -> None:
        super(MultiHeadAttentionScore, self).__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be div by n_heads"
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        q = np.random.uniform(-uniform_bound, uniform_bound, size=(1, 1, self.hidden_dim))
        q = torch.from_numpy(q).type(torch.float32)

        self.query = nn.Parameter(data=q, requires_grad=True)

        self.fc_query = nn.Linear(hidden_dim, hidden_dim)
        self.fc_key = nn.Linear(hidden_dim, hidden_dim)

        self.scale = torch.tensor(sqrt(n_heads), dtype=torch.float32)

    def forward(self, key:torch.Tensor,
                    mask:torch.BoolTensor=None):
        """[summary]
        Parameters
        ----------
        query : torch.Tensor
            [shape : (batch_size, query_len, hidden_dim)]
        key : torch.Tensor
            [shape : (batch_size, key_len, hidden_dim)]
        mask : torch.BoolTensor
            [shape : (batch_size, query_len, key_len)]
        """
        batch_size = key.shape[0]
        query = self.query.expand((batch_size, 1, self.hidden_dim))

        Q, K= self.fc_query(query), self.fc_key(key)

        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        #Q = [batch size, n heads, query len, head dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim)
        #K = [batch size, key len, nheads, head dim]
        
        energy = torch.matmul(Q, K.permute(0, 2, 3, 1)) / self.scale

        #energy = [batch size, n heads, query len, key len]
        if mask is not None:
            # mask (batch_size, query_len, key_len)
            mask = mask.unsqueeze(1).expand(batch_size, self.n_heads, -1, -1)
            energy = energy.masked_fill(mask, float("-inf"))

        score = torch.softmax(energy, dim=-1)
        #score = [batch size, n heads, query len, key len]
        score = score.mean(dim=1)
        return score