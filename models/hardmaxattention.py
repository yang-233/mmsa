import torch
from math import sqrt
from torch import nn


def _hardmax(t):
    idx = t.argmax(dim=-1).view(-1)
    _t = 1
    for i in t.shape[:-1]:
        _t *= i
    _range = torch.arange(_t, device=t.device)
    step = t.shape[-1]
    _range *= step
    idx += _range
    res = torch.zeros_like(t).view(-1)
    res[idx] = 1.
    return res.view(t.shape)

class HardMaxAttention(nn.Module):
    def __init__(self, hidden_dim:int) -> None:
        super(HardMaxAttention, self).__init__()

        self.hidden_dim = hidden_dim

        self.fc_query = nn.Linear(hidden_dim, hidden_dim)
        self.fc_key = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, hidden_dim)

        self.scale = torch.tensor(sqrt(hidden_dim), dtype=torch.float32)

    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor,
                    mask:torch.BoolTensor=None):
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

        K = K.permute(0, 2, 1)
        #Q = [batch size, query len, hidden dim]
        #K = [batch size, hidden dim, key len]
        #V = [batch size, key len, hidden dim]

        energy = torch.matmul(Q, K) / self.scale
        #energy = [batch size, query len, key len]

        if mask is not None:
            # mask (batch_size, query_len, key_len)
            mask = mask.unsqueeze(1).expand(batch_size, query.shape[1], key.shape[1])
            energy = energy.masked_fill(mask, float("-inf"))

        score = _hardmax(energy) #[batch size, query len, key len]

        res = torch.matmul(score, V)
        #res = [batch size, query len, hidden dim]
        
        res = self.fc_output(res)
        #res = [batch size, query len, hiddden dim]
        return res, score
