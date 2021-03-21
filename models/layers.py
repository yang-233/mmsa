import enum
import numpy as np
import torch
import math
from typing import *
from torch import nn
from torch._C import dtype
from torch.nn.modules import dropout
from torch.nn.modules import padding
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def hn_reshape(h_n:torch.Tensor) -> torch.Tensor:
    # input_size (num_layers, batch_size, hidden_size) output_size (batch_size, -1)
    b = h_n.shape[1]
    return h_n.permute(1, 0, 2).contiguous().view(b, -1)

class OutputLayer(nn.Module):
    def __init__(self, task:str, input_size:int, output_size:int, dropout):
        super(OutputLayer, self).__init__()
        self.task = task
        self.output_size = output_size
        if task == "CLS":
            self.fc = nn.Linear(input_size, output_size)
            self.dropout = nn.Dropout(dropout, inplace=True)
        elif task == "REG":
            self.fc = nn.Linear(input_size, 1)
            self.dropout = nn.Dropout(0., inplace=True)
        else:
            raise RuntimeError(f"task mush be CLS or REG but got {task}")
    
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        
        return self.dropout(self.fc(X))

    def predict(self, output:torch.Tensor) -> List[int]:
        if self.task == "CLS":
            pred = output.argmax(dim=1).cpu().numpy().ravel().tolist()
        else:
            pred = output.clone().detach().requires_grad_(False) 
            pred = (pred.cpu().numpy().ravel() * self.output_size).tolist()
        return pred

class BaseModel(nn.Module):
    def predict(self, output):
        return self.output_layer.predict(output)
        
class SimpleImageEncoder(nn.Module):
    def __init__(self, input_size:int, output_size:int, img_num:int,  dropout:float=0.):
        super(SimpleImageEncoder, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.relu = nn.ReLU(True)
        self.output_size = output_size
        self.img_num = img_num
        self.pooling = nn.MaxPool2d((img_num, 1))
    
    def forward(self, X):
        h = self.linear(X)
        h = self.dropout(h)
        h = self.relu(h)
        h = h.view((-1, 1, self.img_num, self.output_size))
        return self.pooling(h).view((-1, self.output_size))

class DynamicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0.,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type = 'LSTM', bias_init=1.0):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).
        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        :param bias_init: the init value of the bias in {LSTM, GRU, RNN}
        """

        super(DynamicRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type
        
        if self.rnn_type == 'LSTM': 
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)  
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        else:
            raise RuntimeError('rnn_type must be LSTM or GRU or RNN')
        
        # if self.rnn_type  == "LSTM" or self.rnn_type == "GRU": 
        #     for name, param in self.RNN.named_parameters():
        #         if "weight" in name:
        #             nn.init.orthogonal_(param)
        #         if "bias" in name:
        #             nn.init.constant_(param, bias_init)
    
    # def init_hn(self, batch_size): # batch, , hidden
    #     shape = (batch_size, self.num_layers * (1 + self.bidirectional), self.hidden_size)
    #     if self.rnn_type == 'LSTM':
    #         return torch.from_numpy(np.random.uniform(low=-0.1, high=-0.1, size=shape)),\
    #             torch.from_numpy(np.random.uniform(low=-0.1, high=-0.1, size=shape))
    #     else:
    #         return torch.from_numpy(np.random.uniform(low=-0.1, high=-0.1, size=shape))

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort
        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        
        # process using the selected RNN
        if self.rnn_type == 'LSTM': 
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else: 
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type =='LSTM':
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)
            return out, (ht, ct)


class SentencePaddingLayer(nn.Module): # padding wordencoder的输出
    def forward(self, h_n:torch.Tensor, lens:torch.Tensor, sents_num:torch.Tensor, padding_val:float=0.) -> torch.Tensor:
        d = h_n.shape[1]
        n = len(sents_num)
        max_sents = sents_num.max()
        res = torch.full((n, max_sents, d), padding_val, dtype=h_n.dtype, device=h_n.device)
        idx = 0
        for i, lens in enumerate(sents_num):
            res[i,:lens] = h_n[idx:idx+lens]
            idx += lens
        return res
class PaddingLayer(nn.Module):
    def forward(self, batch:torch.Tensor, lens:Iterable, padding_val:float=0.):
        n = len(lens)
        max_lens = max(lens)
        d = batch.shape[-1]
        padded = torch.full((n, max_lens, d), padding_val, dtype=batch.dtype, device=batch.device)
        idx = 0
        for i, l in enumerate(lens):
            padded[i,:l] = batch[idx:idx+l]
            idx += l
        return padded

class MaskLayer(nn.Module): # 
    def forward(self, batch:torch.Tensor, lens:torch.LongTensor, mask_val:float=float("-inf")) -> torch.Tensor:
        res = torch.full_like(batch, mask_val) #　(batch_size, seq, hidden)
        for i in range(len(batch)):
            res[i, :lens[i]] = batch[i, :lens[i]]
        return res

class AttentionDynamicRNN(nn.Module):
    def __init__(self, input_size:int, rnn_hidden_size:int, rnn_num_layers:int, uniform_bound:float=0.1,
                        bias=True, batch_first=True, dropout=0.,
                        bidirectional=False, rnn_type = 'LSTM', bias_init=1.0):
        super(AttentionDynamicRNN, self).__init__()
        self.rnn = DynamicRNN(input_size, rnn_hidden_size, rnn_num_layers, bias=bias, batch_first=batch_first, dropout=dropout,
                                    bidirectional=bidirectional, rnn_type=rnn_type, bias_init=bias_init)
        self.rnn_output_size = (1 + bidirectional) * rnn_num_layers * rnn_hidden_size

        self.proj_fc = nn.Linear(self.rnn_output_size, self.rnn_output_size)
        self.tanh = nn.Tanh()
        self.u = nn.Parameter(data=torch.from_numpy(np.random.uniform(-uniform_bound, uniform_bound,
                              size=(self.rnn_output_size))).type(torch.float32))
        self.mask_layer = MaskLayer()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, X, lens):
        hi, _ = self.rnn(X, lens)
        shape = hi.shape[:2]
        hi = hi.view(-1, self.rnn_output_size) # 压平
        ui = self.proj_fc(hi) # 过全连接
        ui = self.tanh(ui) 
        ui = ui @ self.u # 再做一次线性变换
        ui = ui.view(shape) # 变回原形
        masked_ui = self.mask_layer(ui, lens) # mask
        score = self.softmax(masked_ui) # 计算attention score
        score = score.view(*score.shape, 1) # 增加一个1维度
        
        hi = hi.view(*shape, -1)
        output = (hi * score).sum(dim=1) # 注意力机制
        return output

class PositionEmbedding(nn.Module):
    def __init__(self, size:int, n_dim:int, learn:bool=False) -> None:
        super(PositionEmbedding, self).__init__()
        if not learn:
            pe = torch.zeros(size, n_dim)
            position = torch.arange(0, size).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, n_dim, 2) *
                             -(math.log(10000.0) / n_dim))
            temp = (position * div_term)
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)[:,:n_dim//2]
            self.embedding = nn.Embedding.from_pretrained(pe)
            self.embedding.requires_grad_(False)
        else:
            self.embedding = nn.Embedding(size, n_dim)
            self.embedding.requires_grad_(True)

    def forward(self, X):
        return self.embedding(X)

class MultiheadAttentionEncoder(nn.Module):
    def __init__(self, input_size:int, nhead:int, uniform_bound:float) -> None:
        super(MultiheadAttentionEncoder, self).__init__()
        self.input_size = input_size
        q = np.random.uniform(-uniform_bound, uniform_bound, size=(1, 1, input_size))
        q = torch.from_numpy(q).type(torch.float32)
        self.q = nn.Parameter(data=q, requires_grad=True)
        self.attn = nn.MultiheadAttention(input_size, nhead)

    def forward(self, x, mask): # (batch_size, seq, input_size)
        batch_size = len(x)
        x = x.permute(1, 0, 2).contiguous() # (seq, batch_size, input_size)
        query = self.q.expand(1, batch_size, self.input_size)
        v, s = self.attn(query, x, x, key_padding_mask=mask, need_weights=False) # don't return score
        v = v.permute(1, 0, 2).contiguous().squeeze() # (batch_size, input_size)
        return v
    
class SimpleRCNNEncoder(nn.Module):
    def __init__(self, input_size:int, bbox_head:int, output_size:int, imgs_head:int, uniform_bound:float) -> None:
        super(SimpleRCNNEncoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bbox_encoder = MultiheadAttentionEncoder(input_size, bbox_head, uniform_bound)

        self.fc = nn.Linear(input_size, output_size)
        self.bbox_tanh = nn.Tanh()
        self.imgs_encoder = MultiheadAttentionEncoder(output_size, imgs_head, uniform_bound)

    
    def forward(self, padded_x, mask, imgs_num): # padded_x (batch_size, imgs_num, bboxes_num, input_size)
        h = self.bbox_encoder(padded_x, mask)
        h = self.bbox_tanh(self.fc(h))
        
        ## padd_ing
        max_imgs = max(imgs_num)
        batch_size = len(imgs_num)
        padded_h = torch.zeros((batch_size, max_imgs, self.output_size), 
                               dtype=h.dtype, device=h.device)
        mask = torch.ones((batch_size, max_imgs), dtype=torch.bool, device=h.device)
        idx = 0
        for i, l in enumerate(imgs_num):
            padded_h[i,:l] = h[idx:idx+l]
            mask[i,:l] = False
            idx += l
        return self.imgs_encoder(padded_h, mask)

