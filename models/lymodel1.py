from numpy.core.fromnumeric import reshape
from numpy.lib.utils import source
import torch
from torch import nn
from models.layers import *
from utils.dataset import padding_text_1d
from utils.load_yelp import load_rcnn_features

default_config = {
    "task" : "CLS",
    "embedding_dim" : 100,
    "embedding" : None, 
    "freeze_embedding" : False,
    "text_hidden_size" : 100,
    "text_layers" : 1,
    "max_tokens" : 512,
    "uniform_bound" : 0.1,

    "img_input_size" : 2048,
    "img_encoder_layers" : 1,
    "attention_nhead" : 4,

    "fusion_hidden_size" : 200,
    "funsion_layers" : 1,
    "dropout" : 0.5,
    "output_size" : 5, 
    "bias_init" : 1.0
}

config = default_config

class Model(BaseModel):
    def __init__(self, config:dict) -> None:
        super(Model, self).__init__()
        self.word_embedding = config["embedding"]
        if not config["freeze_embedding"]:
            self.word_embedding.requires_grad_(True)
        else:
            self.word_embedding.requires_grad_(False)

        self.word_encoder = DynamicRNN(config["embedding_dim"], hidden_size=config["text_hidden_size"], 
                                num_layers=config["text_layers"], dropout=config["dropout"],
                                bias_init=config["bias_init"], batch_first=True, bidirectional=True, rnn_type="GRU")
        
        self.word_output_size = config["text_hidden_size"] * config["text_layers"] * 2
        
        self.img_fc = nn.Linear(config["img_input_size"], self.word_output_size)
        self.tanh = nn.Tanh()
        self.attn = nn.MultiheadAttention(self.word_output_size, config["attention_nhead"])

        self.fusion_encoder = DynamicRNN(self.word_output_size, hidden_size=config["fusion_hidden_size"], 
                                num_layers=config["funsion_layers"], dropout=config["dropout"],
                                bias_init=config["bias_init"], batch_first=True, bidirectional=True,
                                 rnn_type="GRU")

        self.output_layer = OutputLayer(config["task"], config["fusion_hidden_size"] * config["funsion_layers"] * 2,
                             config["output_size"], config["dropout"])

    def forward(self, X):
        padded_text, lens, padded_x, img_key_padding_mask = X
        emb = self.word_embedding(padded_text)
        output, (hn, _) = self.word_encoder(emb, lens) # output(batch_size, seq, d)

        padded_x = padded_x.permute(1, 0, 2) # (seq, batch_size, d)
        transform_x = self.tanh(self.img_fc(padded_x))  # (batch_size, seq, d)
        transform_x = transform_x.permute(1, 0, 2) # (seq, batch_size, d)

        ### !BUG
        # attnmask 出现nan要考虑mask是否合理 
        # attn_output, score = self.attn(output.permute(1, 0, 2), transform_x, transform_x, attn_mask=atten_mask)
        # attn_output = attn_output.permute(1, 0, 2) # (batch_size, seq, d)

        output = output.permute(1, 0, 2)
        attn_output, score = self.attn(output, transform_x, transform_x, key_padding_mask=img_key_padding_mask)
        attn_output = attn_output.permute(1, 0, 2) # (batch_size, seq, d)
        output, (hn, _) = self.fusion_encoder(attn_output, lens)
        hn = hn_reshape(hn) # (batch_size, d)
        return self.output_layer(hn)

def get_collate_fn(config:dict):
    def collate_fn(batch):
        text, x, bbox_num, y = [], [], [], []
        n = len(batch)
        for (_text, _ids), _y in batch:
            t = []
            for i in _text:
                t += i
            text.append(t)
            t2 = []
            for _id in _ids:
                feature = load_rcnn_features(_id)
                t2.append(feature["x"])
            x.append(np.vstack(t2))
            bbox_num.append(len(x[-1])) # bbox num
            y.append(_y)
        padded_text, lens = padding_text_1d(text, config["max_tokens"])
        max_lens = max(lens)
        max_bbox = max(bbox_num)
        padded_x = torch.zeros((n, max_bbox, config["img_input_size"]), dtype=torch.float32)
        img_key_padding_mask = torch.ones((n, max_bbox), dtype=torch.bool)

        for i, (l, s) in enumerate(zip(lens, bbox_num)):
            padded_x[i,:s] = torch.from_numpy(x[i])
            img_key_padding_mask[i,:s] = False

        bbox_num = torch.LongTensor(bbox_num)
        y = torch.tensor(y)
        return (padded_text, lens, padded_x, img_key_padding_mask), y
    return collate_fn

