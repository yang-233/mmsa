import torch
from torch import nn
from utils.train import to_cuda
from models.layers import *
from utils.dataset import padding_text_1d
from utils.load_yelp import *

default_config = {
    "use_cuda" : True,
    "task" : "CLS",
    "embedding_dim" : 100,
    "embedding" : None, 
    "freeze_embedding" : True,
    "filter_size" : (2, 3, 4, 5),
    "filter_num" : 50,

    "max_tokens" : 512,
    "bias_init" : 1.0,
    "use_img" : True,
    "img_input_size" : 4096,
    "img_output_size" : 100,
    "img_num" : 3,
    "output_size" : 5, 
    "dropout" : 0.5
}

config = default_config

class Model(BaseModel):
    # 用BiLSTM建模文档，再用全连接网络和avg_pooling或max_pooling处理图片, concate后输出
    def __init__(self, config:dict):
        super(Model, self).__init__()
        self.word_embedding = config["embedding"]
        if not config["freeze_embedding"]:
            self.word_embedding.requires_grad_(True)
        else:
            self.word_embedding.requires_grad_(False)

        self.filter_size = config["filter_size"]
        self.conv_list = nn.ModuleList([nn.Conv2d(1, config["filter_num"], 
                    (size, config["embedding_dim"])) for size in self.filter_size])
        self.relu = nn.ReLU()

        self.use_img = config["use_img"]
        if config["use_img"]:
            self.img_encoder = SimpleImageEncoder(config["img_input_size"], config["img_output_size"], config["img_num"], config["dropout"])

        self.output_layer = OutputLayer(config["task"], len(config["filter_size"]) * config["filter_num"] +\
                int(config["use_img"]) * config["img_output_size"], config["output_size"], config["dropout"])

    def forward(self, X):
        text, lens, imgs = X
        if torch.__version__ == "1.7.0":
            lens = lens.cpu()
        emb = self.word_embedding(text)
        emb = emb.unsqueeze(1)
        conv_output = []
        for size, conv in zip(self.filter_size, self.conv_list):
            o = conv(emb) # （batch_size, num_filter, seq_len - size, 1) 
            o = o.squeeze() # (batch_size, num_filter, seq_len - size)
            o = self.relu(o)
            o, _ = o.max(dim=2) # (batch_size, num_filter)
            conv_output.append(o)
        h = torch.cat(conv_output, dim=1) #(batch_size, num_filter * len(filter_size))

        if self.use_img:
            imgs_out = self.img_encoder(imgs)
            fc_input = torch.cat((h, imgs_out), dim=1)
        else:
            fc_input = h
        return self.output_layer(fc_input)

def get_collate_fn(config):
    def collate_fn(batch):
        global config
        text, imgs, y = [], [], []
        for (_text, _imgs), _y in batch:
            t = []
            for i in _text:
                t += i
            text.append(t)
            for i in _imgs:
                feature = load_vgg_features(i)
                if feature is not None:
                    imgs.append(feature)
            # imgs += _imgs # _imgs: List[4096, 4096, 4096]
            y.append(_y)
        padded_text, lens = padding_text_1d(text, config["max_tokens"])
        imgs = torch.tensor(imgs) # 转为张量
        y = torch.tensor(y) #
        res = ((padded_text, lens, imgs), y)
        return res
    return collate_fn