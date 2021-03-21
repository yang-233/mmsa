import torch
from utils.train import to_cuda
from models.layers import *
from utils.dataset import padding_text_1d
from utils.load_yelp import *
default_config = {
    "use_cuda" : True,
    "task" : "CLS",
    "embedding_dim" : 200,
    "embedding" : None, 
    "freeze_embedding" : False,
    "text_hidden_size" : 50,
    "text_layers" : 1,
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

        self.gru = DynamicRNN(config["embedding_dim"], hidden_size=config["text_hidden_size"], 
                                num_layers=config["text_layers"], dropout=config["dropout"],
                                bias_init=config["bias_init"], batch_first=True, bidirectional=True, 
                                only_use_last_hidden_state=True, rnn_type="GRU")
        self.use_img = config["use_img"]
        if config["use_img"]:
            self.img_encoder = SimpleImageEncoder(config["img_input_size"], config["img_output_size"], config["img_num"], config["dropout"])

        self.output_layer = OutputLayer(config["task"], config["text_hidden_size"] * config["text_layers"] * 2 +\
                int(config["use_img"]) * config["img_output_size"], config["output_size"], config["dropout"])

    def forward(self, X):
        text, lens, imgs = X
        if torch.__version__ == "1.7.0":
            lens = lens.cpu()
        emb = self.word_embedding(text)
        hn = self.gru(emb, lens)
        hn = hn_reshape(hn)
        if self.use_img:
            imgs_out = self.img_encoder(imgs)
            fc_input = torch.cat((hn, imgs_out), dim=1)
        else:
            fc_input = hn
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