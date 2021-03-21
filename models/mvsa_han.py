import torch
from torch import nn

from utils.dataset import padding_text_2d
from models.layers import *
from utils.load_mvsa import *


default_config = {
    "task" : "CLS",
    "embedding_dim" : 50,
    "embedding" : None, 
    "freeze_embedding" : False,
    "word_hidden_size" : 100,
    "word_layers" : 1,
    "uniform_bound" : 0.1,
    "sentence_hidden_size" : 100,
    "sentence_layers" : 1,

    "use_imgs" : True,
    "img_input_size" : 4096,
    "img_output_size" : 100,
    "img_num" : 1,
    "dropout" : 0.5,
    "output_size" : 5,
    "bias_init" : 1.0
}

config = default_config
class Model(BaseModel):
    def __init__(self, config:dict):
        super(Model, self).__init__()
        self.word_embedding = config["embedding"]
        if not config["freeze_embedding"]:
            self.word_embedding.requires_grad_(True)
        else:
            self.word_embedding.requires_grad_(False)

        self.word_encoder = AttentionDynamicRNN(config["embedding_dim"], config["word_hidden_size"], 
                                rnn_num_layers=config["word_layers"], uniform_bound=config["uniform_bound"],
                                batch_first=True, dropout=config["dropout"], bidirectional=True,
                                rnn_type="GRU", bias_init=config["bias_init"])

        self.padding_layer = PaddingLayer()

        self.sent_encoder = AttentionDynamicRNN(config["word_hidden_size"] * config["word_layers"] * 2, config["sentence_hidden_size"], 
                                rnn_num_layers=config["sentence_layers"], uniform_bound=config["uniform_bound"], 
                                batch_first=True, dropout=config["dropout"], bidirectional=True,
                                rnn_type="GRU", bias_init=config["bias_init"])

        self.sent_output_size = config["sentence_hidden_size"] * config["sentence_layers"] * 2
        self.use_imgs = config["use_imgs"]

        self.fc_size = self.sent_output_size

        if self.use_imgs:
            self.img_encoder = SimpleImageEncoder(config["img_input_size"], config["img_output_size"], config["img_num"], config["dropout"])
            self.fc_size += config["img_output_size"]

        self.output_layer = OutputLayer(config["task"], self.fc_size, config["output_size"], dropout=config["dropout"])

    
    def forward(self, X):
        padded_text, lens, sents_num, imgs = X
        # 句子编码
        if torch.__version__ == "1.7.0":
            lens = lens.cpu()
            sents_num = sents_num.cpu()
        emb = self.word_embedding(padded_text)
        si = self.word_encoder(emb, lens)
        padded_si = self.padding_layer(si, sents_num)
        # 文档编码
        d = self.sent_encoder(padded_si, sents_num)
        if self.use_imgs:
            imgs_out = self.img_encoder(imgs)
            fc_input = torch.cat((d, imgs_out), dim=1)
        else:
            fc_input = d
        return self.output_layer(fc_input)

def get_collate_fn(config:dict):
    mean = np.load(DIR + "vgg16/" + "MEAN.npy")
    def func(batch):
        text, imgs, y = [], [], []
        for (_text, _id), _y in batch:
            text.append([_text])
            imgs.append(load_vgg(_id)) # _imgs: List[4096, 4096, 4096]
            y.append(_y)
        padded_text, lens, sents_num = padding_text_2d(text) # padding文本
        imgs = torch.tensor(imgs) # 转为张量
        y = torch.tensor(y) #
        return (padded_text, lens, sents_num, imgs), y
    return func