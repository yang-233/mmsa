import torch
from torch import nn

from utils.dataset import padding_text_2d
from models.layers import *

default_config = {
    "task" : "CLS",
    "embedding_dim" : 200,
    "embedding" : None, 
    "freeze_embedding" : False,
    "word_hidden_size" : 50,
    "word_layers" : 1,
    "uniform_bound" : 0.1,
    "sentence_hidden_size" : 50,
    "sentence_layers" : 1,

    "img_input_size" : 4096,
    "visual_attention_size" : 100,
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

        self.word_encoder = AttentionDynamicRNN(config["embedding_dim"], config["word_hidden_size"], rnn_num_layers=config["word_layers"],
                uniform_bound=config["uniform_bound"], batch_first=True, dropout=config["dropout"], bidirectional=True,
                                rnn_type="GRU", bias_init=config["bias_init"])

        self.padding_layer = PaddingLayer()

        self.sent_encoder = DynamicRNN(config["word_hidden_size"] * config["word_layers"] * 2, config["sentence_hidden_size"], 
            config["sentence_layers"], dropout=config["dropout"], bidirectional=True, rnn_type="GRU")

        self.sent_size = 2 * config["sentence_layers"] * config["sentence_hidden_size"]

        self.sent_proj_fc = nn.Linear(self.sent_size, self.sent_size)
        self.sent_proj_tanh = nn.Tanh()
        self.img_proj_fc = nn.Linear(config["img_input_size"], self.sent_size)
        self.img_proj_tanh = nn.Tanh()
        self.v = nn.Parameter(data=torch.from_numpy(np.random.uniform(-config["uniform_bound"], config["uniform_bound"],
                              size=(self.sent_size))).type(torch.float32))
        self.visual_softmax = nn.Softmax(dim=2)
        
        self.doc_proj_fc = nn.Linear(self.sent_size, self.sent_size)
        self.doc_proj_tanh = nn.Tanh()
        self.k = nn.Parameter(data=torch.from_numpy(np.random.uniform(-config["uniform_bound"], config["uniform_bound"],
                              size=(self.sent_size))).type(torch.float32))
        self.doc_visual_softmax = nn.Softmax(dim=1)
        
        self.output_layer = OutputLayer(config["task"], self.sent_size, config["output_size"], dropout=config["dropout"])

    def forward(self, X):
        padded_text, lens, sents_num, imgs = X
        # 句子编码
        emb = self.word_embedding(padded_text)
        si = self.word_encoder(emb, lens)
        padded_si = self.padding_layer(si, sents_num)
        # 文档编码
        hi, _ = self.sent_encoder(padded_si, sents_num) # gru
        b, seq, d = hi.shape
        
        pj = self.img_proj_tanh(self.img_proj_fc(imgs)) # img编码
        pj = pj.view(b, -1, 1, self.sent_size)
        qi = self.sent_proj_tanh(self.sent_proj_fc(hi)) # text 编码
        qi = qi.view(b, 1, seq, d)
        vji = (pj * qi + pj) @ self.v # 注意这里保留了没有特征变换的text编码

        masked_vji = torch.full_like(vji, float("-inf")) # b, img_num, seq
        for i in range(b):
            masked_vji[i,:,:sents_num[i]] = vji[i,:,:sents_num[i]]
        v_score = self.visual_softmax(masked_vji) # b, img_num, seq
        v_score = v_score.view(*v_score.shape, 1)
        hi = hi.view(b, 1, seq, d)
        dj = (v_score * hi).sum(dim=2) # b, img_num, d
        
        kj = self.doc_proj_tanh(self.doc_proj_fc(dj)) @ self.k # 这里实际已经不需要mask了
        doc_v_score = self.doc_visual_softmax(kj)
        doc_v_score = doc_v_score.view(*doc_v_score.shape, 1)
        d = (dj * doc_v_score).sum(dim=1)
        
        return self.output_layer(d)

def get_collate_fn(config:dict):
    mean = np.load("data/yelp-vistanet/raw/photo_features/mean.npy")
    def func(batch):
        text, imgs, y = [], [], []
        for (_text, _imgs), _y in batch:
            text.append(_text)
            imgs.append(mean.copy()) # 空白图片
            imgs += _imgs # _imgs: List[4096, 4096, 4096]
            y.append(_y)
        padded_text, lens, sents_num = padding_text_2d(text) # padding文本
        imgs = torch.tensor(imgs) # 转为张量
        y = torch.tensor(y) #
        return (padded_text, lens, sents_num, imgs), y
    return func