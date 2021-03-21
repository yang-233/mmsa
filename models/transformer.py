import torch
from torch import nn
from models.layers import *
from utils.dataset import padding_text_2d

default_config = {
    "task" : "CLS",
    "embedding_dim" : 200,
    "embedding" : None, 
    "freeze_embedding" : False,
    "word_nhead" : 4,
    "word_encoder_layers" : 2,
    "word_dimfeedward_size" : 2048,
    "learned_word_pos_embedding" : False,
    "sent_nhead" : 4,
    "sent_dimfeedward_size" : 2048,
    "sent_encoder_layers" : 2,
    "learned_sent_pos_embedding" : False,

    "img_input_size" : 4096,
    "img_output_size" : 100,

    "max_tokens" : 50, 
    "max_sents" : 50,
    "sent_head" : 8,
    "img_num" : 3,
    "output_size" : 5, 
    "dropout" : 0.3
}

class Model(BaseModel):
    def __init__(self, config:dict):
        super(Model, self).__init__()
        self.word_embedding = config["embedding"]
        if not config["freeze_embedding"]:
            self.word_embedding.requires_grad_(True)
        else:
            self.word_embedding.requires_grad_(False)

        _layers = nn.TransformerEncoderLayer(config["embedding_dim"], config["word_nhead"], config["word_dimfeedward_size"])
        _transformer = nn.TransformerEncoder(_layers, config["word_encoder_layers"])
        self.word_encoder = TransformerCLSEncoder(config["embedding_dim"], _transformer, max_lens=config["max_tokens"], 
                                        learned_pos_emb=config["learned_word_pos_embedding"])

        
        self.padding_layer = PaddingLayer()
        
        _layers = nn.TransformerEncoderLayer(config["embedding_dim"], config["sent_nhead"], config["sent_dimfeedward_size"])
        _transformer = nn.TransformerEncoder(_layers, config["sent_encoder_layers"])
        self.sent_encoder = TransformerCLSEncoder(config["embedding_dim"], _transformer, max_lens=config["max_sents"], 
                                        learned_pos_emb=config["learned_sent_pos_embedding"])

        self.img_encoder = SimpleImageEncoder(config["img_input_size"], config["img_output_size"], config["img_num"], config["dropout"])

        self.output_layer = OutputLayer(config["task"], config["embedding_dim"] + config["img_output_size"],
                            config["output_size"], config["dropout"])
    
    def forward(self, X):
        input_ids, token_lens, sent_lens, imgs = X
        emb = self.word_embedding(input_ids)
        hi = self.word_encoder(emb, token_lens)
        padded_hi = self.padding_layer(hi, sent_lens)
        doc_hn = self.sent_encoder(padded_hi, sent_lens)
        imgs_out = self.img_encoder(imgs)
        fc_input = torch.cat((doc_hn, imgs_out), dim=1)
        return self.output_layer(fc_input)

def get_collate_fn(config:dict):
    def func(batch):
        text, imgs, y = [], [], []
        for (_text, _imgs), _y in batch:
            text.append(_text)
            imgs += _imgs # _imgs: List[4096, 4096, 4096]
            y.append(_y)
        padded_text, lens, sents_num = padding_text_2d(text, config["max_tokens"], config["max_sents"]) # padding文本
        imgs = torch.tensor(imgs)
        y = torch.tensor(y)
        return (padded_text, lens, sents_num, imgs), y
    return func
