import torch
from torch import nn
from models.layers import *
from transformers import ElectraConfig, ElectraModel
from utils.dataset import padding_text_2d

default_config = {
    "task" : "CLS",
    "pretrained_dir" : "/home/ly/workspace/mmsa/pretrained/electra_small/",
    "img_input_size" : 4096,
    "img_output_size" : 100,
    "max_tokens" : 50, 
    "max_sents" : 50,
    "sent_head" : 8,
    "img_num" : 3,
    "output_size" : 5, 
    "dropout" : 0.5
}

        
class Model(nn.Module):
    def __init__(self, config:dict):
        super(Model, self).__init__()
        self.electra_cfg = ElectraConfig()
        self.electra = ElectraModel.from_pretrained(config["pretrained_dir"] + "electra_small.index",
                                                   config=self.electra_cfg, from_tf=True)
        
        self.sentence_encoder = AttentionSentenceEncoder(self.electra_cfg.hidden_size, config["sent_head"], config["max_sents"] + 1) # 多一个位置给CLS
        self.img_encoder = SimpleImageEncoder(config["img_input_size"], config["img_output_size"],
                                             config["img_num"], dropout=config["dropout"])
        
        self.output_layer = OutputLayer(config["task"], self.electra_cfg.hidden_size + config["img_output_size"], config["output_size"],
                                       config["dropout"])
        
    def forward(self, X):
        input_ids, attention_mask, token_type_ids, position_ids, sents_num, imgs = X
        output = self.electra(input_ids, attention_mask, token_type_ids, position_ids, return_dict=True)
        sent_features = output.last_hidden_state[:,0]
        doc_features, scores = self.sentence_encoder(sent_features, sents_num)
        imgs_out = self.img_encoder(imgs)
        fc_input = torch.cat((doc_features, imgs_out), dim=1)
        return self.output_layer(fc_input)

    def predict(self, res):
        return self.output_layer.predict(res)


def get_collate_2d(config:dict):

    def electra_vgg_collate_2d(batch):
        text, imgs, y = [], [], []
        for (_text, _imgs), _y in batch:
            text.append(_text)
            imgs += _imgs # _imgs: List[4096, 4096, 4096]
            y.append(_y)
        padded_text, lens, sents_num = padding_text_2d(text, config["max_tokens"], config["max_sents"]) # padding文本
        mask = torch.zeros(padded_text.shape, dtype=torch.float32) # 
        for i, l in enumerate(lens): # mask
            mask[i][:l] = 1
        token_type_ids = torch.zeros(padded_text.shape, dtype=torch.long)
        position_ids = torch.zeros(padded_text.shape, dtype=torch.long)
        imgs = torch.tensor(imgs) # 转为张量
        y = torch.tensor(y) #
        return (padded_text, mask, token_type_ids, position_ids, sents_num, imgs), y
    return electra_vgg_collate_2d
    