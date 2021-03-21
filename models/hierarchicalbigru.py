import torch
from torch import nn

from preprocess import load_new_glove_2d_dataset, get_2d_loader
from models.layers import *

default_config = {
    "embedding" : None, 
    "freeze_embedding" : False,
    "word_hidden_size" : 50,
    "word_layers" : 1,
    "sentence_hidden_size" : 50,
    "sentence_layers" : 1,
    "bias_init" : 1.0,

    "use_imgs" : True,
    "img_hidden_size" : 100,
    "pooling" : "MAX", # or AVG
    "task" : "CLS", # or "REG",
    "dropout" : 0.5
}
class Model(nn.Module):
    def __init__(self, config:dict):
        super(Model, self).__init__()
        self.word_embedding = config["embedding"]
        if not config["freeze_embedding"]:
            self.word_embedding.requires_grad_(True)
        else:
            self.word_embedding.requires_grad_(False)

        self.word_encoder = DynamicRNN(EMBEDDING_SIZE, hidden_size=config["word_hidden_size"], 
                                num_layers=config["word_layers"], dropout=config["dropout"],
                                bias_init=config["bias_init"], batch_first=True, bidirectional=True, 
                                only_use_last_hidden_state=True, rnn_type="GRU")
        self.word_output_size = config["word_layers"] * 2 * config["word_hidden_size"]

        self.padding_layer = SentencePaddingLayer()
        self.sentence_encoder = DynamicRNN(self.word_output_size, hidden_size=config["sentence_hidden_size"],
                                num_layers=config["sentence_layers"], dropout=config["dropout"],
                                bias_init=config["bias_init"], batch_first=True, bidirectional=True, 
                                only_use_last_hidden_state=True, rnn_type="GRU")

        self.sentence_output_size = config["sentence_layers"] * 2 * config["sentence_hidden_size"]

        self.use_imgs = config["use_imgs"]
        self.imgs_output_size = 0

        if self.use_imgs:
            self.imgs_output_size = config["img_hidden_size"]
            self.img_encoder = nn.Linear(IMG_SIZE, self.imgs_output_size)
            self.img_dropout = nn.Dropout(config["dropout"], inplace=True)
            self.img_relu = nn.ReLU(True)
            if config["pooling"] == "AVG":
                self.pooling = nn.AvgPool2d((IMGS_NUM, 1))
            elif config["pooling"] == "MAX" :
                self.pooling = nn.MaxPool2d((IMGS_NUM, 1))
            else:
                raise RuntimeError("pooling must be AVG or MAX")

        self.fc_size = self.sentence_output_size + self.imgs_output_size
        if config["task"] == "CLS":
            self.output_layers = ClassificationOutputLayer(self.fc_size)
        elif config["task"] == "REG":
            self.output_layers = RegressionOutputLayer(self.fc_size)
        else:
            raise RuntimeError("task must be CLS or REG")

        self.output_dropout = nn.Dropout(config["dropout"], inplace=True)
    
    def forward(self, X):
        padding_text, lens, sents_num, imgs = X
        if (lens <= 0).sum() > 0:
            print("lens <= 0 ", lens)
        if (sents_num <= 0).sum() > 0:
            print("sents_num <= 0 ", sents_num)

        emb = self.word_embedding(padding_text)
        word_hn = self.word_encoder(emb, lens)
        word_hn = hn_reshape(word_hn)
        padded_word_hn = self.padding_layer(word_hn, lens, sents_num)
        sentence_hn = self.sentence_encoder(padded_word_hn, sents_num)
        sentence_hn = hn_reshape(sentence_hn)

        

        # return sentence_hn
        if self.use_imgs:
            imgs = self.img_encoder(imgs)
            imgs = self.img_dropout(imgs)
            imgs = self.img_relu(imgs)
            imgs = imgs.view((-1, 1, IMGS_NUM, self.imgs_output_size))
            imgs = self.pooling(imgs).view((-1, self.imgs_output_size))
            fc_input = torch.cat((sentence_hn, imgs), dim=1)
        else:
            fc_input = sentence_hn

        fc_out = self.output_layers(fc_input)
        return self.output_dropout(fc_out)

    def predict(self, output:torch.Tensor) -> list:
        return self.output_layers.predict(output)
    def cuda(self, device=None):
        self.padding_layer.cuda(device)
        return super(Model, self).cuda(device)

    def cpu(self):
        self.padding_layer.cpu()
        return super(Model, self).cpu()

def load_model_and_dataset(config:dict=None):
    if config is None:
        global default_config
        config = default_config
    trainset, validset, testset, config["embedding"] = load_new_glove_2d_dataset()
    return Model(config), trainset, validset, testset, get_2d_loader