import torch
from torch import nn
from models.layers import *
from utils.dataset import padding_text_1d
from utils.load_yelp import load_rcnn_features


default_config = {
    "task" : "CLS",
    "embedding_dim" : 100,
    "embedding" : None, 
    "freeze_embedding" : True,
    "text_hidden_size" : 100,
    "text_layers" : 1,
    "bias_init" : 1.0,

    "img_input_size" : 2048,
    "img_output_size" : 100,
    "img_num" : 3,
    "output_size" : 5, 
    "uniform_bound" : 0.1,

    "dropout" : 0.5
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

        self.gru = DynamicRNN(config["embedding_dim"], hidden_size=config["text_hidden_size"], 
                                num_layers=config["text_layers"], dropout=config["dropout"],
                                bias_init=config["bias_init"], batch_first=True, bidirectional=True, 
                                only_use_last_hidden_state=True, rnn_type="GRU")
        self.img_encoder = nn.Linear(config["img_input_size"], config["img_output_size"])
        self.img_dropout = nn.Dropout(config["dropout"])
        self.output_layer = OutputLayer(config["task"], config["text_hidden_size"] * config["text_layers"] * 2 +\
                config["img_output_size"], config["output_size"], config["dropout"])

    def forward(self, X):
        padded_text, lens, padded_x = X
        if torch.__version__ == "1.7.0":
            lens = lens.cpu()
        emb = self.word_embedding(padded_text)
        hn = self.gru(emb, lens)
        hn = hn_reshape(hn)
        img_out = self.img_dropout(self.img_encoder(padded_x))
        img_out = img_out.mean(dim=1)
        fc_input = torch.cat((hn, img_out), dim=1)
        return self.output_layer(fc_input)

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
        padded_text, lens = padding_text_1d(text)
        max_lens = max(lens)
        max_bbox = max(bbox_num)
        padded_x = torch.zeros((n, max_bbox, config["img_input_size"]), dtype=torch.float32)
        for i, (l, s) in enumerate(zip(lens, bbox_num)):
            padded_x[i,:s] = torch.from_numpy(x[i])
        bbox_num = torch.LongTensor(bbox_num)
        y = torch.tensor(y)
        return (padded_text, lens, padded_x), y
    return collate_fn
