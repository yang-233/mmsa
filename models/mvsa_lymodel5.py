import torch
from models.layers import *
from utils.dataset import padding_text_1d
from utils.load_mvsa import *

default_config = {
    "task" : "CLS",
    "embedding_dim" : 100,
    "embedding" : None, 
    "freeze_embedding" : True,
    "text_hidden_size" : 100,
    "text_layers" : 1,
    "bias_init" : 1.0,
    "uniform_bound" : 0.1,
    
    "img_input_size" : 2048,
    "img_encoder_layers" : 1,
    "attention_nhead" : 4,

    "fusion_nheads" : 4,
    "dropout" : 0.1,
    "output_size" : 3
}

config = default_config
# 使用门控机制控制图像输入 而不是直接将图片信息concat上去
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
        self.tanh1 = nn.Tanh()
        self.attn = nn.MultiheadAttention(self.word_output_size, config["attention_nhead"])

        self.forget_gate_img_1 = nn.Linear(self.word_output_size, self.word_output_size, bias=False)
        self.forget_gete_text_1 = nn.Linear(self.word_output_size, self.word_output_size) # 两者只需要其中一个有偏置
        self.sigmoid = nn.Sigmoid()

        self.forget_gate_img_2 = nn.Linear(self.word_output_size, self.word_output_size, bias=False)
        self.forget_gete_text_2 = nn.Linear(self.word_output_size, self.word_output_size)
        self.tanh2 = nn.Tanh()

        self.fusion_encoder = MultiheadAttentionEncoder(self.word_output_size, config["fusion_nheads"], config["uniform_bound"])

        self.output_layer = OutputLayer(config["task"], self.word_output_size,
                             config["output_size"], config["dropout"])


    def forward(self, X):
        padded_text, lens, text_key_padding_mask, padded_x, img_key_padding_mask = X
        if torch.__version__ == "1.7.0":
            lens = lens.cpu()
        emb = self.word_embedding(padded_text)
        output, (hn, _) = self.word_encoder(emb, lens) # output(batch_size, seq, d)

        padded_x = padded_x.permute(1, 0, 2) # (seq, batch_size, d)
        transform_x = self.tanh1(self.img_fc(padded_x))  # (batch_size, seq, d)
        transform_x = transform_x.permute(1, 0, 2) # (seq, batch_size, d)

        attn_bbox, score = self.attn(output.permute(1, 0, 2), transform_x, 
                        transform_x, key_padding_mask=img_key_padding_mask)
        attn_bbox = attn_bbox.permute(1, 0, 2) # (batch_size, seq, d)

        forget_val = self.sigmoid(self.forget_gete_text_1(output) + self.forget_gate_img_1(attn_bbox)) 
        forget_val = forget_val * attn_bbox

        fusion_input = self.tanh2(self.forget_gete_text_2(output) + self.forget_gate_img_2(forget_val))
        fusion_output = self.fusion_encoder(fusion_input, text_key_padding_mask)
        return self.output_layer(fusion_output)

def get_collate_fn(config:dict):
    def collate_fn(batch):
        text, x, bbox_num, y = [], [], [], []
        n = len(batch)
        for (_text, _id), _y in batch:
            text.append(_text)
            feature = load_rcnn(_id)
            x.append(feature["x"])
            bbox_num.append(len(x[-1])) # bbox num
            y.append(_y)
        padded_text, lens = padding_text_1d(text)
        max_lens = max(lens)
        max_bbox = max(bbox_num)
        padded_x = torch.zeros((n, max_bbox, config["img_input_size"]), dtype=torch.float32)
        img_key_padding_mask = torch.ones((n, max_bbox), dtype=torch.bool)
        text_key_padding_mask = torch.ones((n, max_lens), dtype=torch.bool)
        for i, (l, s) in enumerate(zip(lens, bbox_num)):
            padded_x[i,:s] = torch.from_numpy(x[i])
            img_key_padding_mask[i,:s] = False
            text_key_padding_mask[i,:l] = False
        bbox_num = torch.LongTensor(bbox_num)
        y = torch.tensor(y)
        return (padded_text, lens, text_key_padding_mask, padded_x, img_key_padding_mask), y
    return collate_fn
