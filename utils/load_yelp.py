import pickle
import os
import numpy as np
import torch
import collections
from tqdm import tqdm

import sys
sys.path.append("/home/ly/workspace/mmsa")
from utils.dataset import *
from utils.tokenization import BasicTokenizer

split811data = os.path.join("data", "yelp-vistanet", "811data")
split622data = os.path.join("data", "yelp-vistanet", "622data")
base_dir = os.path.join("data", "yelp-vistanet")

CATEGORY_NUM = 5
IMG_NUM = 3
def load_data(dir):
    with open(os.path.join(dir, "clear_data.pickle"), "rb") as r:
        return pickle.load(r)

class YelpSimpleTokenizer(BasicTokenizer):
    def __init__(self, vocab:Dict[str, int]=None, do_lower_case:bool=True) -> None:
        super(YelpSimpleTokenizer, self).__init__(do_lower_case)
        self.SENT_DELIMITER = '|||'
        self.vocab = vocab
        self.UNK = len(vocab) + 1 if vocab is not None else None # 

    def tokenize(self, text:str) -> List[str]: # 默认切成2d
        res = []
        for sent in text.split(self.SENT_DELIMITER):
            if len(sent) > 0: # 有一定几率出现空字符串
                res.append(super(YelpSimpleTokenizer, self).tokenize(sent))
        return res

    def _getidx(self, token:str):
        return self.vocab.get(token, self.UNK)
        
    def to_idx(self, text:str) -> List[int]:
        assert self.vocab is not None, "No vocab!"
        sents = self.tokenize(text)
        res = []
        for sent in sents:
            res.append([self._getidx(token) for token in sent])
        return res

def count_word_freq(reviews:List[dict], freq_dict:Dict[str, int]=None) -> Dict[str, int]: 
    # 统计词频
    tokenizer = YelpSimpleTokenizer(do_lower_case=True)
    if freq_dict is None:
        freq_dict = {}
    for review in tqdm(reviews, "Count word frequency"):
        text = review["Text"]
        for sent in tokenizer.tokenize(text):
            for token in sent:
                freq_dict[token] = freq_dict.get(token, 0) + 1
    return freq_dict

def build_vocab_from_glove(freq_dict:Dict[str, int]):
    glove_dict, _ = load_vocab_file(os.path.join("pretrained", "glove6B", "vocab.txt"))
    _vocab = list(filter(lambda item: item[0] in glove_dict, freq_dict.items())) # 删除掉不在glove中的词
    _vocab = sorted(_vocab, key=lambda item: item[1], reverse=True) # 降序排序
    print(f"There are {len(_vocab)} words in vocab.")
    token2idx = collections.OrderedDict()
    glove_idx = []
    idx2token = ["[PAD]"]
    idx = 1
    for key, val in _vocab:
        token2idx[key] = idx
        idx2token.append(key)
        glove_idx.append(glove_dict[key]) # 用来读取glove词向量
        idx += 1
    idx2token.append("[UNK]")
    return token2idx, idx2token, glove_idx

def save_vocab(dir, token2idx, idx2token, glove_idx):
    vals = {"token2idx" : token2idx,
           "idx2token" : idx2token,
           "glove_idx" : glove_idx}
    path = os.path.join(dir, "glove_vocab.pickle")
    with open(path, "wb") as o:
        pickle.dump(vals, o, protocol=pickle.HIGHEST_PROTOCOL)

def load_vocab_file(path:str):
    token2idx = collections.OrderedDict()
    idx2token = []
    idx = 0
    with open(path, "r") as r:
        for line in tqdm(r):
            key = line.strip()
            idx2token.append(key)
            token2idx[key] = idx
            idx += 1
    return token2idx, idx2token
    
def load_glove_vocab(dir):
    path = os.path.join(dir, "glove_vocab.pickle")
    with open(path, "rb") as r:
        return pickle.load(r)

def load_glove_weight(d:int):
    path = os.path.join("pretrained", "glove6B", "glove.6B." + str(d) + "d.txt")
    NUM = 400000
    weight = np.empty((NUM, d), dtype=np.float32)
    with open(path, "r", encoding='utf-8') as r:
        for i, line in enumerate(tqdm(r.readlines(), "Load glove")):
            values = line.split()
            weight[i] = np.asarray(values[1:], dtype=np.float32)
    return weight

def get_yelp_glove_weight(dir, d:int, _uniform:float=0.1):
    path = os.path.join(dir, "glove6B" + str(d) + "d.npy")
    if os.path.exists(path):
        return np.load(path)
    glove_weight = load_glove_weight(d)
    vocab = load_glove_vocab(dir)
    n = len(vocab["token2idx"]) 
    weight = np.empty((n + 12, d), dtype=np.float32) 
    weight[0] = np.zeros(d, dtype=np.float32) # [PAD]
    glove_weight = glove_weight[vocab["glove_idx"]]
    weight[1:n+1] = glove_weight # 正文
    weight[n+1] = glove_weight.mean(axis=0) # [UNK]
    weight[n+2:] = np.random.uniform(-_uniform, _uniform, size=(10, d))
    np.save(path, weight)
    return weight

def load_glove_vgg_data(dir, config:dict):
    vocab = load_glove_vocab(dir)
    config["vocab_size"] = len(vocab["token2idx"])
    embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(get_yelp_glove_weight(dir, config["embedding_dim"])))
    config["embedding"] = embedding

    config["output_size"] = CATEGORY_NUM
    with open(os.path.join(dir, "glove_vgg_data.pickle"), "rb") as r:
        data = pickle.load(r)
    train_set = MultiModalDataset.from_yelp(data["train"], config)
    valid_set = MultiModalDataset.from_yelp(data["valid"], config)
    test_set = MultiModalDataset.from_yelp(data["test"], config)
    return train_set, valid_set, test_set

def load_electra_vgg_data(config:dict):
    config["output_size"] = CATEGORY_NUM
    with open(os.path.join(dir, "electra_vgg_data.pickle"), "rb") as r:
        data = pickle.load(r)
    train_set = MultiModalDataset.from_yelp(data["train"], config)
    valid_set = MultiModalDataset.from_yelp(data["valid"], config)
    test_set = MultiModalDataset.from_yelp(data["test"], config)
    return train_set, valid_set, test_set

def glove_vgg_collate_2d(batch):
    text, imgs, y = [], [], []
    for (_text, _imgs), _y in batch:
        text.append(_text)
        imgs += _imgs # _imgs: List[4096, 4096, 4096]
        y.append(_y)
    padded_text, lens, sents_num = padding_text_2d(text) # padding文本
    imgs = torch.tensor(imgs) # 转为张量
    y = torch.tensor(y) #
    return (padded_text, lens, sents_num, imgs), y

features = {}
def load_vgg_features(i):
    if i not in features:    
        path = base_dir + "/vgg16/" + i[:2] + "/" + i + ".npy"
        if os.path.exists(path):
            features[i] = np.load(path)
        else:
            features[i] = None
    return features[i]

def load_rcnn_features(i): 
    path = os.path.join(base_dir, "rcnn_data", i[:2], i + ".npz")
    if os.path.exists(path):
        d = {}
        npz = np.load(path)
        d["x"] = npz["x"]
        d["bbox"] = npz["bbox"]
        return d
    else:
        return None

### !内存不够
def load_rcnn_data(name="valid"):
    num = {"train" : 14, "valid" : 2, "test" : 2}
    if name not in num:
        print("name must be train or valid or test")
    res = {}
    for i in range(num[name]):
        path = os.path.join(base_dir, "rcnn_" + name + str(i) + ".pickle")
        with open(path, "rb") as r:
            l = pickle.load(r)
        res.update(l)
    return res

def load_glove_data(dir, config:dict):
    vocab = load_glove_vocab(dir)
    config["vocab_size"] = len(vocab["token2idx"])
    embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(get_yelp_glove_weight(dir, config["embedding_dim"])))
    config["embedding"] = embedding

    config["output_size"] = CATEGORY_NUM
    with open(os.path.join(dir, "glove_data.pickle"), "rb") as r:
        data = pickle.load(r)
    train_set = MultiModalDataset.from_yelp(data["train"], config)
    valid_set = MultiModalDataset.from_yelp(data["valid"], config)
    test_set = MultiModalDataset.from_yelp(data["test"], config)
    return train_set, valid_set, test_set