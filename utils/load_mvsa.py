import json
import os
import pickle
from collections import OrderedDict
import re
import random
import numpy as np
import torch
from tqdm import tqdm
from .dataset import MultiModalDataset
DIR = "data/mvsa/MVSA-multiple/"
data_path = DIR + "data.json"
CATEGORY_NUM = 3

def load_data():
    data = None
    with open(data_path, "r") as r:
        data = json.load(r)
    return data

def build_glove_vocab():
    vocab = OrderedDict()
    with open("pretrained/glove27b/glove.twitter.27B.25d.txt", "r") as r:
        for i, l in enumerate(r.readlines()):
            l = l.strip().split()
            vocab[l[0]] = i
    topic_vocab = set()
    for i in vocab.keys():
        if len(i) >= 2 and i.startswith("#"):
            topic_vocab.add(i)
    return vocab, topic_vocab

p1 = re.compile(r"&[a-zA-Z]+;") # 清除转义字符
p2 = re.compile(r"([\W])") # 用于切分字符串 

def clear_text(text, topic_vocab):
    res = []
    split_text = text.strip().lower().split()
    for i in split_text:
        if i.startswith("http") or p1.match(i) is not None: # 忽略URL和转义字符
            continue
        elif i.startswith("@"): # 
            res.append(i[1:])
        elif i.startswith("#"):
            if len(i) >= 2:
                if i in topic_vocab: # 在词表中 则直接加入
                    res.append(i)
                else: # 否则拆分
                    res.append("#")
                    res.append(i[1:])
            else:
                res.append(i)
        else: # 其他类型切分然后加入结果中
            i = p2.split(i)
            for _i in i:
                if len(_i) > 0: # 会有空字符串 忽略
                    res.append(_i)
    return res 

def build_freq(freq, data):
    for i in data:
        for j in i["text"]:
            freq[j] = freq.get(j, 0) + 1
    return freq

def build_vocab_from_glove(freq_dict, glove_vocab, dir):
    _vocab = list(filter(lambda item: item[0] in glove_vocab, freq_dict.items())) # 删除掉不在glove中的词
    _vocab = sorted(_vocab, key=lambda item: item[1], reverse=True) # 降序排序
    token2idx = OrderedDict()
    glove_idx = []
    idx = 1
    for key, val in _vocab:
        token2idx[key] = idx
        glove_idx.append(glove_vocab[key]) # 用来读取glove词向量
        idx += 1
    d = {}
    d["token2idx"] = token2idx
    d["glove_idx"] = glove_idx
    with open(dir + "glove_vocab.pickle", "wb") as o:
        pickle.dump(d, o, protocol=pickle.HIGHEST_PROTOCOL)
    return d

def load_glove_vocab():
    with open(DIR + "glove_vocab.pickle", "rb") as r:
        return pickle.load(r)
    return None

UNK_NUM = 100
class GloveTokenizer:
    def __init__(self, glove_vocab, unk_num:int=UNK_NUM):
        self.vocab = glove_vocab
        self.vocab_size = len(glove_vocab)
        self.unk_num = unk_num
        print(self.vocab_size + unk_num)
    def tokenize(self, tokens_list):
        res = []
        for i in tokens_list:
            if i in self.vocab:
                res.append(self.vocab[i])
            else:
                res.append(random.randint(self.vocab_size + 1, self.vocab_size + self.unk_num))
        return res
# tokenizer = GloveTokenizer(vocab["token2idx"])

def load_glove_weight(d:int):
    p = re.compile(r"\s")
    path = os.path.join("pretrained", "glove27b", "glove.twitter.27B." + str(d) + "d.txt")
    with open(path, "r") as r:
        file = r.readlines()
    n = len(file)
    weight = np.zeros((n, d), dtype=np.float32)
    for i, line in enumerate(tqdm(file)):
        values = p.split(line.strip())
        if len(values) == d:
            weight[i] = np.asarray(values, dtype=np.float32)
        else:
            weight[i] = np.asarray(values[1:], dtype=np.float32)
    return weight

def get_mvsa_glove_weight(dir, d:int, _uniform:float=0.1):
    path = os.path.join(dir, "glove27b" + str(d) + "d.npy")
    if os.path.exists(path):
        return np.load(path)
    glove_weight = load_glove_weight(d)
    vocab = load_glove_vocab()
    n = len(vocab["token2idx"]) 
    weight = np.zeros((n + UNK_NUM + 1, d), dtype=np.float32) 
    weight[1:n+1] = glove_weight[vocab["glove_idx"]] # 正文
    glove_size = len(glove_weight)
    for i in range(UNK_NUM):
        temp_weight = glove_weight[random.sample(list(range(glove_size)), 100000)]
        weight[n + i + 1] = temp_weight.mean(axis=0) # UNK
    np.save(path, weight) # all 5.4mb
    return weight

def load_mvsa_glove_data():
    with open(DIR + "glove_data.pickle", "rb") as r:
        return pickle.load(r)

def load_glove_data(config:dict):
    vocab = load_glove_vocab()
    config["vocab_size"] = len(vocab["token2idx"])
    embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(get_mvsa_glove_weight(DIR, config["embedding_dim"])))
    config["embedding"] = embedding
    config["output_size"] = CATEGORY_NUM

    with open(DIR + "glove_data.pickle", "rb") as r:
        data = pickle.load(r)
    train_set = MultiModalDataset.from_mvsa(data["train"], config)
    valid_set = MultiModalDataset.from_mvsa(data["valid"], config)
    test_set = MultiModalDataset.from_mvsa(data["test"], config)
    return train_set, valid_set, test_set


def read_vgg_feature(_id):
    path = DIR + "vgg16/" + _id + ".npy"
    return np.load(path)

def read_rcnn_feature(_id):
    path = DIR + "rcnn/" + _id + ".npz"
    return np.load(path)

features = {}
def load_vgg(_id):
    if _id not in features:
        features[_id] = read_vgg_feature(_id)
    return features[_id]

def load_rcnn(_id):
    if _id not in features:
        features[_id] = read_rcnn_feature(_id)
    return features[_id]