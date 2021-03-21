import enum
import torch
from torch.utils.data import Dataset, DataLoader
from typing import *

class MultiModalItem(object):
    def __init__(self, text, imgs, y):
        self.text = text
        self.imgs = imgs
        self.y = y

class MultiModalDataset(Dataset):
    def __init__(self, data:List[MultiModalItem]):
        self.size = len(data)
        self.text = []
        self.imgs = []
        self.y = []
        for i in data:
            self.text.append(i.text)
            self.imgs.append(i.imgs)
            self.y.append(i.y)
        
    def __getitem__(self, index: int):
        return (self.text[index], self.imgs[index]), self.y[index]

    def __len__(self) -> int:
        return self.size
    
    @classmethod
    def from_yelp(cls, reviews, config):
        data = []
        if config["task"] == "CLS":
            for r in reviews:
                data.append(MultiModalItem(r["Text"], r["Photos"], r["Rating"] - 1)) # 排名需要-1
        elif config["task"] == "REG":
            for r in reviews:
                data.append(MultiModalItem(r["Text"], r["Photos"], (r["Rating"] - 1) / config["output_size"])) # 把排名转化为0-1区间的实数
        return cls(data)

    @classmethod
    def from_mvsa(cls, data, config=None):
        label2idx = {"negative" : 0, "neutral" : 1, "positive" : 2}
        _data = []
        for i in data:
            _data.append(MultiModalItem(i["text"], i["id"], label2idx[i["label"]]))

        return cls(_data)
    
    

def padding_text_1d(batch:List[List[int]], _max_lens=1e9, padding_val:int=0) -> Tuple[torch.LongTensor, torch.LongTensor]:
    lens = list(map(len, batch))
    max_lens = min(max(lens), _max_lens)
    for i, text in enumerate(batch):
        if lens[i] < max_lens:
            batch[i] += [padding_val] * (max_lens - lens[i])
        elif lens[i] > max_lens: # 截长
            batch[i] = batch[i][:max_lens]
            lens[i] = max_lens
    return torch.LongTensor(batch), torch.LongTensor(lens)

def padding_text_2d(batch:List[List[List[int]]], _max_tokens=1e9, _max_sents=1e9, padding_val:int=0) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    for i, sents in enumerate(batch):
        if len(sents) > _max_sents: # 截长
            batch[i] = sents[:_max_sents]
    sents_num = list(map(len, batch)) # 每个有多少个句子 (batch,)
    res = []
    for i in batch:
        res += i
    padded_text, lens = padding_text_1d(res, _max_tokens, padding_val)
    return padded_text, lens, torch.LongTensor(sents_num)


def get_loader(batch_size:int, workers:int, collate_fn:Callable, train_set, valid_set, test_set=None):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn)
    if test_set is not None:
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn)
        return train_loader, valid_loader, test_loader
    else:
        return train_loader, valid_loader
