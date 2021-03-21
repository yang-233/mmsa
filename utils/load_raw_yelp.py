import pickle
import os
import numpy as np

from tqdm import tqdm

import sys
sys.path.append("/home/ly/workspace/mmsa")
from utils.dataset import *

from .load_yelp import *
from .train import predict, evalute

DATA_DIR = "data/yelp-vistanet/"
train_json = DATA_DIR + "raw/train.json"
valid_json = DATA_DIR + "raw/valid.json"
cities = ["Boston", "Chicago", "Los Angeles", "New York", "San Francisco"]

def check_photo(_id:str):
    path = os.path.join(DATA_DIR, "photos", _id[:2], _id + ".jpg")
    return os.path.exists(path)

def read_reviews(file_path:str, clean_data:bool=False) -> List[Dict[str, str]]: 
    # 读入数据
    reviews = None
    if file_path.endswith(".json"):
         with open(file_path, 'r', encoding="utf-8") as f:
            reviews = []
            for line in tqdm(f, "Read json"):
                review = json.loads(line)
                imgs = []
                captions = []
                for photo in review['Photos']:
                    _id = photo['_id']
                    caption = photo["Caption"]
                    if clean_data:
                        if check_photo(_id):
                            imgs.append(_id)
                            captions.append(caption)
                    else:
                        imgs.append(_id)
                        captions.append(caption)
                        
                reviews.append({'_id': review['_id'],
                      'Text': review['Text'],
                      'Photos': imgs,
                      'Captions': captions,
                      'Rating': review['Rating']})
    elif file_path.endswith(".pickle"):
        with open(file_path, 'rb') as f:
            reviews = pickle.load(f) # 直接从pickle中加载
    else:
        raise RuntimeError("Illegal file path!")
    return reviews

def load_clean_data():
    data = None
    with open(DATA_DIR + "raw/" + "clean_data.pickle", "rb") as r:
        data = pickle.load(r)
    return data

def _load_glove_data():
    with open(DATA_DIR + "raw/" + "glove_data.pickle", "rb") as r:
        return pickle.load(r)

def load_glove_data(config):
    embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(
        get_yelp_glove_weight(DATA_DIR + "raw/", config["embedding_dim"])))
    config["embedding"] = embedding
    config["output_size"] = CATEGORY_NUM
    data = _load_glove_data()
    train_set = MultiModalDataset.from_yelp(data["train"], config)
    valid_set = MultiModalDataset.from_yelp(data["valid"], config)
    test_sets = {}
    for key, val in data["test"].items():
        test_sets[key] = MultiModalDataset.from_yelp(data["test"][key], config)
    return train_set, valid_set, test_sets

def get_yelp_raw_loader(batch_size:int, workers:int, collate_fn:Callable, train_set, valid_set, test_set):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn)
    for key, dataset in test_set.items():
        test_set[key] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn)
    return train_loader, valid_loader, test_set

def eval_model_on_raw_yelp(model, loss, test_loaders, use_cuda=True):
    all_true = []
    all_pred = []
    total_loss = 0.
    res = {}
    for key, loader in test_loaders.items():
        y_true, y_pred, _loss= predict(model, loader, loss, use_cuda)
        acc, f1 = evalute(y_true, y_pred)
        res[key] = {"Accuracy" : acc, "F1" : f1, "Loss" : _loss}
        all_true += y_true
        all_pred += y_pred
        total_loss += _loss * len(y_pred)
    
    acc, f1 = evalute(all_true, all_pred)
    total_loss /= len(all_pred)
    res["Total"] = {"Accuracy" : acc, "F1" : f1, "Loss" : total_loss}
    return res
    
if __name__ == "__main__":
    train = read_reviews(train_json, True)
    valid = read_reviews(valid_json, True)
    test = {}
    for city in cities:
        test[city] = read_reviews(DATA_DIR + "raw/test/" + city +
         "_test.json", True)
    data = {"train" : train, "valid" : valid, "test" : test}
    with open(DATA_DIR + "raw/" + "clean_data.pickle", "wb") as o:
        pickle.dump(data, o, protocol=pickle.HIGHEST_PROTOCOL)

    
    freq_dict = count_word_freq(train)
    freq_dict = count_word_freq(valid, freq_dict)
    token2idx, idx2token, glove_idx = build_vocab_from_glove(freq_dict)
    save_vocab(DATA_DIR + "raw/", token2idx, idx2token, glove_idx)
    glove_vocab = load_glove_vocab(DATA_DIR + "raw/")
    glove_tokenizer = YelpSimpleTokenizer(glove_vocab["token2idx"], True)
    
    for key in ["train", "valid"]:
        for i in data[key]:
            i["Text"] = glove_tokenizer.to_idx(i["Text"])
    for city in cities:
        for i in data["test"][city]:
            i["Text"] = glove_tokenizer.to_idx(i["Text"])
    with open(DATA_DIR + "raw/" + "glove_data.pickle", "wb") as o:
        pickle.dump(data, o, protocol=pickle.HIGHEST_PROTOCOL)