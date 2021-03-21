import sys
sys.path.append("/home/ly/workspace/mmsa")
import os
import pickle
import numpy as np
from typing import *
from tqdm import tqdm
from collections import OrderedDict
from utils.load_yelp import load_data

base_dir = os.path.join("data","yelp-vistanet")

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

def build_rcnn_data(reviews:List[dict]):
    rcnn_data = []
    total_img = 0
    for review in tqdm(reviews):
        for _id in review["Photos"]:
            features = load_rcnn_features(_id)
            if features is not None:
                rcnn_data.append((_id, features)) # key, val
                total_img += 1
    print(f"Image num : {total_img}")
    return rcnn_data

if __name__ == "__main__":
    data = load_data()
    k = 10000
    for _key in ["train", "valid", "test"]:
        rcnn_data = build_rcnn_data(data[_key])
        i = 0
        n = len(rcnn_data)
        while True:
            bound = min((i + 1) * k, n) # 找到右边界
            path = os.path.join(base_dir, "rcnn_" + _key + str(i) + ".pickle") # 按序号划分
            with open(path, "wb") as w:
                pickle.dump(rcnn_data[i*k:bound], # 划分几次
                 w, protocol=pickle.HIGHEST_PROTOCOL)
            if bound == n: # 最后一个
                break
            i += 1

