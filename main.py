import sys
sys.path.append("/home/ly/workspace/mmsa")
seed = 1938
import numpy as np
import torch
from torch import nn
from torch import optim

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
from models.bigru_rcnn_gate import *
from utils.train import *
from typing import *
from utils.load_raw_yelp import *
from utils.dataset import *
from utils.train import *
from utils.train import *

def main(): 

    train_set, valid_set, test_set = load_glove_data(config)
    
    batch_size = 2
    workers = 2
    train_loader, valid_loader, test_loader = get_loader(batch_size, workers, get_collate_fn(config), 
                        train_set, valid_set, test_set)

    model = Model(config)
    #X, y = iter(valid_loader).next()
    #res = model(X)
    loss = nn.CrossEntropyLoss()
    # get_parameter_number(model), loss

    viz = get_Visdom()
    lr = 1e-3
    epoches = 20
    optimizer = get_regal_optimizer(model, optim.AdamW, lr)
    k_batch_train_visdom(model, optimizer, loss, valid_loader, viz, 30, 10, use_cuda=False)

if __name__ == "__main__":
    # torch.cuda.set_device(1)

    main()