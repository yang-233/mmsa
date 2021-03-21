import sys
sys.path.append("/home/ly/workspace/mmsa")
seed = 1323
import numpy as np
import torch
from torch import nn
from torch import optim

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
from models.vistanet import *
from utils.train import *
from typing import *
from collections import Counter

from utils.load_yelp import *
from utils.dataset import *
from utils.train import *

def main():
    config = default_config
    train_set, valid_set, test_set= load_glove_vgg_data(config)
    # batch_size = 128
    # workers = 6
    # train_loader, valid_loader, test_loader = get_loader(batch_size, workers, get_collate_fn(config), train_set, valid_set, test_set)
    model = Model(config)
    model = model.cuda()
    loss = nn.CrossEntropyLoss()
    print(get_parameter_number(model))
if __name__ == "__main__":
    main()
