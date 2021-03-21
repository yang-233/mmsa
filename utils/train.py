import os
import re
import subprocess
import signal
import torch
import time

from collections import OrderedDict
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from typing import *
from sklearn import metrics 
from copy import deepcopy
from tqdm import tqdm
from visdom import Visdom
from math import isnan
try:
    from torch.utils import SummaryWriter
except Exception as e:
    pass
    # from tensorboardX import SummaryWriter

_tensorboard_process = None

def rm(path):
    if os.path.isfile(path): # 删除文件
        os.remove(path)
    else:
        for i in os.listdir(path): # 递归删除
            rm(path + "/" + i)
        os.rmdir(path)

def get_tensorboard_id() -> int:
    with open(".tensorboard_id", "r") as r:
        return int(r.read())

def save_tensorboard_id(id:int):
    with open(".tensorboard_id", "w") as o:
        o.write(str(id))

def get_writer(path="./log"): #　正常情况下在log目录下
    for i in os.listdir(path): # 清空该目录
        rm(path + "/" + i)
    global _tensorboard_process
    if _tensorboard_process is None:
        try:  # 杀掉旧的tensorboard并开启一个新的
            tensorboard_id = get_tensorboard_id()
            p = psutil.Process(tensorboard_id)
            if p.name() == "tensorboard":
                os.kill(tensorboard_id, signal.SIGKILL)
        except Exception as e:
            pass
    else:
        _tensorboard_process.kill() # 自杀掉
    
    _tensorboard_process = subprocess.Popen(["/home/ly/miniconda3/bin/tensorboard", "--logdir", path,
                                        "--bind_all", "--reload_interval", "1"], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # 
    save_tensorboard_id(_tensorboard_process.pid) # 保存pid
    return SummaryWriter(path, flush_secs=1)

class VisdomScalar(object):
    def __init__(self, viz:Visdom, title:str, line_num:int=1):
        self.viz = viz
        self.title = title
        self.win = None
        self.num = line_num
        self.step = 1
        self.names = []
        for i in range(line_num):
            self.names.append("line" + str(i + 1))
    def draw(self, *y):
        assert len(y) == self.num, f"expect {self.num} values, but got {len(y)} "
        x = [self.step]
        for i, yi in enumerate(y):
            if self.win is None:
                self.win = self.viz.line([yi], x, name=self.names[i],
                                        opts={'title' : self.title})
            else:
                self.viz.line([yi], x, name=self.names[i], update="append",
                              win=self.win)
        self.step += 1
class VisdomTextWriter(object):
    def __init__(self, viz:Visdom, title:str):
        self.viz = viz
        self.title = title
        self.win = None

    def write(self, text):
        if self.win is None:
            self.win = self.viz.text(text, opts={'title' : self.title})
        else:
            self.viz.text(text, win=self.win, opts={'title' : self.title}, append=True)

class WeightViewer(object):
    def __init__(self, viz:Visdom, model:nn.Module, names:List[str]):
        self.viz = viz
        self.params = model.state_dict()
        self.named_win = OrderedDict()
        self.step = 1
        for name in names:
            self.named_win[name] = None

    def _get_param(self, name):
        return self.params[name].view(-1)
    
    def draw(self):
        for name in self.named_win.keys():
            if self.named_win[name] is not None: # 先关闭原来的
                self.viz.close(win=self.named_win[name])
            self.named_win[name] = self.viz.histogram(self._get_param(name), 
                opts={'title' : f"No.{self.step} step: {name}\'s weight"})
        self.step += 1
                
def get_time():
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time()))

def get_Visdom():
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    os.environ["ftp_proxy"] = ""
    return Visdom()

def to_cuda(var):
    if isinstance(var, torch.Tensor):
        return var.cuda()
    else:
        return tuple(to_cuda(i) for i in var) # 递归下去

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def predict(model:nn.Module, dataloader:DataLoader, loss:Callable=None, use_cuda:bool=True):
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        _loss = 0.
        for i, (X, y) in enumerate(dataloader):
            if use_cuda:
                X, y = to_cuda((X, y))
            # print(f"No. {i} batch")
            res = model(X)
            if loss is not None: # 如果需要计算损失
                if isinstance(loss, nn.MSELoss): # 如果是均方误差
                    l = loss(res, y.type_as(res)) # 转换类型
                else:
                    l = loss(res, y)
                _loss += l.item() * len(y)
            
            y_pred += model.predict(res)
            y_true += y.cpu().numpy().ravel().tolist() # 转为list
            if isnan(_loss):
                print(f"No {i} batch _loss became nan")
                break
    return y_true, y_pred, _loss / len(y_true)

def evalute(y_true:List[int], y_pred:List[int], f1_average:str="weighted") -> Tuple[float, float]:
    return metrics.accuracy_score(y_true, y_pred), metrics.f1_score(y_true, y_pred, average=f1_average)

def eval_model(model:nn.Module, dataloader:DataLoader, loss:Callable=None,use_cuda:bool=True) -> Tuple[float, float, float]:
    y_true, y_pred, _loss = predict(model, dataloader, loss, use_cuda)
    return evalute(y_true, y_pred), _loss

def get_regal_optimizer(model:nn.Module, optimizer:Callable, _lr:float, _weight_decay:float=0.0, **kw):
    no_decay_list = []
    weight_decay_list = []
    for name, param in model.named_parameters():
        if param.requires_grad == False:
            continue
        if "bias" in name or "bn" in name: # bias项和batch norm项不需要正则化
            no_decay_list.append(param)
        else:
            weight_decay_list.append(param)
    params = [{"params" : weight_decay_list}, {"params" : no_decay_list, "weight_decay" : 0.0}]
    return optimizer(params, lr=_lr, weight_decay=_weight_decay, *kw)


def train_visdom(model:nn.Module, optimizer:torch.optim.Optimizer, loss:Callable, viz:Visdom,
          train_dataloader:DataLoader, valid_dataloader:DataLoader, num_epoches:int, 
          batch_loss:List[float], batch_loss_drawer:VisdomScalar,
          train_loss:List[float], valid_loss:List[float], epoch_loss_drawer:VisdomScalar,
          train_acc:List[float], valid_acc:List[float], acc_drawer:VisdomScalar, text_writer:VisdomTextWriter,
          _interval:int=10, use_cuda:bool=True, early_stop:int=5, 
          grad_max_norm:float=-1, debug=False, weight_viewer:WeightViewer=None) -> str:

    result = {"max_acc" : 0, "max_acc_epoch": -1, "max_train_acc" : 0,
           "max_acc_train_loss" : -1, "max_acc_valid_loss" : -1,
           "last_acc" : 0, "last_train_acc" : 0, "last_epoch": -1,
           "last_train_loss" : -1, "last_valid_loss" : -1}
    best_model = None
    if debug:
        params = model.state_dict()
    for i in range(num_epoches):
        #train
        model.train()
        _train_loss = 0.
        y_true = []
        y_pred = []
        for j, (X, y) in tqdm(enumerate(train_dataloader), f"No {i + 1} epoch"):
            y_true += y.numpy().ravel().tolist()
            if use_cuda:
                X, y = to_cuda((X, y))
            res = model(X)
            y_pred += model.predict(res)
            l = loss(res, y)
            _l = l.item() 
            if isnan(_l):
                print(f"No {i + 1} epoch No {j + 1} batch _l became nan")
                return
            _train_loss += _l * len(y) # 求总loss
            batch_loss.append(_l)
            if len(batch_loss) % _interval == 1:
                batch_loss_drawer.draw(_l)
            optimizer.zero_grad()
            l.backward()
            if debug:
                if weight_viewer is not None:
                    weight_viewer.draw()
            if grad_max_norm > 0:
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=grad_max_norm, norm_type=2)
            optimizer.step()
        _train_loss /= len(y_true) # 求一个回合的均值loss
        (_valid_acc, _f1), _valid_loss = eval_model(model, valid_dataloader, loss, use_cuda)
        _train_acc = metrics.accuracy_score(y_true, y_pred) # 

        train_loss.append(_train_loss)
        valid_loss.append(_valid_loss)
        train_acc.append(_train_acc)
        valid_acc.append(_valid_acc)
        
        epoch_loss_drawer.draw(_train_loss, _valid_loss)
        acc_drawer.draw(_train_acc, _valid_acc)
        text_writer.write(f"No {i + 1} epoch: train_loss:{_train_loss} train_acc:{_train_acc} valid_loss:{_valid_loss} valid_acc:{_valid_acc}")
        result["last_acc"] = _valid_acc
        result["last_train_acc"] = _train_acc
        result["last_epoch"] = i + 1
        result["last_train_loss"] = _train_loss
        result["last_valid_loss"] = _valid_loss

        if _valid_acc > result["max_acc"]: # 结果更优
            result["max_acc"] = _valid_acc
            result["max_train_acc"] = _train_acc
            result["max_acc_epoch"] = i + 1
            result["max_acc_train_loss"] = _train_loss
            result["max_acc_valid_loss"] = _valid_loss
            best_model = deepcopy(model)
        if i + 1 - result["max_acc_epoch"] >= early_stop: # 很久没有提升了
            return result, best_model
    return result, best_model

def train_visdom_v2(model:nn.Module, optimizer:torch.optim.Optimizer, loss:Callable, viz:Visdom,
          train_dataloader:DataLoader, valid_dataloader:DataLoader, num_epoches:int, 
          batch_loss:List[float], batch_loss_drawer:VisdomScalar,
          train_loss:List[float], valid_loss:List[float], epoch_loss_drawer:VisdomScalar,
          train_acc:List[float], valid_acc:List[float], acc_drawer:VisdomScalar, text_writer:VisdomTextWriter,
          _interval:int=10, use_cuda:bool=True, early_stop:int=5, 
          grad_max_norm:float=-1, debug=False, weight_viewer:WeightViewer=None) -> str:

    result = {"min_valid_loss" : float("inf"), "min_valid_loss_epoch": -1, "min_loss_train_acc" : 0,
           "min_valid_loss_train_loss" : -1, "min_loss_valid_acc" : 0,
           "last_valid_acc" : 0, "last_train_acc" : 0, "last_epoch": -1,
           "last_train_loss" : -1, "last_valid_loss" : -1}
    best_model = None
    if debug:
        params = model.state_dict()
    for i in range(num_epoches):
        #train
        model.train()
        _train_loss = 0.
        y_true = []
        y_pred = []
        for j, (X, y) in tqdm(enumerate(train_dataloader), f"No {i + 1} epoch"):
            y_true += y.numpy().ravel().tolist()
            if use_cuda:
                X, y = to_cuda((X, y))
            res = model(X)
            y_pred += model.predict(res)
            l = loss(res, y)
            _l = l.item() 
            if isnan(_l):
                print(f"No {i + 1} epoch No {j + 1} batch _l became nan")
                return
            _train_loss += _l * len(y) # 求总loss
            batch_loss.append(_l)
            if len(batch_loss) % _interval == 1:
                batch_loss_drawer.draw(_l)
            optimizer.zero_grad()
            l.backward()
            if debug:
                if weight_viewer is not None:
                    weight_viewer.draw()
            if grad_max_norm > 0:
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=grad_max_norm, norm_type=2)
            optimizer.step()
        _train_loss /= len(y_true) # 求一个回合的均值loss
        (_valid_acc, _f1), _valid_loss = eval_model(model, valid_dataloader, loss, use_cuda)
        _train_acc = metrics.accuracy_score(y_true, y_pred) # 
        train_loss.append(_train_loss)
        valid_loss.append(_valid_loss)
        train_acc.append(_train_acc)
        valid_acc.append(_valid_acc)

        epoch_loss_drawer.draw(_train_loss, _valid_loss)
        acc_drawer.draw(_train_acc, _valid_acc)
        text_writer.write(f"No {i + 1} epoch: train_loss:{_train_loss} train_acc:{_train_acc} valid_loss:{_valid_loss} valid_acc:{_valid_acc}")

        result["last_valid_acc"] = _valid_acc
        result["last_train_acc"] = _train_acc
        result["last_epoch"] = i + 1
        result["last_train_loss"] = _train_loss
        result["last_valid_loss"] = _valid_loss

        if _valid_loss < result["min_valid_loss"]: # 结果更优
            result["min_valid_loss"] = _valid_loss
            result["min_valid_loss_epoch"] = i + 1
            result["min_loss_valid_acc"] = _valid_acc
            result["min_loss_train_acc"] = _train_acc
            result["min_valid_loss_train_loss"] = _train_loss
            best_model = deepcopy(model)
        
        if i + 1 - result["min_valid_loss_epoch"] >= early_stop: # 很久没有提升了
            return result, best_model
    return result, best_model

def k_batch_train_visdom(model:nn.Module, optimizer:optim.Optimizer, loss:Callable,
                valid_dataloader:DataLoader, viz:Visdom, num_epoches:int, k:int, 
                use_cuda:bool=True):
    _loss = []
    acc = []
    loss_drawer = VisdomScalar(viz, f"Loss")
    acc_drawer = VisdomScalar(viz, f"Accuracy")

    model.train()
    y_true = []
    k_batch = []
    it = iter(valid_dataloader)
        
    for i in range(k): # 取出全部k个batch
        X, y = next(it)
        y_true += y.numpy().ravel().tolist() # 转为list
        if use_cuda:
            X, y = to_cuda((X, y))
        k_batch.append((X, y))
    for i in tqdm(range(num_epoches), f"No {i + 1} epoch"):
        y_pred = []
        _l = 0.
        for X, y in k_batch:
            res = model(X)
            l = loss(res, y)
            _l += l.item() # 求总loss
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            y_pred += model.predict(res)
        _acc = metrics.accuracy_score(y_true, y_pred)
        _loss.append(_l)
        acc.append(_acc)
        loss_drawer.draw(_loss[-1])
        acc_drawer.draw(acc[-1])