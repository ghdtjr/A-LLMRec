import contextlib
import logging
import os
import glob

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from utils import *
from pre_train.sasrec.model import SASRec


def load_checkpoint(recsys, pre_trained):
    path = f'pre_train/{recsys}/{pre_trained}/'
    
    pth_file_path = find_filepath(path, '.pth')
    assert len(pth_file_path) == 1, 'There are more than two models in this dir.\n'
    kwargs, checkpoint = torch.load(pth_file_path[0], map_location="cpu")
    logging.info("load checkpoint from %s" % pth_file_path[0])

    return kwargs, checkpoint

class RecSys(nn.Module):
    def __init__(self, recsys_model, pre_trained_data, device, option=None):
        super().__init__()
        kwargs, checkpoint = load_checkpoint(recsys_model, pre_trained_data)
        kwargs['args'].device = device
        model = SASRec(**kwargs)
        model.load_state_dict(checkpoint)
            
        for p in model.parameters():
            p.requires_grad = False
            
        self.item_num = model.item_num
        self.user_num = model.user_num
        self.model = model.to(device)
        self.hidden_units = kwargs['args'].hidden_units
        
    def forward():
        print('forward')

    def show_n_params(self, return_str=True):
        tot = 0
        for p in self.parameters():
            w = 1
            for x in p.shape:
                w *= x
            tot += w
        if return_str:
            if tot >= 1e6:
                return "{:.1f}M".format(tot / 1e6)
            else:
                return "{:.1f}K".format(tot / 1e3)
        else:
            return tot

