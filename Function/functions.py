#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch.autograd import Function
import torch
# import torch.nn as nn
# import numpy as np
import csv
import random
from Function import param

def try_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e


class ReverseLayerF(Function):
    
    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.alpha = alpha
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output*ctx.alpha
        return grad_input, None

# CSV出力
def write_csv(file, save_dict):
    save_row = {}

    with open(file,'w') as f:
        writer = csv.DictWriter(f, fieldnames=save_dict.keys(),delimiter=",",quotechar='"')
        writer.writeheader()

        k1 = list(save_dict.keys())[0]
        length = len(save_dict[k1])

        for i in range(length):
            for k, vs in save_dict.items():
                save_row[k] = vs[i]

            writer.writerow(save_row)
    

# 映画のクロスバリデーション選択
def cv_random(seed=0):
    L_list = []
    label_0 = [2,3,6,11,14] # Negative data
    label_1 = [1,4,7,10,12] # Neutral data
    label_2 = [0,5,8,9,13] # Positive data
    # random.seed(seed)
    # random.shuffle(label_0)
    # random.shuffle(label_1)
    # random.shuffle(label_2)
    L_list.append(label_0)
    L_list.append(label_1)
    L_list.append(label_2)
    return L_list

def mk_list():
    L_list = []
    label_0 = [2,3,6,11,14] # Negative data
    label_1 = [1,4,7,10,12] # Neutral data
    label_2 = [0,5,8,9,13] # Positive data
    L_list.append(label_0)
    L_list.append(label_1)
    L_list.append(label_2)
    return L_list    