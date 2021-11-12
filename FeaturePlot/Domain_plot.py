#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import os
import sys
import csv
import umap.umap_ as umap

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold

from Function.data_load import mkdata
from Function.functions import try_gpu
from Function.mk_data import mk_domain_plt
from Function import param

import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt

def UMAP_plot(EEG, label, kf, L_list, load_direct, save_direct):
    seed=42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    label_0 = [2,3,6,11,14] # Negative data
    label_1 = [1,4,7,10,12] # Neutral data
    label_2 = [0,5,8,9,13] # Positive data

    for i in range(15):
        movie = [i]

        X, y, domain = mk_domain_plt(EEG, label, movie)

        # UMAPで可視化
        mapper = umap.UMAP(random_state=42, n_neighbors = 10, n_components = 2).fit(X)
        embedding0 = mapper.transform(X)

        ##############色の定義###############
        # 同系15色
        # color_map = ["#cc2828","#bbcc28","#28cc49","#289bcc","#6a28cc","#cc6428","#7fcc28","#28cc85",
        #                 "#285fcc","#a528cc","#cca028","#44cc28","#28ccc1","#2e28cc","#cc28b6"]
        # color_map = ["#ff0000","#ccff00","#00ff65","#0065ff","#cb00ff","#ff6600","#65ff00","#00ffcb",
        #                 "#0000ff","#ff00cb","#ffcc00","#00ff00","#00cbff","#6600ff","#ff0066"]

        # みためで色分け
        color_map = ["black","grey","darkred","red","orangered","orange","gold","olive","lawngreen","darkgreen",
                        "aqua","dodgerblue","blue","darkviolet","magenta"]
        ####################################
        size = 40
        plt.figure(figsize=(7, 7))

        embedding_x = embedding0[:,0]
        embedding_y = embedding0[:,1]
      
        for n in np.unique(domain):
            plt.scatter(embedding_x[domain == n],embedding_y[domain == n],label=n,c=color_map[n])
        plt.title("Movie "+str(i+1), fontsize=20)
        plt.xlim([-20,25])
        plt.ylim([-25,30])
        plt.xlabel("Embedding Dimension 1", fontsize=18), plt.ylabel("Embedding Dimention 2", fontsize=18), plt.tick_params(labelsize=18)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=20, handletextpad=0.01)
        plt.savefig(save_direct+'Movie_'+str(i)+'.png', bbox_inches="tight", pad_inches=0)
        plt.close()

