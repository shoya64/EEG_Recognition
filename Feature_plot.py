#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import csv
import umap.umap_ as umap

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold

from Function.data_load import mkdata
from Function.model import FeatureExtractor, Classifier, Discriminator
from Function.functions import try_gpu
from Function.mk_data import mk_train, mk_test_plt
from Function import param

import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt

# 実験番号
exp = str(3)
# 被験者ID
sub = str(7)

# 描画サイズを設定
max_size = 18
min_size = -9

########## ラベル定義  #############
Emotion_labels = ["Negative", "Neutral", "Positive"]
Domain_labels = ["Source", "Target", "Test"]

save_direct = 'FeaturePlot/save_tri_re/'
os.makedirs(save_direct, exist_ok=True)
load_direct = 'FeaturePlot/save_repre/data/'
number = exp+'_'+sub+'.npy'
embedding0 = np.load(load_direct+'X0_'+number)
embedding1 = np.load(load_direct+'X1_'+number)
embedding2 = np.load(load_direct+'X2_'+number)
y0 = np.load(load_direct+'y0_'+number)
y1 = np.load(load_direct+'y1_'+number)
y2 = np.load(load_direct+'y2_'+number)

########## embedとｙを格納 ##########
embed_dict = {"Source": embedding0, "Target": embedding1, "Test": embedding2}
y_dict = {"Source": y0, "Target": y1, "Test": y2}
# Emotion_labels = []
# Domain_labels = []
########## ラベル関連定数  ##########
__colors_dict = {"Source": "red", "Target": "blue", "Test": "green"}
__shapes_dict = {"Negative": "v", "Neutral": "^", "Positive": "o"}
__label_dict = {"Negative": 0, "Neutral": 1, "Positive": 2}


####################################
size = 40
plt.figure(figsize=(7, 7))
#############  感情の凡例づくり  ###############
handles1=[]
for class_name, shapes in __shapes_dict.items():
    space_0 = embed_dict["Target"]
    space_1 = space_0[y_dict["Target"] == __label_dict[class_name]]
    p=plt.scatter(x=space_1[:,0], y=space_1[:,1], alpha=0.9, 
                c="black", marker=shapes)
    handles1.append(p)
    p.remove()

#############  ドメインの凡例づくり  #############
handles2=[]
for domain_name, embed in embed_dict.items():
    space_2 = embed[y_dict[domain_name]==1] #感情ラベルはなんでもよい
    p=plt.scatter(x=space_2[:,0], y=space_2[:,1], alpha=0.9, 
                c=__colors_dict[domain_name], marker=",")
    handles2.append(p)
    p.remove()

#############  実際のプロット  ################
for domain_name, embed in embed_dict.items():
    for class_name, shapes in __shapes_dict.items():
        emospace = embed[y_dict[domain_name] == __label_dict[class_name]]
        plt.scatter(x=emospace[:, 0], y=emospace[:,1], alpha=0.9, c=__colors_dict[domain_name],
                    marker=shapes, s=size, linewidths=1, edgecolors="black")

plt.xlabel("Embedding Dimension 1", fontsize=18), plt.ylabel("Embedding Dimension 2", fontsize=18), plt.tick_params(labelsize=18)
plt.xlim([min_size,max_size])
plt.ylim([min_size,max_size])
legend = plt.legend(handles1, Emotion_labels, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=20, handletextpad=0.01)
legend2 = plt.legend(handles2, Domain_labels, bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0, fontsize=20, handletextpad=0.01)
plt.gca().add_artist(legend)
plt.savefig(save_direct+'Proposed_'+exp+'_'+sub+'.jpeg', bbox_extra_artists=(legend,legend2), bbox_inches="tight", pad_inches=0)
plt.close()

