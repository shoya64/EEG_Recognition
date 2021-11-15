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
from Function.model import FeatureExtractor, Classifier, Discriminator
from Function.functions import try_gpu
from Function.mk_data_exp1 import mk_train, mk_test, mk_val
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
    for CV_num in range(5):
        final_average = []
        count = 0
        for train_index, test_index in kf.split(EEG):
            count += 1 #実験回数
            if count >= 0:
                if count <= 9:
                    num = "0"+str(count)
                else:
                    num = str(count)

                load_model = load_direct+"data_num_"+str(1)+"/CV_num"+str(CV_num+1)+"/model/sub_"+str(count)+"/"

                # print('############ 実験番号: '+str(count) + ' #############')
                # train,target,testデータの生成
                xtrain, ytrain, dtrain = mk_train(EEG, label, train_index)
                xtarget, ytarget, dtarget, X_test, Y_test = mk_val(EEG, label, test_index, L_list, num = CV_num)
               
                X_train = try_gpu(torch.from_numpy(xtrain)).float()
                Y_train = try_gpu(torch.from_numpy(ytrain)).long()
                D_train = try_gpu(torch.from_numpy(dtrain)).long()
                X_target = try_gpu(torch.from_numpy(xtarget)).float()
                Y_target = try_gpu(torch.from_numpy(ytarget)).long()
                D_target = try_gpu(torch.from_numpy(dtarget)).long()
                X_test = try_gpu(X_test).float()
                Y_test = try_gpu(Y_test).long()
            
                # #各モデルの読み込み
                FE_path = load_model+"FE"+ str(count) + ".pth"
                C_path = load_model+"C"+str(count)+".pth"
                D_path = load_model+"D"+str(count)+".pth"
                model_F = FeatureExtractor(param.lstm_input_dim, param.lstm_hidden_dim, param.time_steps, param.K)
                model_C = Classifier(param.lstm_hidden_dim, param.K, param.class_num)
                model_D = Discriminator(param.lstm_hidden_dim, param.K, param.dense_hidden_dim, param.domain_num2)
                model_F.load_state_dict(torch.load(FE_path))
                model_C.load_state_dict(torch.load(C_path))
                model_D.load_state_dict(torch.load(D_path))
                model_F = try_gpu(model_F)
                model_C = try_gpu(model_C)
                model_D = try_gpu(model_D)
                
                feature_output_source = model_F(X_train)
                feature_output_target = model_F(X_target)
                feature_output_test = model_F(X_test)

                X0, y0, d0 = feature_output_source.cpu().detach().numpy(), Y_train.cpu().detach().numpy(), D_train.cpu().detach().numpy()
                X1, y1, d1 = feature_output_target.cpu().detach().numpy(), Y_target.cpu().detach().numpy(), D_target.cpu().detach().numpy()
                X2, y2 = feature_output_test.cpu().detach().numpy(), Y_test.cpu().detach().numpy()

                # UMAPで可視化
                mapper = umap.UMAP(random_state=42, n_neighbors = 10, n_components = 2).fit(X0)
                embedding0 = mapper.transform(X0)
                embedding1 = mapper.transform(X1)
                embedding2 = mapper.transform(X2)

                ########## ラベル定義  #############
                Emotion_labels = ["Negative", "Neutral", "Positive"]
                Domain_labels = ["Source", "Target", "Test"]
                
                ########## embedとｙを格納 ##########
                embed_dict = {"Source": embedding0, "Target": embedding1, "Test": embedding2}
                y_dict = {"Source": y0, "Target": y1, "Test": y2}

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
                os.makedirs(save_direct+'data/', exist_ok=True)
                np.save(save_direct+'data/'+'X0_'+str(CV_num+1)+'_'+str(count), embedding0)
                np.save(save_direct+'data/'+'X1_'+str(CV_num+1)+'_'+str(count), embedding1)
                np.save(save_direct+'data/'+'X2_'+str(CV_num+1)+'_'+str(count), embedding2)
                np.save(save_direct+'data/'+'y0_'+str(CV_num+1)+'_'+str(count), y0)
                np.save(save_direct+'data/'+'y1_'+str(CV_num+1)+'_'+str(count), y1)
                np.save(save_direct+'data/'+'y2_'+str(CV_num+1)+'_'+str(count), y2)

                plt.xlabel("Embedding Dimension 1", fontsize=18), plt.ylabel("Embedding Dimention 2", fontsize=18), plt.tick_params(labelsize=18)
                plt.xlim([-20,20])
                plt.ylim([-20,20])
                legend = plt.legend(handles1, Emotion_labels, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=20, handletextpad=0.01)
                legend2 = plt.legend(handles2, Domain_labels, bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0, fontsize=20, handletextpad=0.01)
                plt.gca().add_artist(legend)
                plt.savefig(save_direct+'Proposed_'+str(CV_num+1)+'_'+str(count)+'.png', bbox_extra_artists=(legend,legend2), bbox_inches="tight", pad_inches=0)
                plt.close()

