
#AFLAC実行ファイル
import numpy as np
import os
import sys
import random
import csv

import umap.umap_ as umap

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

from Function.data_load import mkdata_lab as mkdata
from Function.DANN import FeatureExtractor, Classifier, Discriminator
from Function.functions import try_gpu
from Function.dataset import all_label_dataset as MyDataset
from Function.mk_data_2domain import mk_train, mk_test
from Function import param_2domain as param

import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt

# data load
EEG, label = mkdata()
# cross validation
kf = KFold(n_splits = 15, shuffle = False)
    
for exp in range(3):
    count = 0
    for train_index, test_index in kf.split(EEG):
        count += 1 #実験回数
        if count >= 0:
            if count <= 9:
                num = "0"+str(count)
            else:
                num = str(count)

            load_model = "exp/exp"+str(exp+1)+"/model/sub_"+str(count)+"/"
            save_direct = "IEEJ_Plot_noUMAP/"
            os.makedirs(save_direct, exist_ok=True)


            print('############ 実験番号: '+str(count) + ' #############')
            # train,target,testデータの生成
            xtrain, ytrain, dtrain = mk_train(EEG, label, train_index)
            xtarget, ytarget, dtarget, X_test, Y_test = mk_test(EEG, label, test_index, mult=1)


            X_train = try_gpu(torch.from_numpy(xtrain)).float()
            Y_train = try_gpu(torch.from_numpy(ytrain)).long()
            D_train = try_gpu(torch.from_numpy(dtrain)).long()
            X_target = try_gpu(torch.from_numpy(xtarget)).float()
            Y_target = try_gpu(torch.from_numpy(ytarget)).long()
            D_target = try_gpu(torch.from_numpy(dtarget)).long()
            X_test = try_gpu(X_test).float()
            Y_test = try_gpu(Y_test).long()
            # D_test = try_gpu(torch.from_numpy(D_))
        
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
            

            _, feature_output_source = model_F(X_train)
            _, feature_output_target = model_F(X_target)
            _, feature_output_test = model_F(X_test)

            X0, y0, d0 = feature_output_source.cpu().detach().numpy(), Y_train.cpu().detach().numpy(), D_train.cpu().detach().numpy()
            X1, y1, d1 = feature_output_target.cpu().detach().numpy(), Y_target.cpu().detach().numpy(), D_target.cpu().detach().numpy()
            X2, y2 = feature_output_test.cpu().detach().numpy(), Y_test.cpu().detach().numpy()

            mapper = umap.UMAP(random_state=0, n_neighbors = 10, n_components = 2).fit(X0)
            embedding0 = mapper.transform(X0)
            embedding1 = mapper.transform(X1)
            embedding2 = mapper.transform(X2)

            embed = [embedding0, embedding1, embedding2]
            y = [y0, y1, y2]

            # 結果を二次元でプロットする
            embedding_x0 = embedding0[:, 0]
            embedding_y0 = embedding0[:, 1]
            embedding_x1 = embedding1[:, 0]
            embedding_y1 = embedding1[:, 1]
            embedding_x2 = embedding2[:, 0]
            embedding_y2 = embedding2[:, 1]
            # fig = plt.figure(figsize=(10,10))
            size = 120
            plt.figure(figsize=(7, 7))

            # __all_labels = ["Negative", "Neutral", "Positive"]
            # __colors_dict = {"Negative": "blue", "Neutral": "gray", "Positive": "red"}
            # # {"Neutral": "gray", "Happy": "#ff7f00", "Surprise": "#ffff33", "Fear": "#a65628",
            # #  "Anger": "#e41a1c", "Disgust": "#984ea3", "Sad": "#377eb8", "Calm": "#4daf4a"}

            # ############# ラベル関連定数 ##############
            # labels = __all_labels
            # colors = [__colors_dict[l] for l in labels]

            # #######################################
            # plt.figure(figsize=(7, 7))
            # for embedding, y_ in zip(embed, y):
            #     for idx, cl in enumerate(labels):  # クラスごとにサンプルをプロット
            #         emospace_cor = embedding[y_ == idx]
            #         plt.scatter(x=emospace_cor[:, 0], y=emospace_cor[:, 1], alpha=0.9, c=colors[idx], label=cl,
            #                     s=120, linewidths=1, edgecolors="black")

            # # # 軸の範囲の設定
            # plt.xlabel("Test", fontsize=18), plt.ylabel("Testtest", fontsize=18), plt.tick_params(labelsize=18)
            # legend = plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=20, handletextpad=0.01)
            # # plt.savefig(f"{out_parentDir}/emotionalSpace_{modal}_{mode}.pdf", bbox_extra_artists=(legend,), bbox_inches="tight", pad_inches=0)
            # # __pltclf_ifneed()
            # plt.savefig(save_direct+'Proposed_'+str(exp+1)+'_'+str(count)+'.png')
            # plt.close()
            labels = ["Negative", "Neutral", "Positive"]
            l = ["Source", "Target", "Test"]
            handles = []
            h = []

            p0= plt.scatter(embedding_x0[(y0 == 0)],
                        embedding_y0[(y0 == 0)],
                        c = 'black', s=size, label = "Negative", marker = 'v')

            p00 = plt.scatter(embedding_x0[(y0 == 1)],
                        embedding_y0[(y0 == 1)],
                        c = 'black', s=size, label = "Neutral", marker = '^')

            p000 = plt.scatter(embedding_x0[(y0 == 2)],
                        embedding_y0[(y0 == 2)],
                        c = 'black', s=size, label = "Positive", marker = 'o')
        
            q1= plt.scatter(embedding_x0[(y0 == 0)],
                        embedding_y0[(y0 == 0)],
                        c = 'red', s=size, label = "Source", marker = ',')

            q2 = plt.scatter(embedding_x1[(y1 == 1)],
                        embedding_y1[(y1 == 1)],
                        c = 'blue', s=size, label = "Target", marker = ',')

            q3 = plt.scatter(embedding_x2[(y2 == 2)],
                        embedding_y2[(y2 == 2)],
                        c = 'green', s=size, label = "Test", marker = ',')
     
            p1= plt.scatter(embedding_x0[(y0 == 0)],
                        embedding_y0[(y0 == 0)],
                        c = 'red', s=size, label = "Negative", marker = 'v', linewidths=1, edgecolors="black")

            p2 = plt.scatter(embedding_x0[(y0 == 1)],
                        embedding_y0[(y0 == 1)],
                        c = 'red', s=size, label = "Neutral", marker = '^', linewidths=1, edgecolors="black")

            p3 = plt.scatter(embedding_x0[(y0 == 2)],
                        embedding_y0[(y0 == 2)],
                        c = 'red', s=size, label = "Positive", marker = 'o', linewidths=1, edgecolors="black")

            p4 = plt.scatter(embedding_x1[(y1 == 0)],
                        embedding_y1[(y1 == 0)],
                        c = 'blue', s=size, label = "Negative", marker = 'v', linewidths=1, edgecolors="black")

            p5 = plt.scatter(embedding_x1[(y1 == 1)],
                        embedding_y1[(y1 == 1)],
                        c = 'blue', s=size, label = "Neutral", marker = '^', linewidths=1, edgecolors="black")

            p6 = plt.scatter(embedding_x1[(y1 == 2)],
                        embedding_y1[(y1 == 2)],
                        c = 'blue', s=size, label = "Positive", marker = 'o', linewidths=1, edgecolors="black")

            p7 = plt.scatter(embedding_x2[(y2 == 0)],
                        embedding_y2[(y2 == 0)],
                        c = 'green', s=size, label = "Negative", marker = 'v', linewidths=1, edgecolors="black")

            p8 = plt.scatter(embedding_x2[(y2 == 1)],
                        embedding_y2[(y2 == 1)],
                        c = 'green', s=size, label = "Neutral", marker = '^', linewidths=1, edgecolors="black")

            p9 = plt.scatter(embedding_x2[(y2 == 2)],
                        embedding_y2[(y2 == 2)],
                        c = 'green', s=size, label = "Positive", marker = 'o', linewidths=1, edgecolors="black")

            handles1 = [p0, p00, p000]
            handles2 = [q1, q2, q3]

            p0.remove()
            p00.remove()
            p000.remove()
            q1.remove()
            q2.remove()
            q3.remove()
  
            # plt.grid()
            # leg1 = plt.legend(handles1, labels, fontsize = 15, loc=1)
            # leg2 = plt.legend(handles2, l, fontsize = 15, loc=4)
            plt.xlabel("Embedding Dimension 1", fontsize=18), plt.ylabel("Embedding Dimention 2", fontsize=18), plt.tick_params(labelsize=18)
            legend = plt.legend(handles1, labels, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=20, handletextpad=0.01)
            legend2 = plt.legend(handles2, l, bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0, fontsize=20, handletextpad=0.01)
            # plt.savefig(f"{out_parentDir}/emotionalSpace_{modal}_{mode}.pdf", bbox_extra_artists=(legend,), bbox_inches="tight", pad_inches=0)
            # __pltclf_ifneed()
            # plt.savefig(save_direct+'Proposed_'+str(exp+1)+'_'+str(count)+'.png')
            # plt.close()
            plt.gca().add_artist(legend)
            plt.savefig(save_direct+'Proposed_'+str(exp+1)+'_'+str(count)+'.pdf', bbox_extra_artists=(legend,legend2), bbox_inches="tight", pad_inches=0)
            plt.close()


            # feature_output_source = model_F(X_train)
            # feature_output_target = model_F(X_test)

            # X0, y0, d0 = feature_output_source.cpu().detach().numpy(), Y_train.cpu().detach().numpy(), D_train.cpu().detach().numpy()
            # X1, y1 = feature_output_target.cpu().detach().numpy(), Y_test.cpu().detach().numpy()#, D_target.cpu().detach().numpy()

            # mapper = umap.UMAP(random_state=0, n_neighbors = 10, n_components = 2).fit(X0)
            # embedding0 = mapper.transform(X0)
            # embedding1 = mapper.transform(X1)

            # # 結果を二次元でプロットする
            # embedding_x0 = embedding0[:, 0]
            # embedding_y0 = embedding0[:, 1]
            # embedding_x1 = embedding1[:, 0]
            # embedding_y1 = embedding1[:, 1]
            # fig = plt.figure(figsize=(10,10))
            # size = 30

            # # plt.scatter(embedding_x0[(y0 == 0)],
            # #             embedding_y0[(y0 == 0)],
            # #             c = 'red', s=size, label = 0, marker = 'o')

            # # plt.scatter(embedding_x0[(y0 == 1)],
            # #             embedding_y0[(y0 == 1)],
            # #             c = 'red', s=size, label = 1, marker = '^')

            # plt.scatter(embedding_x0[(y0 == 2)],
            #             embedding_y0[(y0 == 2)],
            #             c = 'red', s=size, label = "Source", marker = 'o')

            # # plt.scatter(embedding_x1[(y1 == 0)],
            # #             embedding_y1[(y1 == 0)],
            # #             c = 'blue', s=size, label = 0, marker = 'o')

            # # plt.scatter(embedding_x1[(y1 == 1)],
            # #             embedding_y1[(y1 == 1)],
            # #             c = 'blue', s=size, label = 1, marker = '^')

            # plt.scatter(embedding_x1[(y1 == 2)],
            #             embedding_y1[(y1 == 2)],
            #             c = 'blue', s=size, label = "Target", marker = 'o')

            # plt.grid()
            # plt.legend(fontsize = 15)
            # plt.savefig(save_direct+'Proposed_'+str(exp+1)+'_'+str(count)+'.png')
            # plt.close()

