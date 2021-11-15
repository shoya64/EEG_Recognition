#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
# import random
import csv

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

from Function.data_load import mkdata
from Function.model import FeatureExtractor, Classifier, Discriminator
from Function.functions import try_gpu
from Function.mk_data import mk_train, mk_test
from Function.dataset import Mydataset
from Function import param

import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt


# train-test
def validation(EEG, label, kf, L_list, direct, csv_file):
    for CV_num in range(5):
        print('############# 映画組番号：'+str(CV_num+1)+'　################')
        with open(csv_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(["CV:"+str(L_list[0][CV_num])+"/"+str(L_list[1][CV_num])+"/"+str(L_list[2][CV_num])])
        count = 0
        # csv出力用 リスト
        list_avg = []
        for train_index, test_index in kf.split(EEG):
            count += 1 #実験回数
            if count >= 0:
                if count <= 9:
                    num = "0"+str(count)
                else:
                    num = str(count)

                load_model = direct+"CV_"+str(CV_num+1)+"/model/sub_"+str(count)+"/"
                save_direct = direct+"VAL/CV_"+str(CV_num+1)+"/result/save"+num+"/"
                os.makedirs(save_direct, exist_ok=True)

                print('############ 実験番号: '+str(count) + ' #############')
                # train,target,testデータの生成
                xtrain, ytrain, dtrain = mk_train(EEG, label, train_index)
                xtarget, ytarget, dtarget, X_test, Y_test = mk_test(EEG, label, test_index, L_list, num = CV_num, mult = ytrain.shape[0])

                xtrain = np.concatenate([xtrain, xtarget], 0)
                ytrain = np.concatenate([ytrain, ytarget], 0)
                dtrain = np.concatenate([dtrain, dtarget], 0)

                X_train = try_gpu(torch.from_numpy(xtrain)).float()
                Y_train = try_gpu(torch.from_numpy(ytrain)).long()
                D_train = try_gpu(torch.from_numpy(dtrain)).long()
                X_target = try_gpu(torch.from_numpy(xtarget)).float()
                Y_target = try_gpu(torch.from_numpy(ytarget)).long()
                D_target = try_gpu(torch.from_numpy(dtarget)).long()
                X_test = try_gpu(X_test).float()
                Y_test = try_gpu(Y_test).long()
            
                #各モデルの定義
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

                label_eval = []
                class_eval = []
                train_accuracy = 0.0
                label_t_eval = []
                class_t_eval = []
                label_d_eval = []
                domain_d_eval = []
                test_accuracy = 0.0
               

                model_F.eval()
                model_C.eval()
                model_D.eval()
                
                alpha = 0.8
                with torch.no_grad():
                    # train flow
                    feature_output_train = model_F(X_train)

                    class_output_train = model_C(feature_output_train)

                    domain_output_train = model_D(feature_output_train, alpha)

                    # test flow
                    feature_output_test = model_F(X_test)

                    class_output_test = model_C(feature_output_test)

                    domain_output_test = model_D(feature_output_test, alpha)
                 
                    for k in range(class_output_train.shape[0]):
                        label_eval.append(Y_train[k].cpu().detach().item())
                        class_eval.append(torch.argmax(class_output_train[k, :]).cpu().detach().item())
                    for k in range(class_output_test.shape[0]):
                        label_t_eval.append(Y_test[k].cpu().detach().item())
                        class_t_eval.append(torch.argmax(class_output_test[k, :]).cpu().detach().item())

                    training_accuracy = accuracy_score(label_eval, class_eval)
                    test_accuracy = accuracy_score(label_t_eval, class_t_eval)

                    print('training_accuracy: %.5f' % (training_accuracy*100))
                    print('test_accuracy: %.5f' % (test_accuracy*100))

                    list_avg.append(test_accuracy*100)


                plt.figure()
                c = confusion_matrix(label_eval, class_eval)
                sns.heatmap(c, annot = True, fmt = 'g', square = True)
                plt.title("training")
                plt.xlabel("predict")
                plt.ylabel("true")
                plt.savefig(save_direct+'train_cm.png')
                plt.close()

                plt.figure()
                cx = confusion_matrix(label_t_eval, class_t_eval)
                sns.heatmap(cx, annot = True, fmt = 'g', square = True)
                plt.title("test")
                plt.xlabel("predict")
                plt.ylabel("true")
                plt.savefig(save_direct+'test_cm.png')
                plt.close()

                # plt.figure()
                # ct = confusion_matrix(domain_eval, d_label)
                # sns.heatmap(ct, annot = True, fmt = 'g', square = True)
                # plt.title("domain train")
                # plt.xlabel("predict")
                # plt.ylabel("true")
                # plt.savefig(save_direct+'domain_train.png')
                # plt.close()
                
                # plt.figure()
                # cxt = confusion_matrix(label_d_eval, domain_d_eval)
                # sns.heatmap(cxt, annot = True, fmt = 'g', square = True)
                # plt.title("domain test")
                # plt.xlabel("predict")
                # plt.ylabel("true")
                # plt.savefig(save_direct+'domain_test.png')
                # plt.close()

        with open(csv_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(list_avg)