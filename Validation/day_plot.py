#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import random

from Function.data_load import mkdata, mkdata2d, mkdata3d

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from Function.model import FeatureExtractor, Classifier, Discriminator
from Function.functions import try_gpu
from Function.mk_data_exp1 import mk_train, mk_val
from Function import param
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt

def make_average(matrix_, plt_version):
    c = np.zeros((3,3))
    if plt_version == "precision":
        for i in range(3):
            c[i] = (matrix_[i]/np.sum(matrix_[i]))*100
    else:
        for i in range(3):
            c[:,i] = (matrix_[:,i]/np.sum(matrix_[:,i]))*100
    return c

def class_accuracy(label, output):
    match_Negative = 0
    match_Neutral = 0
    match_Positive = 0
    for true, predict in zip(label, output):
        if true == predict and true == 0:
            match_Negative += 1
        if true == predict and true == 1:
            match_Neutral += 1
        if true == predict and true == 2:
            match_Positive += 1

    return match_Negative/label.count(0), match_Neutral/label.count(1), match_Positive/label.count(2)

def cfm_plot(all_true_label1, all_predict_label1, all_true_label2, all_predict_label2, all_true_label3, all_predict_label3, save_direct, plt_version):
    # fontのサイズ定義
    per_fontsize = 22
    xylabel_fontsize = 15
    label_fontsize = 14
    
    #################################################
    labels = ['Negative', 'Neutral', 'Positive']
    #################################################
    
    plt.figure()
    cx = confusion_matrix(all_true_label1, all_predict_label1)
    c1 = make_average(cx, plt_version)
    ax = plt.subplot()
    sns.heatmap(c1, annot = True, fmt = '.1f', ax = ax, cmap = 'Blues', vmax = 100, vmin = 0, square = True, annot_kws={"fontsize": per_fontsize})

    ax.set_xlabel('Predicted', fontsize = xylabel_fontsize)
    ax.set_ylabel('True', fontsize = xylabel_fontsize)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels, verticalalignment="center")
    ax.tick_params(labelsize=label_fontsize)
    plt.savefig(save_direct+'all_day1.png')
    plt.close()

    #################################################

    plt.figure()
    cxx = confusion_matrix(all_true_label2, all_predict_label2)
    c2 = make_average(cxx, plt_version)
    ax = plt.subplot()
    sns.heatmap(c2, annot = True, fmt = '.1f', ax = ax, cmap = 'Blues', vmax = 100, vmin = 0,square = True, annot_kws={"fontsize": per_fontsize})

    ax.set_xlabel('Predicted', fontsize = xylabel_fontsize)
    ax.set_ylabel('True', fontsize = xylabel_fontsize)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels, verticalalignment="center") 
    ax.tick_params(labelsize=label_fontsize)
    plt.savefig(save_direct+'all_day2.png')
    plt.close()

    #################################################

    plt.figure()
    cxxx = confusion_matrix(all_true_label3, all_predict_label3)
    c3 = make_average(cxxx, plt_version)
    ax = plt.subplot()
    sns.heatmap(c3, annot = True, fmt = '.1f', ax = ax, cmap = 'Blues', vmax = 100, vmin = 0,square = True, annot_kws={"fontsize": per_fontsize})

    ax.set_xlabel('Predicted', fontsize = xylabel_fontsize)
    ax.set_ylabel('True', fontsize = xylabel_fontsize)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels, verticalalignment="center") 
    ax.tick_params(labelsize=label_fontsize)
    plt.savefig(save_direct+'all_day3.png')
    plt.close()

def day_plot(EEG, label, EEG2, label2, EEG3, label3, kf, L_list, 
    load_direct, save_direct, plt_version):

    all_average1, all_average2, all_average3 = [], [], []

    all_Negative1, all_Neutral1, all_Positive1 = [], [], []
    all_Negative2, all_Neutral2, all_Positive2 = [], [], []
    all_Negative3, all_Neutral3, all_Positive3 = [], [], []

    all_true_label1, all_predict_label1 = [], []
    all_true_label2, all_predict_label2 = [], []
    all_true_label3, all_predict_label3 = [], []
    save_direct = save_direct + plt_version +"/"
    os.makedirs(save_direct, exist_ok=True)
    for CV_num in range(5):
        count = 0
        for train_index, test_index in kf.split(EEG):
                
            count += 1 #実験回数
            # print('############ 実験番号: '+str(count) + ' #############')
            # train,target,testデータの生成
            X_train1, Y_train2,_ = mk_train(EEG, label, train_index)
            X_target1, Y_target1, _, X_test1, Y_test1 = mk_val(EEG, label, test_index, L_list, num = CV_num)
            X_train2, Y_train2,_ = mk_train(EEG2, label2, train_index)
            X_target2, Y_target2, _, X_test2, Y_test2 = mk_val(EEG2, label2, test_index, L_list, num = CV_num)
            X_train3, Y_train3,_ = mk_train(EEG3, label3, train_index)
            X_target3, Y_target3, _, X_test3, Y_test3 = mk_val(EEG3, label3, test_index, L_list, num = CV_num)
            
            # testデータの感情ごとのデータ数確認
            # print(np.count_nonzero(Y_test1 == 0))
            # print(np.count_nonzero(Y_test1 == 1))
            # print(np.count_nonzero(Y_test1 == 2))
            # load_model = "model-one/model"+str(exp+1)+"/"
           
            load_model = "save/CV_data_run/"+"data_num_"+str(1)+"/CV_num"+str(CV_num+1)+"/model/sub_"+str(count)+"/"
           
            #各モデルの定義
            FE_path = load_model+"FE"+ str(count) + ".pth"
            C_path = load_model+"C"+str(count)+".pth"
            D_path = load_model+"D"+str(count)+".pth"
            model_F = FeatureExtractor(param.lstm_input_dim, param.lstm_hidden_dim, param.time_steps, param.K)
            model_C = Classifier(param.lstm_hidden_dim, param.K, param.class_num)
            model_D = Discriminator(param.lstm_hidden_dim, param.K, param.dense_hidden_dim, param.domain_num)
            model_F.load_state_dict(torch.load(FE_path))
            model_C.load_state_dict(torch.load(C_path))
            model_D.load_state_dict(torch.load(D_path))
            model_F = try_gpu(model_F)
            model_C = try_gpu(model_C)
            model_D = try_gpu(model_D)

            model_F.eval()
            model_C.eval()
            model_D.eval()

            label_t_eval1 = []
            class_t_eval1 = []
            test_accuracy = 0.0

            alpha = 0.8
            with torch.no_grad():
                X_test1 = try_gpu(X_test1)
                Y_test1 = try_gpu(Y_test1).long()

                feature_output_test = model_F(X_test1)
                class_output_test = model_C(feature_output_test)
                domain_output_test = model_D(feature_output_test, alpha)
            
                for k in range(class_output_test.shape[0]):
                    label_t_eval1.append(Y_test1[k].cpu().detach().item())
                    class_t_eval1.append(torch.argmax(class_output_test[k, :]).cpu().detach().item())
                    all_true_label1.append(Y_test1[k].cpu().detach().item())
                    all_predict_label1.append(torch.argmax(class_output_test[k, :]).cpu().detach().item())
        
                test_accuracy = accuracy_score(label_t_eval1, class_t_eval1)

                all_average1.append(test_accuracy)

                # print('test_accuracy: %.5f' % (test_accuracy*100))
                # print('Negative_accuracy1: %.5f' % (class_t_eval1.count(0)/label_t_eval1.count(0)*100))
                # print('Neutral_accuracy1: %.5f' % (class_t_eval1.count(1)/label_t_eval1.count(1)*100))
                # print('Positive_accuracy1: %.5f' % (class_t_eval1.count(2)/label_t_eval1.count(2)*100))

            Negative1, Neutral1, Positive1 = class_accuracy(label_t_eval1, class_t_eval1)

            all_Negative1.append(Negative1)
            all_Neutral1.append(Neutral1)
            all_Positive1.append(Positive1)


            label_t_eval2 = []
            class_t_eval2 = []
            test_accuracy = 0.0

            alpha = 0.8
            with torch.no_grad():
                X_test2 = try_gpu(X_test2)
                Y_test2 = try_gpu(Y_test2).long()

                feature_output_test = model_F(X_test2)
                class_output_test = model_C(feature_output_test)
                domain_output_test = model_D(feature_output_test, alpha)
                
                for k in range(class_output_test.shape[0]):
                    label_t_eval2.append(Y_test2[k].cpu().detach().item())
                    class_t_eval2.append(torch.argmax(class_output_test[k, :]).cpu().detach().item())
                    all_true_label2.append(Y_test2[k].cpu().detach().item())
                    all_predict_label2.append(torch.argmax(class_output_test[k, :]).cpu().detach().item())
        
                test_accuracy = accuracy_score(label_t_eval2, class_t_eval2)
                all_average2.append(test_accuracy)

                # print('test_accuracy: %.5f' % (test_accuracy*100))
                # print('Negative_accuracy2: %.5f' % (class_t_eval2.count(0)/label_t_eval2.count(0)))
                # print('Neutral_accuracy2: %.5f' % (class_t_eval2.count(1)/label_t_eval2.count(1)))
                # print('Positive_accuracy2: %.5f' % (class_t_eval2.count(2)/label_t_eval2.count(2)))

            Negative2, Neutral2, Positive2 = class_accuracy(label_t_eval2, class_t_eval2)

            all_Negative2.append(Negative2)
            all_Neutral2.append(Neutral2)
            all_Positive2.append(Positive2)
            
            label_t_eval3 = []
            class_t_eval3 = []
            test_accuracy = 0.0

            alpha = 0.8
            with torch.no_grad():
                X_test3 = try_gpu(X_test3)
                Y_test3 = try_gpu(Y_test3).long()

                feature_output_test = model_F(X_test3)
                class_output_test = model_C(feature_output_test)
                domain_output_test = model_D(feature_output_test, alpha)
            
                for k in range(class_output_test.shape[0]):
                    label_t_eval3.append(Y_test3[k].cpu().detach().item())
                    class_t_eval3.append(torch.argmax(class_output_test[k, :]).cpu().detach().item())
                    all_true_label3.append(Y_test3[k].cpu().detach().item())
                    all_predict_label3.append(torch.argmax(class_output_test[k, :]).cpu().detach().item())
        
                test_accuracy = accuracy_score(label_t_eval3, class_t_eval3)
                all_average3.append(test_accuracy)

                # print('test_accuracy: %.5f' % (test_accuracy*100))
                # print('Negative_accuracy3: %.5f' % (class_t_eval3.count(0)/label_t_eval3.count(0)))
                # print('Neutral_accuracy3: %.5f' % (class_t_eval3.count(1)/label_t_eval3.count(1)))
                # print('Positive_accuracy3: %.5f' % (class_t_eval3.count(2)/label_t_eval3.count(2)))

            Negative3, Neutral3, Positive3 = class_accuracy(label_t_eval3, class_t_eval3)

            all_Negative3.append(Negative3)
            all_Neutral3.append(Neutral3)
            all_Positive3.append(Positive3)
    
    cfm_plot(all_true_label1, all_predict_label1, all_true_label2, all_predict_label2, all_true_label3, all_predict_label3, save_direct, plt_version)


    
