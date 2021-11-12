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
def train_test(EEG, label, kf, L_list, direct, csv_file):
    for CV_num in range(5):
        print('############# 映画組番号：'+str(CV_num)+'　################')
        with open(csv_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(["CV:"+str(L_list[0][CV_num])+"/"+str(L_list[1][CV_num])+"/"+str(L_list[2][CV_num])])
        for exp in range(3):
            final_average = []
            count = 0
            for train_index, test_index in kf.split(EEG):
                count += 1 #実験回数
                if count >= 0:
                    if count <= 9:
                        num = "0"+str(count)
                    else:
                        num = str(count)

                    save_model = direct+"CV_num"+str(CV_num+1)+"/exp"+str(exp+1)+"/model/sub_"+str(count)+"/"
                    os.makedirs(save_model, exist_ok=True)
                    save_direct = direct+"CV_num"+str(CV_num+1)+"/exp"+str(exp+1)+"/result/save"+num+"/"
                    os.makedirs(save_direct, exist_ok=True)

                    print('############ 実験番号: '+str(count) + ' #############')
                    # train,target,testデータの生成
                    xtrain, ytrain, dtrain = mk_train(EEG, label, train_index)
                    xtarget, ytarget, dtarget, X_test, Y_test = mk_test(EEG, label, test_index, L_list, num = CV_num, mult = ytrain.shape[0])

                    xtrain = np.concatenate([xtrain, xtarget], 0)
                    ytrain = np.concatenate([ytrain, ytarget], 0)
                    dtrain = np.concatenate([dtrain, dtarget], 0)

                    train = Mydataset(xtrain, ytrain, dtrain, transform = None)

                    #各モデルの定義
                    model_F = FeatureExtractor(param.lstm_input_dim, param.lstm_hidden_dim, param.time_steps, param.K)
                    model_C = Classifier(param.lstm_hidden_dim, param.K, param.class_num)
                    model_D = Discriminator(param.lstm_hidden_dim, param.K, param.dense_hidden_dim, param.domain_num2)
                    model_F = try_gpu(model_F)
                    model_C = try_gpu(model_C)
                    model_D = try_gpu(model_D)

                    # AdamW
                    optimizer_F = optim.AdamW(model_F.parameters(), lr = param.Adam_lr, weight_decay = param.weight_decay)
                    optimizer_C = optim.AdamW(model_C.parameters(), lr = param.Adam_lr, weight_decay = param.weight_decay)
                    optimizer_D = optim.AdamW(model_D.parameters(), lr = param.Adam_lr, weight_decay = param.weight_decay)

                    # 学習率を緩める
                    scheduler_F = optim.lr_scheduler.StepLR(optimizer_F, step_size = param.step_size, gamma = param.gamma)
                    scheduler_C = optim.lr_scheduler.StepLR(optimizer_C, step_size = param.step_size, gamma = param.gamma)
                    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size = param.step_size, gamma = param.gamma)

                    # scheduler_F = optim.lr_scheduler.ReduceLROnPlateau(optimizer_F, mode='min', factor=0.5, patience=5, cooldown=5)
                    # scheduler_C = optim.lr_scheduler.ReduceLROnPlateau(optimizer_C, mode='min', factor=0.5, patience=5, cooldown=5)
                    # scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5, cooldown=5)

                    class_loss_function = nn.CrossEntropyLoss()

                    pltloss = np.array([0])
                    e = np.arange(1, param.epochs_num + 1)

                    plttrain_accuracy = np.array([0])
                    plttest_accuracy = np.array([0])
                    test_pltloss = np.array([0])
                    accuracy = 0.0

                    for epoch in range(param.epochs_num):
                        train_loader = torch.utils.data.DataLoader(train, batch_size = param.batch_size, shuffle = True)

                        running_loss = 0.0
                        training_accuracy = 0.0
                        label_eval = []
                        class_eval = []
                        domain_eval = []
                        d_label = []

                        d_loss = 0.0
                        c_loss = 0.0

                        model_F.train()
                        model_C.train()
                        model_D.train()
                        for X_train, Y_train, D_train in train_loader:
                            optimizer_F.zero_grad()
                            optimizer_C.zero_grad()
                            optimizer_D.zero_grad()
                            torch.autograd.set_detect_anomaly(True)

                            X_train = try_gpu(X_train)
                            Y_train = try_gpu(Y_train).long()
                            D_train = try_gpu(D_train).long()

                            feature_output_C = model_F(X_train)
                            feature_output_D = feature_output_C.clone().detach() ####

                            # Discriminatorの更新
                            domain_output = model_D(feature_output_D, param.alpha) 

                            domain_loss = class_loss_function(domain_output, D_train)
                        
                            domain_loss.backward()
                            optimizer_D.step()
                            optimizer_D.zero_grad()

                            # FeatureExtractorとClassifierの更新
                            class_output = model_C(feature_output_C)

                            class_loss = class_loss_function(class_output, Y_train)

                            domain_output = model_D(feature_output_C, param.alpha)

                            domain_loss = class_loss_function(domain_output, D_train)

                            loss = class_loss + domain_loss

                            loss.backward()
                            optimizer_F.step()
                            optimizer_C.step()

                            d_loss += float(domain_loss)
                            c_loss += float(class_loss)
                            
                            running_loss += float(loss.item())

                            with torch.no_grad():
                                for k in range(class_output.shape[0]):
                                    label_eval.append(Y_train[k].cpu().detach().item())
                                    class_eval.append(torch.argmax(class_output[k, :]).cpu().detach().item())
                                for m in range(domain_output.shape[0]):
                                    domain_eval.append(D_train[m].cpu().detach().item())
                                    d_label.append(torch.argmax(domain_output[m,:]).cpu().detach().item())

                                training_accuracy = accuracy_score(label_eval, class_eval)

                        scheduler_F.step()
                        scheduler_C.step()
                        scheduler_D.step()

                        with torch.no_grad():    
                            if epoch == 0:
                                pltloss = np.array([running_loss])
                                plttrain_accuracy = np.array([training_accuracy*100])
                            else:
                                pltloss = np.append(pltloss, running_loss)
                                plttrain_accuracy = np.append(plttrain_accuracy, training_accuracy*100)
                            
                            # 表示させたければ
                            print('%d loss: %.3f, training_accuracy: %.5f' % (epoch + 1, running_loss, training_accuracy*100))
                            print('class_loss: %.5f, domain_loss: %.5f' % (c_loss, d_loss))


                        model_F.eval()
                        model_C.eval()
                        model_D.eval()

                        label_t_eval = []
                        class_t_eval = []
                        label_d_eval = []
                        domain_d_eval = []
                        test_accuracy = 0.0
                        eval_loss = 0.0
                
                        alpha = 0.8
                        with torch.no_grad():
                            X_test = try_gpu(X_test)
                            Y_test = try_gpu(Y_test).long()
                            domain_label = torch.ones(Y_test.shape,)

                            feature_output_test = model_F(X_test)
                            class_output_test = model_C(feature_output_test)
                            domain_output_test = model_D(feature_output_test, alpha)
                            
                            test_loss = class_loss_function(class_output_test, Y_test)

                            eval_loss += float(test_loss.item())
                        
                            for k in range(class_output_test.shape[0]):
                                label_t_eval.append(Y_test[k].cpu().detach().item())
                                class_t_eval.append(torch.argmax(class_output_test[k, :]).cpu().detach().item())
                                
                            for k in range(domain_output_test.shape[0]):
                                label_d_eval.append(domain_label[k].cpu().detach().item())
                                domain_d_eval.append(torch.argmax(domain_output_test[k, :]).cpu().detach().item())

                            test_accuracy = accuracy_score(label_t_eval, class_t_eval)

                        with torch.no_grad():
                            if epoch == 0:
                                test_pltloss = np.array([eval_loss])
                                plttest_accuracy = np.array([test_accuracy*100])
                            else:
                                test_pltloss = np.append(test_pltloss, eval_loss)
                                plttest_accuracy = np.append(plttest_accuracy, test_accuracy*100)
                            # 表示させたければ
                            print('test_loss: %.3f, test_accuracy: %.5f' % (eval_loss, test_accuracy*100))

                   
                    print("SAVE")
                    SE_path = save_model+"FE"+ str(count) + ".pth"
                    C_path = save_model+"C"+str(count)+".pth"
                    D_path = save_model+"D"+str(count)+".pth"
                    torch.save(model_F.state_dict(), SE_path)
                    torch.save(model_C.state_dict(), C_path)
                    torch.save(model_D.state_dict(), D_path)
                    
                    plt.figure()
                    plt.plot(e, pltloss)
                    plt.title('train loss')
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')
                    plt.legend(['Train', 'Test'], loc = 'upper left')
                    plt.savefig(save_direct+'train_loss.png')
                    plt.close()

                    plt.figure()
                    plt.plot(e, plttrain_accuracy, color = 'red')
                    plt.plot(e, plttest_accuracy, color = 'blue')
                    plt.title('Accuracy')
                    plt.ylabel('correct answer rate')
                    plt.xlabel('Epoch')
                    plt.legend(['Train', 'Test'], loc = 'upper left')
                    plt.savefig(save_direct+'accuracy.png')
                    plt.close()

                    plt.figure()
                    plt.plot(e, test_pltloss)
                    plt.title('Test loss')
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')
                    plt.legend(['Train', 'Test'], loc = 'upper left')
                    plt.savefig(save_direct+'test_loss.png')
                    plt.close()

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

                    plt.figure()
                    ct = confusion_matrix(domain_eval, d_label)
                    sns.heatmap(ct, annot = True, fmt = 'g', square = True)
                    plt.title("domain train")
                    plt.xlabel("predict")
                    plt.ylabel("true")
                    plt.savefig(save_direct+'domain_train.png')
                    plt.close()
                    
                    plt.figure()
                    cxt = confusion_matrix(label_d_eval, domain_d_eval)
                    sns.heatmap(cxt, annot = True, fmt = 'g', square = True)
                    plt.title("domain test")
                    plt.xlabel("predict")
                    plt.ylabel("true")
                    plt.savefig(save_direct+'domain_test.png')
                    plt.close()

                    final_average.append(test_accuracy*100)

            # print(final_average)
            # print(sum(final_average)/len(final_average))
            # print(str(exp+1),last_dict)
            with open(csv_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(final_average)