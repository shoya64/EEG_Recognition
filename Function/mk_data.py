#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
import sys
from operator import mul
from functools import reduce
import itertools

# 訓練データ生成
def mk_train(EEG, label, train_index):
    xtrain, ytrain, dtrain = [], [], []
    for ti in train_index:
        eeg_train = np.concatenate(EEG[ti])
        label_train = np.concatenate(label[ti])
        xtrain.append(eeg_train)
        ytrain.append(label_train)

    xtrain = np.concatenate(xtrain)
    ytrain = np.concatenate(ytrain)

    # (被験者番号，　データ数，　時系列数，　特徴数)
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)
    dtrain = np.zeros(ytrain.shape, dtype = 'int')

    return xtrain, ytrain, dtrain

# 推定対象被験者データの生成
def mk_test(EEG, label, test_index, L_list, num=0, mult = 5180, val = 1): # val=0(3),1(6),2(9),3(12)
    val += 1 #調整
    if val == 1:
          # ターゲットデータのvalidationパターン
        if num == 0:
            sort = [0,1,2,3,4]
        elif num == 1:
            sort = [1,0,2,3,4]
        elif num == 2:
            sort = [1,2,0,3,4]
        elif num == 3:
            sort = [1,2,3,0,4]
        else:
            sort = [1,2,3,4,0]
    
        X_target_L = []
        X_test_L = []
        Y_target_L = []
        Y_test_L = []
        for te in test_index:
            count = 0
            for i in sort:
                if i == 0:
                    for j in range(len(L_list)):
                        X_target_L.append(EEG[te][L_list[j][count]])
                        Y_target_L.append(label[te][L_list[j][count]])
                else:
                    for j in range(len(L_list)):
                        X_test_L.append(EEG[te][L_list[j][count]])
                        Y_test_L.append(label[te][L_list[j][count]])
                count += 1
        
        X_target = np.concatenate(X_target_L)
        Y_target = np.concatenate(Y_target_L)
        X_test = np.concatenate(X_test_L)
        Y_test = np.concatenate(Y_test_L)

        # Targetデータ生成
        bat = int(mult/X_target.shape[0])
        X_target = np.tile(X_target,(bat,1,1))
        Y_target = np.tile(Y_target,bat)
        D_target = np.ones(Y_target.shape, dtype = 'int')

        #　評価用データ
        X_test = torch.from_numpy(X_test).float()
        Y_test = torch.from_numpy(Y_test)
    
    # 仕様変更：combinationの全組み合わせをリスト化する
    elif val ==2:
        # sort = [0,1,2,3,4]
        # c = itertools.combinations(sort, 2)
        # for v in c:
        #     print(v)
        
        # ごり押し 汚いのでいつか直すこと
        if num == 0:
            sort = [0,0,1,1,1]
        if num == 1:
            sort = [0,1,0,1,1]
        if num == 2:
            sort = [0,1,1,0,1]
        if num == 3:
            sort = [0,1,1,1,0]
        if num == 4:
            sort = [1,0,0,1,1]
        if num == 5:
            sort = [1,0,1,0,1]
        if num == 6:
            sort = [1,0,1,1,0]
        if num == 7:
            sort = [1,1,0,0,1]
        if num == 8:
            sort = [1,1,0,1,0]
        if num == 9:
            sort = [1,1,1,0,0]
    
        X_target_L = []
        X_test_L = []
        Y_target_L = []
        Y_test_L = []
        for te in test_index:
            count = 0
            for i in sort:
                if i==0:
                    for j in range(len(L_list)):
                        X_target_L.append(EEG[te][L_list[j][count]])
                        Y_target_L.append(label[te][L_list[j][count]])
                else:
                    for j in range(len(L_list)):
                        X_test_L.append(EEG[te][L_list[j][count]])
                        Y_test_L.append(label[te][L_list[j][count]])
                count += 1
        
        X_target = np.concatenate(X_target_L)
        Y_target = np.concatenate(Y_target_L)
        X_test = np.concatenate(X_test_L)
        Y_test = np.concatenate(Y_test_L)

        # Targetデータ生成
        bat = int(mult/X_target.shape[0])
        X_target = np.tile(X_target,(bat,1,1))
        Y_target = np.tile(Y_target,bat)
        D_target = np.ones(Y_target.shape, dtype = 'int')

        #　評価用データ
        X_test = torch.from_numpy(X_test).float()
        Y_test = torch.from_numpy(Y_test)

    elif val ==3:

        if num == 0:
            sort = [0,0,1,1,1]
        if num == 1:
            sort = [0,1,0,1,1]
        if num == 2:
            sort = [0,1,1,0,1]
        if num == 3:
            sort = [0,1,1,1,0]
        if num == 4:
            sort = [1,0,0,1,1]
        if num == 5:
            sort = [1,0,1,0,1]
        if num == 6:
            sort = [1,0,1,1,0]
        if num == 7:
            sort = [1,1,0,0,1]
        if num == 8:
            sort = [1,1,0,1,0]
        if num == 9:
            sort = [1,1,1,0,0]
        
        X_target_L = []
        X_test_L = []
        Y_target_L = []
        Y_test_L = []
        for te in test_index:
            count = 0
            for i in sort:
                if i == 1:
                    for j in range(len(L_list)):
                        X_target_L.append(EEG[te][L_list[j][count]])
                        Y_target_L.append(label[te][L_list[j][count]])
                else:
                    for j in range(len(L_list)):
                        X_test_L.append(EEG[te][L_list[j][count]])
                        Y_test_L.append(label[te][L_list[j][count]])
                count += 1
        
        X_target = np.concatenate(X_target_L)
        Y_target = np.concatenate(Y_target_L)
        X_test = np.concatenate(X_test_L)
        Y_test = np.concatenate(Y_test_L)

        # Targetデータ生成
        bat = int(mult/X_target.shape[0])
        X_target = np.tile(X_target,(bat,1,1))
        Y_target = np.tile(Y_target,bat)
        D_target = np.ones(Y_target.shape, dtype = 'int')

        #　評価用データ
        X_test = torch.from_numpy(X_test).float()
        Y_test = torch.from_numpy(Y_test)


    else:
          # ターゲットデータのvalidationパターン
        if num == 0:
            sort = [0,1,2,3,4]
        elif num == 1:
            sort = [1,0,2,3,4]
        elif num == 2:
            sort = [1,2,0,3,4]
        elif num == 3:
            sort = [1,2,3,0,4]
        else:
            sort = [1,2,3,4,0]
    
        X_target_L = []
        X_test_L = []
        Y_target_L = []
        Y_test_L = []
        for te in test_index:
            count = 0
            for i in sort:
                if i == 0:
                    for j in range(len(L_list)):
                        X_test_L.append(EEG[te][L_list[j][count]])
                        Y_test_L.append(label[te][L_list[j][count]])
                else:
                    for j in range(len(L_list)):
                        X_target_L.append(EEG[te][L_list[j][count]])
                        Y_target_L.append(label[te][L_list[j][count]])
                count += 1
        
        X_target = np.concatenate(X_target_L)
        Y_target = np.concatenate(Y_target_L)
        X_test = np.concatenate(X_test_L)
        Y_test = np.concatenate(Y_test_L)

        # Targetデータ生成
        bat = int(mult/X_target.shape[0])
        X_target = np.tile(X_target,(bat,1,1))
        Y_target = np.tile(Y_target,bat)
        D_target = np.ones(Y_target.shape, dtype = 'int')

        #　評価用データ
        X_test = torch.from_numpy(X_test).float()
        Y_test = torch.from_numpy(Y_test)
    

    return X_target, Y_target, D_target, X_test, Y_test


def mk_val(EEG, label, test_index, L_list, num=0, val = 0): # val=0(3),1(6),2(9),3(12)
    val += 1 #調整
    if val == 1:
          # ターゲットデータのvalidationパターン
        if num == 0:
            sort = [0,1,2,3,4]
        elif num == 1:
            sort = [1,0,2,3,4]
        elif num == 2:
            sort = [1,2,0,3,4]
        elif num == 3:
            sort = [1,2,3,0,4]
        else:
            sort = [1,2,3,4,0]
    
        X_target_L = []
        X_test_L = []
        Y_target_L = []
        Y_test_L = []
        for te in test_index:
            count = 0
            for i in sort:
                if i == 0:
                    for j in range(len(L_list)):
                        X_target_L.append(EEG[te][L_list[j][count]])
                        Y_target_L.append(label[te][L_list[j][count]])
                else:
                    for j in range(len(L_list)):
                        X_test_L.append(EEG[te][L_list[j][count]])
                        Y_test_L.append(label[te][L_list[j][count]])
                count += 1
        
        X_target = np.concatenate(X_target_L)
        Y_target = np.concatenate(Y_target_L)
        X_test = np.concatenate(X_test_L)
        Y_test = np.concatenate(Y_test_L)
        D_target = np.ones(Y_target.shape, dtype = 'int')

        #　評価用データ
        X_test = torch.from_numpy(X_test).float()
        Y_test = torch.from_numpy(Y_test)
    
    # 仕様変更：combinationの全組み合わせをリスト化する
    elif val ==2:
        # sort = [0,1,2,3,4]
        # c = itertools.combinations(sort, 2)
        # for v in c:
        #     print(v)
        
        # ごり押し 汚いのでいつか直すこと
        if num == 0:
            sort = [0,0,1,1,1]
        if num == 1:
            sort = [0,1,0,1,1]
        if num == 2:
            sort = [0,1,1,0,1]
        if num == 3:
            sort = [0,1,1,1,0]
        if num == 4:
            sort = [1,0,0,1,1]
        if num == 5:
            sort = [1,0,1,0,1]
        if num == 6:
            sort = [1,0,1,1,0]
        if num == 7:
            sort = [1,1,0,0,1]
        if num == 8:
            sort = [1,1,0,1,0]
        if num == 9:
            sort = [1,1,1,0,0]
    
        X_target_L = []
        X_test_L = []
        Y_target_L = []
        Y_test_L = []
        for te in test_index:
            count = 0
            for i in sort:
                if i==0:
                    for j in range(len(L_list)):
                        X_target_L.append(EEG[te][L_list[j][count]])
                        Y_target_L.append(label[te][L_list[j][count]])
                else:
                    for j in range(len(L_list)):
                        X_test_L.append(EEG[te][L_list[j][count]])
                        Y_test_L.append(label[te][L_list[j][count]])
                count += 1
        
        X_target = np.concatenate(X_target_L)
        Y_target = np.concatenate(Y_target_L)
        X_test = np.concatenate(X_test_L)
        Y_test = np.concatenate(Y_test_L)
        
        D_target = np.ones(Y_target.shape, dtype = 'int')


        #　評価用データ
        X_test = torch.from_numpy(X_test).float()
        Y_test = torch.from_numpy(Y_test)

    elif val ==3:

        if num == 0:
            sort = [0,0,1,1,1]
        if num == 1:
            sort = [0,1,0,1,1]
        if num == 2:
            sort = [0,1,1,0,1]
        if num == 3:
            sort = [0,1,1,1,0]
        if num == 4:
            sort = [1,0,0,1,1]
        if num == 5:
            sort = [1,0,1,0,1]
        if num == 6:
            sort = [1,0,1,1,0]
        if num == 7:
            sort = [1,1,0,0,1]
        if num == 8:
            sort = [1,1,0,1,0]
        if num == 9:
            sort = [1,1,1,0,0]
        
        X_target_L = []
        X_test_L = []
        Y_target_L = []
        Y_test_L = []
        for te in test_index:
            count = 0
            for i in sort:
                if i == 1:
                    for j in range(len(L_list)):
                        X_target_L.append(EEG[te][L_list[j][count]])
                        Y_target_L.append(label[te][L_list[j][count]])
                else:
                    for j in range(len(L_list)):
                        X_test_L.append(EEG[te][L_list[j][count]])
                        Y_test_L.append(label[te][L_list[j][count]])
                count += 1
        
        X_target = np.concatenate(X_target_L)
        Y_target = np.concatenate(Y_target_L)
        X_test = np.concatenate(X_test_L)
        Y_test = np.concatenate(Y_test_L)
        
        D_target = np.ones(Y_target.shape, dtype = 'int')

        #　評価用データ
        X_test = torch.from_numpy(X_test).float()
        Y_test = torch.from_numpy(Y_test)


    else:
          # ターゲットデータのvalidationパターン
        if num == 0:
            sort = [0,1,2,3,4]
        elif num == 1:
            sort = [1,0,2,3,4]
        elif num == 2:
            sort = [1,2,0,3,4]
        elif num == 3:
            sort = [1,2,3,0,4]
        else:
            sort = [1,2,3,4,0]
    
        X_target_L = []
        X_test_L = []
        Y_target_L = []
        Y_test_L = []
        for te in test_index:
            count = 0
            for i in sort:
                if i == 0:
                    for j in range(len(L_list)):
                        X_test_L.append(EEG[te][L_list[j][count]])
                        Y_test_L.append(label[te][L_list[j][count]])
                else:
                    for j in range(len(L_list)):
                        X_target_L.append(EEG[te][L_list[j][count]])
                        Y_target_L.append(label[te][L_list[j][count]])
                count += 1
        
        X_target = np.concatenate(X_target_L)
        Y_target = np.concatenate(Y_target_L)
        X_test = np.concatenate(X_test_L)
        Y_test = np.concatenate(Y_test_L)

        D_target = np.ones(Y_target.shape, dtype = 'int')

        #　評価用データ
        X_test = torch.from_numpy(X_test).float()
        Y_test = torch.from_numpy(Y_test)
    

    return X_target, Y_target, D_target, X_test, Y_test

