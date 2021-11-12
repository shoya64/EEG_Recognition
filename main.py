#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
# import random
import csv

import numpy as np

from Function.data_load import mkdata
from Function.functions import cv_random, mk_list
from sklearn.model_selection import KFold
from train import train
from Function import param
def main():
    # data load
    EEG, label = mkdata()
    # Leave-One-Person-Out cross-validation
    kf = KFold(n_splits = 15, shuffle = False)
    L_list = cv_random() #cv_random(param.seed)
    # Save direct
    direct = "save/exp/"
    os.makedirs(direct, exist_ok=True)
    # Save CSV file
    csv_file = direct+'accuracy.csv'
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(np.array(L_list)) # ランダムのk-foldを保存
    
    train.train_test(EEG, label, kf, L_list, direct, csv_file)

if __name__=='__main__':
    main()