import os
import sys
# import random
import csv

import numpy as np

from Function.data_load import mkdata, mkdata2d, mkdata3d
from Function.functions import cv_random, mk_list
from sklearn.model_selection import KFold
from Validation.day_plot import day_plot 
from Function import param
def main():
    # data load
    EEG, label = mkdata()
    EEG2, label2 = mkdata2d()
    EEG3, label3 = mkdata3d()
    # Leave-One-Person-Out cross-validation
    kf = KFold(n_splits = 15, shuffle = False)
    L_list = mk_list() #cv_random(param.seed)
    # Save direct
    load_direct = "save/"
    save_direct = "save/fig/3day_cfm/png/"
    os.makedirs(save_direct, exist_ok=True)

    ### precision or recallを選択する
    plt_version = "precision"
    # plt_version = "recall"
   
    day_plot(EEG, label, EEG2, label2, EEG3, label3, 
        kf, L_list, load_direct, save_direct, plt_version)

if __name__=='__main__':
    main()