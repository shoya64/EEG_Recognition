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
from FeaturePlot import Feature_plot
from Function import param
def main():
    # data load
    EEG, label = mkdata()
    # Leave-One-Person-Out cross-validation
    kf = KFold(n_splits = 15, shuffle = False)
    L_list = mk_list() #cv_random(param.seed)
    # Save direct
    save_direct = "FeaturePlot/save/"
    os.makedirs(save_direct, exist_ok=True)
    load_direct = "save/CV_data/"
    
    Feature_plot.UMAP_plot(EEG, label, kf, L_list, load_direct, save_direct)

if __name__=='__main__':
    main()