#!/usr/bin/env python3
# -*- coding: utf-8 -*-
####パラメータ設定####
# イテレーション
epochs_num = 50#50 #100 #40
batch_size = 256 #256 # 学習率を落とさず，バッチサイズを大きくする

# モデル
lstm_input_dim = 310
lstm_hidden_dim = 150
time_steps = 9
K = 5
dense_hidden_dim = 100
class_num = 3
domain_num = 2
domain_num2 = 2
alpha = 1.0

# optimizer
SGD_lr = 0.01
momentum = 0.9
Adam_lr = 1e-3
weight_decay = 0.01

# schedular
step_size = 10 
gamma = 0.5 #1.0

# 初期値
seed = 0

run_num = 8 #1