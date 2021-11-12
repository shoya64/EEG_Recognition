#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn
from .functions import ReverseLayerF
import numpy as np
import torch

    
class FeatureExtractor(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, time_steps, K):
        super(FeatureExtractor, self).__init__()

        self.lstm = nn.LSTM(input_size = lstm_input_dim, 
                           hidden_size = lstm_hidden_dim,
                           batch_first = True
                           )

        self.dropout = nn.Dropout(p = 0.4) 
        
        self.sigmoid = nn.Sigmoid()
        
        self.capsule = nn.Linear(time_steps, K)

    def forward(self, X_input, hidden0 = None):

        lstm_out, _ = self.lstm(X_input) #X_input.shape = (batch_size, time_steps, lstm_hidden_dim)
       
        out = torch.transpose(lstm_out, 1, 2) #out.shape = (batch_size, lstm_hidden_dim, time_steps)
        
        H = self.capsule(out) #H.shape = (batch_size, lstm_hidden, K = 3)
                
        H = H.view(H.shape[0], -1) #H.shape = (batch_size, K * lstm_hidden)
        
        H = self.sigmoid(H)

        H = self.dropout(H)
    
        return H

class Classifier(nn.Module):
    def __init__(self, lstm_hidden_dim, K, class_num):
        super(Classifier, self).__init__()

        self.dense = nn.Linear(K * lstm_hidden_dim, class_num)

        self.dropout = nn.Dropout(p = 0.3)

    def forward(self, feature_output):

        output = self.dense(feature_output)

        output = self.dropout(output)

        return output

class Discriminator(nn.Module):
    def __init__(self, lstm_hidden_dim, K, dense_hidden_dim, domain_num):
        super(Discriminator, self).__init__()

        self.dense1 = nn.Linear(K * lstm_hidden_dim, dense_hidden_dim)
        
        self.relu = nn.ReLU() ## nn.LeakyReLU(0.01) ##

        self.dense2 = nn.Linear(dense_hidden_dim, domain_num)

        self.dropout1 = nn.Dropout(p = 0.4)

        self.dropout2 = nn.Dropout(p = 0.3)

    def forward(self, feature_output, alpha):

        #GRL
        reverse_feature = ReverseLayerF.apply(feature_output, alpha)
        
        #ドメイン分類層
        output = self.dense1(reverse_feature)

        output = self.relu(output)

        output = self.dropout1(output)

        output = self.dense2(output)

        output = self.dropout2(output)

        return output
    
