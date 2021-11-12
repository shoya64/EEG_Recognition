#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:20:45 2020

@author: furukawashouya
"""

import torch
class Mydataset(torch.utils.data.Dataset):
    def __init__(self, data, label, domain, transform = None):
        super().__init__()
        
        self.transform = transform
        self.data = data
        self.label = label
        self.domain = domain
        self.data_num = data.shape[0]
        
        
        self.x = []
        for i in range(self.data.shape[0]):
            self.x.append(data[i, :, :])
            
        self.y = []
        for j in range(self.label.shape[0]):
            self.y.append(label[j])
            
        self.d = []
        for l in range(self.domain.shape[0]):
            self.d.append(domain[l])
         
         
        
        self.len = len(self.x)
        
        
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        label = self.y[idx]
        domain = self.d[idx]
        data = self.x[idx]
        data = torch.Tensor(data)
        
        return data, label, domain