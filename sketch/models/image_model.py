#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2021-06-30 17:48:24
# @Author: zzm

from .base_model import BaseModel

import torchvision
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.loggers import CSVLogger
import numpy as np

class GlobalAvgPool2d(torch.nn.Module):
    def forward(self,x):
        return F.adaptive_avg_pool2d(x,(1,1))

class ImageModel(BaseModel):
   
    def get_model(self):
        self.use_cuda=True
        self.enc_hidden_size=256
        self.dropout = self.config.dropout
        self.batch_size = self.config.batch_size
        
        model= torchvision.models.resnet18(pretrained=True)
        
        model.avgpool=GlobalAvgPool2d()
        model.fc=torch.nn.Sequential(torch.nn.Linear(model.fc.in_features,512))
        self.img_model=model
        
        self.train_metric=torchmetrics.Accuracy(ignore_index=255)
        self.valid_metric=torchmetrics.Accuracy(ignore_index=255)
        
        self.csv_logger=CSVLogger(self.config.log_dir,name=self.config.note)
    
    def forward(self,x):
        return self.img_model(x)
    
    def fetch_batch(self,batch):
        print("batch:",type(batch),batch)
        assert type(batch) is dict
        img1 = np.stack(batch["source_ske_img_data"])
        img1 = torch.from_numpy(img1).float()
        img1 = torch.autograd.Variable(img1)
        img2 = np.stack([batch['target_img_data']])
        img2 = torch.from_numpy(img2).float()
        img2 = torch.autograd.Variable(img2)
        return img1,img2
    
    def training_step(self,train_batch,batch_idx):
        loss=self.compute_loss(self.forward(train_batch["source_ske_img_data"]),self.forward(train_batch["target_img_data"]))

#         acc=self.train_metric(result,y)
        self.log('train_loss',loss,on_step=True,on_epoch=True)
        return loss
    
    def validation_step(self,val_batch,batch_idx):
        loss=self.compute_loss(self.forward(val_batch["source_ske_img_data"]),self.forward(val_batch["target_img_data"]))
        
        self.log('val_loss',loss,on_step=False,on_epoch=True)
#         self.log('acc/val',acc,on_step=True,on_epoch=False)
        return loss
        
    def training_epoch_end(self,outs):
#         acc=self.train_metric.compute()
#         self.log('acc/train_epoch',acc)
        self.csv_logger.log_metrics({'acc/train_epoch':acc})
        
    def validation_epoch_end(self,outs):
#         acc=self.valid_metric.compute()
#         self.log('acc/val_epoch',acc)
        self.csv_logger.log_metrics({'acc/val_epoch':acc})
    
    def on_train_end(self):
        self.csv_logger.save()