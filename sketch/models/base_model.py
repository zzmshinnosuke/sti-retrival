#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2021-06-30 10:30:18
# @Author: zzm

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl

class BaseModel(pl.LightningModule):
    def __init__(self,config):
        super().__init__()
        self.save_hyperparameters()
        self.config=config
        self._set_hparams(config)
        self.get_model()
        
    def get_model(self):
        raise NotImplementedError
        
    def forward(self,x):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer=torch.optim.SGD(self.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        scheduler=ReduceLROnPlateau(optimizer,mode='min',
                                factor=self.config.learning_rate_factor,
                                patience=self.config.learning_rate_decay_frequency)
        return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'monitor': 'val_loss',
        }
    }
    
    def training_step(self,train_batch,batch_idx):
        raise NotImplementedError
        
    def validation_step(self,val_batch,batch_idx):
        raise NotImplementedError
    
    
        
    


