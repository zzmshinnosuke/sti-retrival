#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2021-06-30 10:30:18
# @Author: zzm

import torch_functions

import torch
import pytorch_lightning as pl
import numpy as np

class BaseModel(pl.LightningModule):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self._set_hparams(config)
        self.get_model()
        self.normalization_layer = torch_functions.NormalizationLayer(normalize_scale=4.0, learn_scale=True)
        self.soft_triplet_loss = torch_functions.TripletLoss()
    
    def get_model(self):
        raise NotImplementedError
        
    def forward(self,x):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer=torch.optim.SGD(self.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        return optimizer
    
    def training_step(self,train_batch,batch_idx):
        raise NotImplementedError
        
    def validation_step(self,val_batch,batch_idx):
        raise NotImplementedError
    
    def compute_loss(self,ske_img,imgs_target,soft_triplet_loss=True):
        ske_img=self.normalization_layer(ske_img)
        imgs_target=self.normalization_layer(imgs_target)

        assert (ske_img.shape[0] == imgs_target.shape[0] and
                ske_img.shape[1] == imgs_target.shape[1])

        if soft_triplet_loss:
            return self.compute_soft_triplet_loss_(ske_img,imgs_target)
        else:
            return self.compute_batch_based_classification_loss_(ske_img,imgs_target)

    def compute_soft_triplet_loss_(self, mod_img1, img2):
        triplets = []
        labels = list(range(mod_img1.shape[0])) + list(range(img2.shape[0]))
        for i in range(len(labels)):
            triplets_i = []
            for j in range(len(labels)):
                if labels[i] == labels[j] and i != j:
                    for k in range(len(labels)):
                        if labels[i] != labels[k]:
                            triplets_i.append([i, j, k])
            np.random.shuffle(triplets_i)
            triplets += triplets_i[:3]
        assert (triplets and len(triplets) < 2000)
        return self.soft_triplet_loss(torch.cat([mod_img1, img2]), triplets)

    def compute_batch_based_classification_loss_(self, mod_img1, img2):
        x = torch.mm(mod_img1, img2.transpose(0, 1))
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()
        return F.cross_entropy(x, labels)
        
    


