#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2021-06-30 17:48:24
# @Author: zzm

from .base_model import BaseModel

import torchvision
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import CSVLogger
import numpy as np

from .torch_functions import NormalizationLayer,TripletLoss

from tqdm import tqdm

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
        model.fc=torch.nn.Sequential(torch.nn.Linear(model.fc.in_features,self.config.embed_dim))
        self.img_model=model

        self.normalization_layer = NormalizationLayer(normalize_scale=4.0, learn_scale=True)
        self.soft_triplet_loss = TripletLoss()
    
    def forward(self,x):
        return self.img_model(x)
    
    def training_step(self,train_batch,batch_idx):
        ske_feat=self.forward(train_batch["source_ske_img_data"])
        img_feat=self.forward(train_batch["target_img_data"])
        loss=self.compute_loss(ske_feat,img_feat)
        self.log('train_loss',loss,on_step=True,on_epoch=True)
        return loss
    
    def validation_step(self,val_batch,batch_idx):
        ske_feat=self.forward(val_batch["source_ske_img_data"])
        img_feat=self.forward(val_batch["target_img_data"])
        loss=self.compute_loss(ske_feat,img_feat)     
        self.log('val_loss',loss,on_step=False,on_epoch=True)
        return loss
        
    def training_epoch_end(self,outs):
        self.log('learning-rate',self.optimizers().param_groups[0]['lr'])
        ac=self.acc(self.train_dataloader())
        for name,value in ac:
            self.log('acc/train_epoch_'+name,value)
        
    def validation_epoch_end(self,outs):
        ac=self.acc(self.val_dataloader())
        for name,value in ac:
            self.log('acc/val_epoch_'+name,value)

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

    def acc(self,dataloader):
        all_imgs = []
        all_queries = []

        for i,item in enumerate(dataloader):
            ske_imgs=self.forward(item['source_ske_img_data'].cuda()).data.cpu().numpy()
            imgs=self.forward(item['target_img_data'].cuda()).data.cpu().numpy()

            all_queries+=[ske_imgs]
            all_imgs+=[imgs]
        all_imgs=np.concatenate(all_imgs)
        all_queries=np.concatenate(all_queries)

        # feature normalization
        for i in range(all_queries.shape[0]):
            all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
        for i in range(all_imgs.shape[0]):
            all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

        # match test queries to target images, get nearest neighbors
        sims = all_queries.dot(all_imgs.T)
        nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]

        out=[]
        top1=0
        for k in [1,5,10,50,100]:
            r=0.0
            for i,nns in enumerate(nn_result):
                if i in nns[:k]:
                    r+=1
            r /= len(nn_result)
            out+=[('top' + str(k), r)]
            if k==1:
                top1=r

        return out