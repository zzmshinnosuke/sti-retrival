#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2021-06-30 15:25:29
# @Author: zzm

from .base_dataset import BaseDataset as Dataset

import os
import PIL

class StiSketchImageDataset(Dataset):
    
    def __getitem__(self,idx):
        out={}
        file_name=self.files[idx].split('.')[0]   
        out['id']=idx
        out['source_ske_name']=file_name+'.png'
        out['source_ske_img_data']=self.get_img(os.path.join(self.root_path,"sketch-png",out['source_ske_name']))
        out['target_img_name']=file_name+'.jpg'
        out['target_img_data']=self.get_img(os.path.join(self.root_path,"image",out['target_img_name']))
        return out
    
    def get_img(self,img_path,raw_img=False):
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        if raw_img:
            return img
        if self.transform:
            img = self.transform(img)     
        return img
        