#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2021-06-30 10:29:48
# @Author:     

from torch.utils.data import Dataset
import torchvision
import os

class BaseDataset(Dataset):
    
    def __init__(self,config,split='train'):
        self.config=config
        self.split=split
        self.files=[]
        self.root_path=self.config.dataset_root_path
        self.load_files()
        size=224
        self.transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(size),
                torchvision.transforms.CenterCrop(size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ])
    
    def load_files(self):
        assert self.split in ['train','val','test'],'unknown split {}'.format(self.split)
        
        filename="train_names.txt" if self.split=="train" else 'test_names.txt'
        filepath=os.path.join(self.root_path,filename)
        assert os.path.exists(filepath),'not find {}'.format(filepath)
        
        with open(filepath) as f:
            self.files=[line.strip() for line in f.readlines()]
        assert len(self.files)>0,'no json file find in {}'.format(self.root_path)
    
    def __getitem__(self,index):
        pass
    
    def __len__(self):
        return len(self.files)