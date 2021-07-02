#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2021-06-30 10:29:38
# @Author: zzm

from .base_dataset import BaseDataset
from .sti_sketch_image_dataset import StiSketchImageDataset

import torch.utils.data as td

def get_dataset(config,split='train'):
    return globals()[config.dataset](config,split)

def get_loader(config,split='train'):
    dataset=get_dataset(config,split)
    loader=td.DataLoader(dataset,
                        batch_size=config.batch_size,
                        shuffle=True if split=='train' else False,
                         num_workers=config.loader_num_workers,
                        drop_last=False)
    return loader
