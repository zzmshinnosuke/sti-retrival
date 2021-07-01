#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2021-06-30 10:39:53
# @Author: zzm

from sketch.datasets import get_loader
from sketch.models import get_model
from sketch.configs import get_parser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pprint import pprint 

import os

if __name__ == '__main__':
    parser=get_parser()
    args=parser.parse_args()
    
    dataset=dict()
    for split in ['train','test']:
        dataset[split]=get_loader(args,split)
    
    model=get_model(args)
    pprint(args)
    
    logger=TensorBoardLogger(save_dir=args.log_dir,
                             name=args.note)
    trainner=pl.Trainer(gpus=1,
                        max_epochs=args.n_epoch,
                        logger=logger)
    
    trainner.fit(model,dataset['train'],dataset['test'])
