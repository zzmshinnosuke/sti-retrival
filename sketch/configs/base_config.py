#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2021-06-30 10:29:07
# @Author: zzm

import argparse
import os

def get_parser():
    parser=argparse.ArgumentParser()
    
    #dataset
    parser.add_argument('--dataset',
                        required=True,
                        help='the dataset type')
    
    parser.add_argument('--dataset_root_path',
                        required=True,
                        help='the root path for dataset')
    
    parser.add_argument('--loader_num_workers',
                        type=int,
                        default=10,
                        help='the number of loader workers')
    
    parser.add_argument('--shuffle',
                        type=bool,
                        default=True,
                        help='shuffle')
    
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='the batch size')
    
    #model
    parser.add_argument('--model',
                        required=True,
                        help='the model type')

    parser.add_argument('--embed_dim',
                        type=int,
                        default=512,
                        help='output feature dim')
    
    parser.add_argument('--n_epoch',
                        type=int,
                        default=100,
                        help='the max epoch number')
      
    parser.add_argument('--dropout',
                        type=float,
                        default=0.1,
                        help='the dropout ratio')  
    
    parser.add_argument('--model_opt',
                        default="",
                        help="the extra model option (eg: --model_opt '--a 10 --b 20')")
    
    #log
    parser.add_argument('--log_dir',
                        default=os.path.expanduser('~/tmp/logs/sti-retrival'),
                        help='the log directory')
    
    parser.add_argument('--note',
                        required=True,
                        help='the experiment name')
    
    #optimizer
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.1,
                        help='optimizer learning_rate')
    
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='optimizer momentum')
    
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.00005,
                        help='optimizer weight_decay')

    #lr_scheduler:
    parser.add_argument('--learning_rate_decay_frequency',
                        type=int,
                        default=30,
                        help='lr_scheduler learning_rate_decay_frequency')
    
    parser.add_argument('--learning_rate_factor',
                        type=float,
                        default=0.5,
                        help='lr_scheduler learning_rate_factor')
    
    return parser
