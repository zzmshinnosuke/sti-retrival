#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2021-06-30 10:30:08
# @Author: zzm

from .base_model import BaseModel
from .image_model import ImageModel


def get_model(config):
    return globals()[config.model](config)