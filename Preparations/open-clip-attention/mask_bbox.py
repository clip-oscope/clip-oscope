import torch
from PIL import Image
import open_clip
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import random
mode_dataset=[("EVA02-E-14",'laion2b_s4b_b115k'),('ViT-L-14','datacomp_xl_s13b_b90k')]
model, _, preprocess = open_clip.create_model_and_transforms(mode_dataset[1][0], pretrained=mode_dataset[1][1])
model=model.to("cuda:1")
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer(mode_dataset[1][0])
import logging
%matplotlib inline
logger = logging.getLogger()
old_level = logger.level
logger.setLevel(100)

