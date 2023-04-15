import os
import torch
import numpy as np
from omegaconf import OmegaConf
import argparse
from load_data import *
from utils import *
from train import *
from inference import *
import random

# transformers, sklearn, OmegaConf

if __name__ =='__main__':
    ## Reset the Memory
    torch.cuda.empty_cache()
    ## parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config')
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f'./config/{args.config}.yaml')

    ## set seed
    set_seed(cfg.train.seed)

    ## train
    if cfg.train.train_mode:
        print('------------------- train start -------------------------')
        train(cfg)


    ## inference
    if cfg.test.test_mode:
        print('--------------------- test start ----------------------')
        test(cfg)

    print('----------------- Finish! ---------------------')
