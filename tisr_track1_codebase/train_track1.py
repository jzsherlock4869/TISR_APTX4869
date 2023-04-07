# flake8: noqa
import os.path as osp

import archs_mein
import data
import losses
import models
import basicsr.archs
from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)
