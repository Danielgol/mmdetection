import torch
import numpy as np
import mmcv
from mmcv import Config
from mmcv.ops import get_compiling_cuda_version, get_compiler_version

print(get_compiling_cuda_version())
print(get_compiler_version())

import os.path as osp
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from importlib import import_module

def convert_to_bb(values, width, height):
    norm_x = float(values[0])
    norm_y = float(values[1])
    norm_w = float(values[2])
    norm_h = float(values[3])
    un_h = int(norm_h * height)
    un_w = int(norm_w * width)
    un_x = int((norm_x * width) - (un_w/2))
    un_y = int((norm_y * height) - (un_h/2))
    return [un_x, un_y, un_x+un_w, un_y+un_h]


dataset_type = 'MathAugmentationDataset'
data_format = import_module(f'custom_data.{dataset_type}')
    
cfg = Config.fromfile('./configs_math/fcos_r50_caffe_fpn_gn-head_1x_coco-size320-300eps.py')

# Modify dataset type and path
cfg.dataset_type = dataset_type
cfg.data_root = data_format.DATA_ROOT

cfg.data.test.type = dataset_type
cfg.data.test.data_root = data_format.DATA_ROOT
cfg.data.test.ann_file = 'train.txt'
cfg.data.test.img_prefix = 'data/images/train'

cfg.data.train.type = dataset_type
cfg.data.train.data_root = data_format.DATA_ROOT
cfg.data.train.ann_file = 'train.txt'
cfg.data.train.img_prefix = 'data/images/train'

cfg.data.val.type = dataset_type
cfg.data.val.data_root = data_format.DATA_ROOT
cfg.data.val.ann_file = 'val.txt'
cfg.data.val.img_prefix = 'data/images/val'

# modify num classes of the model in box head
cfg.model.bbox_head.num_classes = 14
# If we need to finetune a model based on a pre-trained detector, we need to
# use load_from to set the path of checkpoints.
cfg.load_from = 'checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './math_fcos_logs'

cfg.device = "cuda"

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 12
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 12

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]


# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

print("\n\n ======== BEGIN TRAINING ========:\n\n")

datasets = [build_dataset(cfg.data.train)]
model = build_detector(cfg.model)
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)