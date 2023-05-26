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



# CUSTOM DATASET:

@DATASETS.register_module()
class MathAugmentationDataset(CustomDataset):

    CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '=', 'c1')

    def load_annotations(self, ann_file):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file
        image_list = mmcv.list_from_file(self.ann_file)
    
        data_infos = []
        # convert annotations to middle format
        for image_id in image_list:
            filename = f'{self.img_prefix}/{image_id}.png'
            image = mmcv.imread(filename)
            height, width = image.shape[:2]
    
            data_info = dict(filename=f'{image_id}.png', width=width, height=height)
    
            # load annotations
            label_prefix = self.img_prefix.replace('images', 'labels')
            lines = mmcv.list_from_file(osp.join(label_prefix, f'{image_id}.txt'))
            content = [line.strip().split(' ') for line in lines]

            bbox_names = []
            for x in content:
                label = x[0]
                if label == "10":
                    label = "+"
                elif label == "11":
                    label = "-"
                elif label == "12":
                    label = "="
                elif label == "13":
                    label = "c1"
                bbox_names.append(label)
            bboxes = [convert_to_bb(x[1:], width, height) for x in content]
    
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
    
            # filter 'DontCare'
            for bbox_name, bbox in zip(bbox_names, bboxes):
                if bbox_name in cat2label:
                    gt_labels.append(cat2label[bbox_name])
                    gt_bboxes.append(bbox)
                else:
                    gt_labels_ignore.append(-1)
                    gt_bboxes_ignore.append(bbox)

            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.longlong),
                bboxes_ignore=np.array(gt_bboxes_ignore,
                                       dtype=np.float32).reshape(-1, 4),
                labels_ignore=np.array(gt_labels_ignore, dtype=np.longlong))

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos
    

# CONFIGS:
    
cfg = Config.fromfile('./configs_math/fcos_r50_caffe_fpn_gn-head_1x_coco-size320-300eps.py')

# Modify dataset type and path
cfg.dataset_type = 'MathAugmentationDataset'
cfg.data_root = 'math_augmentation_mmdet/'

cfg.data.test.type = 'MathAugmentationDataset'
cfg.data.test.data_root = 'math_augmentation_mmdet/'
cfg.data.test.ann_file = 'train.txt'
cfg.data.test.img_prefix = 'data/images/train'

cfg.data.train.type = 'MathAugmentationDataset'
cfg.data.train.data_root = 'math_augmentation_mmdet/'
cfg.data.train.ann_file = 'train.txt'
cfg.data.train.img_prefix = 'data/images/train'

cfg.data.val.type = 'MathAugmentationDataset'
cfg.data.val.data_root = 'math_augmentation_mmdet/'
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


print("\n\n\n ======== BEGIN TRAINING ========:\n\n\n")

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(cfg.model)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)