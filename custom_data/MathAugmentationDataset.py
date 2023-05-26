import mmcv
import numpy as np
import os.path as osp
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

DEF_CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '=', 'c1')
DATA_ROOT = "math_augmentation_mmdet/"

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

# MUST HAVE THE FILE'S SAME NAME
@DATASETS.register_module()
class MathAugmentationDataset(CustomDataset):

    CLASSES = DEF_CLASSES

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