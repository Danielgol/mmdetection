import os
import mmcv
import asyncio
from argparse import ArgumentParser
from mmdet.apis import (inference_detector, init_detector, show_result_pyplot)


# i.e:
# python predict.py --cfg ./configs_math/fcos_r50_caffe_fpn_gn-head_1x_coco-size320-300eps.py --ckpt checkpoints/fcos_epoch_300.pth --source ./math_augmentation_mmdet/data/images/test --save ./output --conf 0.3


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='weights path')
    parser.add_argument('--source', type=str, required=True, help='images path')
    parser.add_argument('--save', type=str, required=True, default='./output', help='save folder')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--conf', type=float, default=0.3, help='conf threshold')
    opt = parser.parse_args()

    conf_thresh = float(opt.conf)
    model = init_detector(opt.cfg, opt.ckpt, device=opt.device)

    # Prediction:
    for name in os.listdir(opt.source):
        print(f"predicted: {name}")
        img = mmcv.imread(os.path.join(opt.source, name))
        width = img.shape[1]
        height = img.shape[0]

        mmcv.mkdir_or_exist(os.path.abspath(opt.save))

        result = inference_detector(model, img)
        writer = open(os.path.join(opt.save, name.split(".")[0]+".txt"), "w")

        for i, bbs in enumerate(result):
            for bb in bbs:
                label = i
                x1 = bb[0]
                y1 = bb[1]
                x2 = bb[2]
                y2 = bb[3]
                conf = bb[4]
                if conf < conf_thresh:
                    continue
                line = "{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}".format(label, conf, ((x2+x1)/2)/width, ((y2+y1)/2)/height, (x2-x1)/width, (y2-y1)/height)+"\n"
                writer.write(line)

        writer.close()
        show_result_pyplot(
            model,
            os.path.join(opt.source, name),
            result,
            palette="coco",
            score_thr=opt.conf,
            out_file=os.path.join(opt.save, name)
        )