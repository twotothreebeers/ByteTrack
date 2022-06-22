from loguru import logger

import tensorrt as trt
import torch
from torch2trt import torch2trt
from enum import Enum

from yolox.exp import get_exp

import argparse
import os
import shutil

from yolox.models.yolo_head import YOLOXHead
from yolox.models.yolo_pafpn import YOLOPAFPN
from yolox.models.yolox import YOLOX

class Model(Enum):
    S = "yolox-s"
    M = "yolox-m"

STD = (0.229, 0.224, 0.225)
RGB_MEANS = (0.485, 0.456, 0.406)
NUM_CLASSES = 1

param_dict = {
    # "yolox-nano": (0.33, 0.25),
    # "yolox-tiny": (0.33, 0.375),
    Model.S: (0.33, 0.50),
    Model.M: (0.67, 0.75)
    # "yolox-l": (1.0, 1.0),
    # "yolox-x": (1.33, 1.25),
}

def _init_yolo(M):
    for m in M.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03


def make_parser():
    parser = argparse.ArgumentParser("YOLOX ncnn deploy")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        help="output",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    # exp = get_exp(args.exp_file, args.name)
    # if not args.experiment_name:
    #     args.experiment_name = exp.exp_name

    # model = exp.get_model()
    # file_name = os.path.join(exp.output_dir, args.experiment_name)
    # os.makedirs(file_name, exist_ok=True)
    # if args.ckpt is None:
    #     ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
    # else:
    #     ckpt_file = args.ckpt

    # ckpt = torch.load(ckpt_file, map_location="cpu")
    # # load the model state dict

    # model.load_state_dict(ckpt["model"])
    # logger.info("loaded checkpoint done.")
    # model.eval()
    # model.cuda()
    model = Model(args.name)
    num_classes = 1
    kwargs = {}
    depth, width = param_dict[model]

    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, **kwargs)
    head = YOLOXHead(num_classes, width, in_channels=in_channels, **kwargs)
    
    model = YOLOX(backbone, head)

    model.apply(_init_yolo)
    model.head.initialize_biases(1e-2)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.cuda(torch.device('cuda'))
    model.eval()
    model.head.decode_in_inference = False
    
#    x = torch.ones(2, 3, 480, 640).cuda()
    x = torch.ones(1, 3, 480, 640).cuda()
    model_trt = torch2trt(
        model,
        [x],
        fp16_mode=True,
        log_level=trt.Logger.INFO,
        max_workspace_size=(1 << 32),
        max_batch_size=2
    )
    file_name = args.output

    torch.save(model_trt.state_dict(), os.path.join(file_name, "model_trt.pth"))
    logger.info("Converted TensorRT model done.")
    engine_file = os.path.join(file_name, "model_trt.engine")
    engine_file_demo = os.path.join("deploy", "TensorRT", "cpp", "model_trt.engine")
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())

    shutil.copyfile(engine_file, engine_file_demo)

    logger.info("Converted TensorRT model engine file is saved for C++ inference.")


if __name__ == "__main__":
    main()
