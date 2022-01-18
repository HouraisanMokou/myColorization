import logging
import os

import time

import numpy as np
import torch

import cv2
from model.SIGGRAPH import SIGGRAPH
from model.ECCV import ECCV
from model.ADVANCED import SIGRES
from util import opt, util
from util.dataset.Dataset import ImageSet

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

if __name__ == '__main__':
    opt = opt.get_parser()
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    file_handler = logging.FileHandler(opt.logging_file_name)
    file_handler.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.info(
        "{}: start to logging\n".format(
            time.strftime("%Y.%m.%d_%H:%M:%S", time.localtime())
        )
    )

    # TODO: assign the test directories
    path = os.path.join(opt.dataset_path, 'small')
    data3 = os.listdir(path)
    dataset = ImageSet(data3, opt, 'val', path)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=opt.batch_size, num_workers=8)

    model = eval(opt.model)(4, 2).to(opt.device)
    # model.load_state_dict(torch.load('/data/11912716/myColor/checkpoint/epoch_35.pth'))
    # TODO: assign the test epoch
    model.load_state_dict(torch.load(
        os.path.join(opt.checkpoints_prefix, 'epoch_43.pth')))  # /data/11912716/myColor/checkpoint/epoch_31.pth

    # # TODO: to test the time, loss, acc, variance
    used_time, mean_l1_loss, mean_ce_loss, mean_acc,mean_var_a,mean_var_b = util.evalu(model, dataloader, opt, 0.98)
    print(used_time, mean_l1_loss, mean_ce_loss, mean_acc,mean_var_a,mean_var_b)

    # TODO: to show picture
    clahe=cv2.createCLAHE(2,(8,8))
    util.visual_eval(model,dataloader,opt,clahe)