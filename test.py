import logging
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
import cv2

from model.SIGGRAPH import SIGGRAPH
from model.ECCV import ECCV
from model.ADVANCED import SIGRES
from util import opt, util
from util.dataset.Dataset import ImageSet

from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torchvision import utils as vutils
from tqdm import tqdm

if __name__ == '__main__':
    # img=cv2.imread('/data/11912716/myColor/dataset/self_made/34786655_p0.jpg')
    # print(img.shape)

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


    path = os.path.join(opt.dataset_path, 'val')
    data3 = os.listdir(path)
    dataset = ImageSet(data3, opt, 'val',path)

    dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, num_workers=8)

    model = eval(opt.model)(4, 2).to(opt.device)
    model.load_state_dict(torch.load(os.path.join(opt.checkpoints_prefix,'epoch_60')))
    new_imgs = []
    model.eval()
    for data in tqdm(dataloader,
                     leave=False,
                     desc="testing",
                     mininterval=0.1,
                     ncols=100,
                     total=len(dataloader),
                     ):
        h,w=data[1],data[2]
        data = data[0].to(opt.device)
        out_class, out_reg = model(data[:, [0], :, :])
        # fake_dis,fake_ab,real_ab

        pre_ab = out_reg.detach()
        avg=pre_ab[:, :, :, :]*0
        for i in range(pre_ab.size()[0]):
            print(pre_ab[i, 0, :, :].max() - pre_ab[i, 0, :, :].min())
            print(pre_ab[i, 1, :, :].max() - pre_ab[i, 1, :, :].min())
        avg+=122
        avg_imgs = torch.cat([data[:, [0], :, :].detach() * 255,
                          avg[:, [0], :, :] * 255,
                          avg[:, [1], :, :] * 255], dim=1).permute(0, 2, 3, 1).cpu().numpy()
        imgs = torch.cat([data[:, [0], :, :].detach() * 255,
                          pre_ab[:, [0], :, :] * 255,
                          pre_ab[:, [1], :, :] * 255], dim=1).permute(0, 2, 3, 1).cpu().numpy()
        for idx in range(imgs.shape[0]):
            new_imgs.append((
                data[idx, :, :, :].permute(1, 2, 0).cpu().numpy()* 255,
                imgs[idx, :, :, :],
                w[idx].cpu().numpy(),h[idx].cpu().numpy(),
                avg_imgs[idx, :, :, :]
            ))
    for idx,pair in enumerate(new_imgs):
        orin=pair[0]
        fake=pair[1]
        avg=pair[4]
        for i,img in enumerate([orin,fake,avg]):
            im = img.astype('uint8')
            im = cv2.cvtColor(im, cv2.COLOR_Lab2BGR)
            im =cv2.resize(im,(pair[2],pair[3]),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(opt.result_pics_path,f'pair{idx}-{i}.jpg'), im)
    # for idx,imgs in enumerate(imgs_epoch):
    # for idx,imgs in enumerate([imgs_epoch[-1]]):
    #     for img in imgs:
    #         im = img.astype('uint8')
    #
    #         img = cv2.cvtColor(im, cv2.COLOR_Lab2BGR)
    #         cv2.imwrite(f'test-{idx}-{random.random()}.jpg', img)
