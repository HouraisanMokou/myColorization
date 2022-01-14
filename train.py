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

    path=os.path.join(opt.dataset_path, 'train')
    data = os.listdir(path)
    dataset = ImageSet(data, opt, 'train',path)
    path = os.path.join(opt.dataset_path, 'test')
    data2 = os.listdir(path)
    testset = ImageSet(data2, opt, 'test',path)
    path = os.path.join(opt.dataset_path, 'val')
    data3 = os.listdir(path)
    valset = ImageSet(data3, opt, 'val',path)

    dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, num_workers=8)
    testloader = DataLoader(testset, shuffle=False, batch_size=opt.batch_size, num_workers=8)
    valloader = DataLoader(valset, shuffle=False, batch_size=opt.batch_size, num_workers=8)

    logger.info(f'dataset has {len(dataset)} pics in train set, {len(testset)} pics in train set, {len(valset)} pics in train set')

    model = eval(opt.model)(4, 2).to(opt.device)
    if opt.start_epoch > -1:
        try:
            model.load_state_dict(torch.load(
                os.path.join(opt.checkpoints_prefix, f'epoch_{opt.start_epoch}')
            ))
            logger.info('load model success')
        except:
            raise 'no such model'
    L = util.Loss(opt).to(opt.device)

    optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)

    # imgs_epoch = []
    results = defaultdict(list)

    for e in range(opt.start_epoch + 1, opt.start_epoch + 1 + opt.epoch):
        # new_imgs = []
        model.train()
        loss_list = []
        t1 = time.time()
        for data in tqdm(dataloader,
                         leave=False,
                         desc="epoch {}".format(e),
                         mininterval=0.1,
                         ncols=100,
                         total=len(dataloader),
                         ):
            data = data.to(opt.device)
            out_class, out_reg = model(data[:, [0], :, :])
            # fake_dis,fake_ab,real_ab
            loss = L(out_class, out_reg, data[:, 1:, :, :].detach())[0]
            loss_list.append(loss.cpu().detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # pre_ab = out_reg.detach()
            # imgs = torch.cat([data[:, [0], :, :].detach() * 255,
            #                   pre_ab[:, [0], :, :] * 255,
            #                   pre_ab[:, [1], :, :] * 255], dim=1).permute(0, 2, 3, 1).cpu().numpy()
            # for idx in range(imgs.shape[0]):
            #     new_imgs.append(imgs[idx, :, :, :])
        used_time = time.time() - t1
        mean_loss = np.mean(loss_list)
        torch.save(model.state_dict(), os.path.join(opt.checkpoints_prefix, f'epoch_{opt.start_epoch}'))
        logger.info('training stage: epoch [{:3d}] [ used_time: [{:<10f}] mean-loss: [{:<10f}]]'.format(e, used_time,
                                                                                                        mean_loss))
        results['times_train'].append(used_time)
        results['mean_losses_train'].append(mean_loss)

        if opt.test_epoch != -1 and e % opt.test_epoch == 0:
            used_time, mean_l1, mean_ce = util.eval(model, valloader, opt, L)
            logger.info(
                'validation stage: epoch [{:3d}][ used_time: [{:<10f}] mean-regression-loss: [{:<10f}]] mean-classification-loss: [{:<10f}]]'
                    .format(e, used_time, mean_l1, mean_ce))
        logger.info('')
        # imgs_epoch.append(new_imgs)

    used_time, mean_l1, mean_ce = util.eval(model, valloader, opt, L)
    logger.info(
        'test stage:[ used_time: [{:<10f}] mean-regression-loss: [{:<10f}]] mean-classification-loss: [{:<10f}]]' \
            .format(used_time, mean_l1, mean_ce))

    # for idx,imgs in enumerate(imgs_epoch):
    # for idx,imgs in enumerate([imgs_epoch[-1]]):
    #     for img in imgs:
    #         im = img.astype('uint8')
    #
    #         img = cv2.cvtColor(im, cv2.COLOR_Lab2BGR)
    #         cv2.imwrite(f'test-{idx}-{random.random()}.jpg', img)
