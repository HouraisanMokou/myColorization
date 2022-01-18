import os
import time

import cv2
import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def encode(imgs, opt):
    data_ab = imgs[:, :, ::4, ::4]
    data_ab_rs = torch.round((data_ab * opt.ab_norm + opt.ab_max) / opt.ab_quant)  # normalized bin number
    data_q = data_ab_rs[:, [0], :, :] * opt.A + data_ab_rs[:, [1], :, :]
    return data_q


class Loss(nn.Module):
    def __init__(self, opt):
        super(Loss, self).__init__()
        # self.l1 = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()
        self.opt = opt

    def forward(self, fake_dis, fake_ab, real_ab):
        l1 = torch.mean(torch.sum(torch.abs(fake_ab - real_ab), dim=1, keepdim=True))  # self.l1(fake_ab, real_ab)
        en = encode(real_ab, self.opt).long()
        ce = self.ce(fake_dis, en[:, 0, :, :])
        return ce + 10 * l1, l1, ce


def evalu(model, dataloader, opt, threshold):
    L = Loss(opt).to(opt.device)
    model.eval()
    loss_l1_list = []
    loss_ce_list = []
    acc_list = []
    var_list_a=[]
    var_list_b = []
    t1 = time.time()
    for data in tqdm(dataloader,
                     leave=False,
                     desc="testing",
                     mininterval=0.1,
                     ncols=100,
                     total=len(dataloader),
                     ):
        data = data[0].to(opt.device)
        out_class, out_reg = model(data[:, [0], :, :])
        # fake_dis,fake_ab,real_ab
        loss = L(out_class, out_reg, data[:, 1:, :, :].detach())
        loss_l1, loss_ce = loss[1], loss[2]
        loss_l1_list.append(loss_l1.cpu().detach().numpy())
        loss_ce_list.append(loss_ce.cpu().detach().numpy())
        for idx in range(data.shape[0]):
            gt = data[idx, 1:, :, :].cpu().detach().numpy()
            pre = out_reg[idx, :, :, :].cpu().detach().numpy()
            abs_a, abs_b = np.abs(pre[0, :, :] - gt[0, :, :]), np.abs(pre[1, :, :] - gt[1, :, :])
            s1 = set(zip(*np.where(abs_a <= (1-threshold) * pre[0, :, :])))
            s2 = set(zip(*np.where(abs_a <= (1-threshold) * pre[1, :, :])))
            s = s1.intersection(s2)
            acc_list.append(len(s) / (data.size()[2] * data.size()[3]))
            f1,f2=pre[0,:,:].flatten(),pre[1,:,:].flatten()
            var_list_a.append(np.var(f1))
            var_list_b.append(np.var(f2))

    used_time = time.time() - t1
    mean_l1_loss = np.mean(loss_l1_list)
    mean_ce_loss = np.mean(loss_ce_list)
    return used_time, mean_l1_loss, mean_ce_loss, np.mean(acc_list),np.mean(var_list_a),np.mean(var_list_b)


def visual_eval(model, dataloader, opt,clahe):
    new_imgs = []
    model.eval()
    for data in tqdm(dataloader,
                     leave=False,
                     desc="testing",
                     mininterval=0.1,
                     ncols=100,
                     total=len(dataloader),
                     ):
        h, w = data[1], data[2]
        data = data[0].to(opt.device)
        out_class, out_reg = model(data[:, [0], :, :])
        # fake_dis,fake_ab,real_ab

        pre_ab = out_reg.detach()
        imgs = torch.cat([data[:, [0], :, :].detach() * 255,
                          pre_ab[:, [0], :, :] * 255,
                          pre_ab[:, [1], :, :] * 255], dim=1).permute(0, 2, 3, 1).cpu().numpy()
        for idx in range(imgs.shape[0]):
            new_imgs.append((
                data[idx, :, :, :].permute(1, 2, 0).cpu().numpy()*255,
                imgs[idx, :, :, :],
                w[idx].cpu().numpy(), h[idx].cpu().numpy(),
            ))
    for idx, pair in tqdm(enumerate(new_imgs),
                          leave=False,
                          desc="testing",
                          mininterval=0.1,
                          ncols=100,
                          total=len(new_imgs),
                          ):

        orin = pair[0]
        fake = pair[1]
        for i, img in [('origin', orin), ('predicted', fake)]:
            im = img.astype('uint8')
            im = cv2.cvtColor(im, cv2.COLOR_Lab2BGR)
            im = cv2.resize(im, (pair[2], pair[3]), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(opt.result_pics_path, f'pair{idx}-{i}.jpg'), im)
        im1 = fake.astype('uint8')
        im1 = cv2.cvtColor(im1, cv2.COLOR_Lab2BGR)
        im1 = cv2.resize(im1, (pair[2], pair[3]), interpolation=cv2.INTER_CUBIC)
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
        his=clahe.apply(im1[:,:,1])
        i_s = np.dstack((im1[:,:,0],his,im1[:,:,2])).astype('uint8')
        i_s = cv2.cvtColor(i_s, cv2.COLOR_HSV2BGR)
        cv2.imwrite(os.path.join(opt.result_pics_path, f'pair{idx}-histed.jpg'), i_s)