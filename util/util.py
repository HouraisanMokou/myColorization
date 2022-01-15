import time

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
        #self.l1 = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()
        self.opt = opt

    def forward(self, fake_dis, fake_ab, real_ab):
        l1 = torch.mean(torch.sum(torch.abs(fake_ab-real_ab), dim=1, keepdim=True))#self.l1(fake_ab, real_ab)
        en = encode(real_ab, self.opt).long()
        ce = self.ce(fake_dis, en[:, 0, :, :])
        return ce + 10 * l1, l1, ce


def eval(model, dataloader, opt,L):
    model.eval()
    loss_l1_list = []
    loss_ce_list = []
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

    used_time = time.time() - t1
    mean_l1_loss = np.mean(loss_l1_list)
    mean_ce_loss = np.mean(loss_ce_list)
    return used_time, mean_l1_loss, mean_ce_loss
