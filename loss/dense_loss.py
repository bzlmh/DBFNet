import torch
import torch.nn as nn
import numpy as np
from models.dfdb.Discriminator import Discriminator
import torch.optim as optim
import functools
import torch.nn.functional as F
def jaccard_loss(input, target):
    smooth = 1e-6
    intersection = torch.sum(input * target)
    union = torch.sum(input) + torch.sum(target) - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    return 1 - jaccard


def d_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)

    a = torch.sum(input * target, dim=1)
    b = torch.sum(input * input, dim=1) + 0.001
    c = torch.sum(target * target, dim=1) + 0.001
    epsilon = 0.0001  # 平滑项，避免零除以零的情况
    d = (2 * a) / (b + c + epsilon)
    dice_loss = torch.mean(d)
    return 1 - dice_loss


class Loss_Doc(nn.Module):
    def __init__(self, model, ganLoss=True):
        super(Loss_Doc, self).__init__()
        self.model = model  # 保存模型
        self.ganLoss = ganLoss
        self.l1 = nn.L1Loss()
        self.cross_entropy = nn.BCELoss()
    def forward(self, ori,mid_out, gt, mid_outP2, mid_outP3, mid_outP4,mid_outP5):
        # 计算 L1 损失
        fin_l1_loss = self.l1(mid_out, gt)
        P2_l1loss = self.l1(mid_outP2, gt)
        P3_l1loss = self.l1(mid_outP3, gt)
        P4_l1loss = self.l1(mid_outP4, gt)
        P5_l1loss = self.l1(mid_outP5, gt)
        # 计算交叉熵损失
        fin_BCEloss = self.cross_entropy(mid_out, gt)
        P2_BCEloss = self.cross_entropy(mid_outP2, gt)
        P3_BCEloss = self.cross_entropy(mid_outP3, gt)
        P4_BCEloss = self.cross_entropy(mid_outP4, gt)
        P5_BCEloss = self.cross_entropy(mid_outP5, gt)
        # 计算 Dice 损失
        fin_dice_loss = d_loss(mid_out, gt)
        P2_dice_loss = d_loss(mid_outP2, gt)
        P3_dice_loss = d_loss(mid_outP3, gt)
        P4_dice_loss = d_loss(mid_outP4, gt)
        P5_dice_loss = d_loss(mid_outP5, gt)
        # 计算 Jaccard Loss
        fin_jaccard_loss = jaccard_loss(mid_out, gt)
        P2_jaccard_loss = jaccard_loss(mid_outP2, gt)
        P3_jaccard_loss = jaccard_loss(mid_outP3, gt)
        P4_jaccard_loss = jaccard_loss(mid_outP4, gt)
        P5_jaccard_loss = jaccard_loss(mid_outP5, gt)
        # 计算 middle L1 Loss
        middle_l1_loss = P2_l1loss + 0.1 * P3_l1loss + 0.01 * P4_l1loss+0.001*P5_l1loss
        # 计算 middle BCE Loss
        middle_BCE_loss = P2_BCEloss + 0.1* P3_BCEloss + 0.01*P4_BCEloss+0.001*P5_BCEloss
        # 计算 middle Dice Loss
        middle_dice_loss = P2_dice_loss + 0.1*P3_dice_loss + 0.01*P4_dice_loss+0.001*P5_dice_loss

        # 计算 middle Jaccard Loss
        middle_jaccard_loss = P2_jaccard_loss + 0.1*P3_jaccard_loss + 0.01*P4_jaccard_loss+0.001*P5_jaccard_loss


        mid_loss=middle_l1_loss+middle_BCE_loss+middle_jaccard_loss+middle_dice_loss



        return fin_l1_loss, fin_BCEloss, fin_dice_loss, fin_jaccard_loss, mid_loss
