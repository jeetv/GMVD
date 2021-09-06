import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Loss(nn.Module):

    def __init__(self, loss_select, k, p):
        super().__init__()
        self.loss_select = loss_select
        self.p = p
        self.k = k

    def kldiv(self, s_map, gt):
        batch_size = s_map.size(0)
        w = s_map.size(1)
        h = s_map.size(2)
        sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
        expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)
    
        assert expand_s_map.size() == s_map.size()

        sum_gt = torch.sum(gt.view(batch_size, -1), 1)
        expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)
        
        assert expand_gt.size() == gt.size()

        s_map = s_map/(expand_s_map*1.0)
        gt = gt / (expand_gt*1.0)

        s_map = s_map.view(batch_size, -1)
        gt = gt.view(batch_size, -1)

        eps = 2.2204e-16
        result = gt * torch.log(eps + gt/(s_map + eps))
        # print(torch.log(eps + gt/(s_map + eps))   )
        return torch.mean(torch.sum(result, 1))

    def cc(self, s_map, gt):
        batch_size = s_map.size(0)
        w = s_map.size(1)
        h = s_map.size(2)
    
        mean_s_map = torch.mean(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
        std_s_map = torch.std(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    
        mean_gt = torch.mean(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
        std_gt = torch.std(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    
        s_map = (s_map - mean_s_map) / std_s_map
        gt = (gt - mean_gt) / std_gt
    
        ab = torch.sum((s_map * gt).view(batch_size, -1), 1)
        aa = torch.sum((s_map * s_map).view(batch_size, -1), 1)
        bb = torch.sum((gt * gt).view(batch_size, -1), 1)
    
        return torch.mean(ab / (torch.sqrt(aa*bb)))
        

    def forward(self, x, target, kernel):
        target = self.target_transform_(x, target, kernel)
        if self.loss_select == 'klcc':
            #### KL div + cc ####
            return (self.k*self.kldiv(x[0], target[0])) - (self.p*self.cc(x[0], target[0]))
            ################################################
        elif self.loss_select == 'mse':
            return F.mse_loss(x, target)

    def target_transform_(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            if self.loss_select == 'klcc':
                #### Use Sigmoid for KLDiv+cc ####
                target = torch.sigmoid(F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2)))
                ############################################
            elif self.loss_select == 'mse':
                target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
        return target
