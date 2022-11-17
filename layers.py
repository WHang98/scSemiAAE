import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()#super() 函数是用于调用父类(超类)的一个方法

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):#gamma: weight of clustering loss
        eps = 1e-10#eps为常量，其值为1e-10。其中1e-10是指数形式表示法，表示的值为1*10^-10
        #scale_factor = scale_factor[:, None]#定义整个应用程序的全局比例因子
        #mean = mean * scale_factor
        
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)# 计算自然对数的伽玛参数的绝对值
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x.float(), 1e-8), zero_case, nb_case)
        
        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge
        
        result = torch.mean(result)
        return result


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma#噪声
    
    def forward(self, x):
        if self.training:
            x = x + self.sigma * torch.randn_like(x)#返回一个和输入大小相同的张量，其由均值为0、方差为1的标准正态分布填充。
        return x


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)#clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)
