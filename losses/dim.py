#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:26:16 2019

@author: nabila
"""

# import torch and mutual information
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn

from torchvision.transforms import ToTensor
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
import statistics as stats

from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage import feature
import numpy as np

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(1, 32,  kernel_size=3, stride=1,padding=0)
        self.c1 = nn.Conv2d(32, 32, kernel_size=3, stride=1,padding=0)
        self.c2 = nn.Conv2d(32, 32, kernel_size=3, stride=1,padding=0)
        self.c3 = nn.Conv2d(32, 32, kernel_size=3, stride=1,padding=0)

        self.b1 = nn.BatchNorm2d(32)
        self.b2 = nn.BatchNorm2d(32)
        self.b3 = nn.BatchNorm2d(32)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.b1(self.c1(h)))
        h = F.relu(self.b2(self.c2(h)))
        h = F.relu(self.b3(self.c3(h)))
        return h

model = Encoder()
x = torch.Tensor(1,1,128,128)
M = model(x)
print(M.shape)    
    
class LocalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(33, 64, kernel_size=1)
        self.c1 = nn.Conv2d(64, 64, kernel_size=1)
        self.c2 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)

model = LocalDiscriminator()
x = torch.Tensor(1,33,128,128)
out = model(x)
print(out.shape)  

class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(33, 64, kernel_size=3)
        self.c1 = nn.Conv2d(64, 64, kernel_size=3)
        self.l0 = nn.Linear(64 * 84 * 84, 256)
        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, 1)

    def forward(self, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = h.view(h.shape[0], -1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


model = GlobalDiscriminator()
x = torch.Tensor(1,33,88,88)
out = model(x)
print(out.shape)  

    
class PriorDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Conv2d(32, 64, kernel_size=1)
        self.l1 = nn.Conv2d(64, 64, kernel_size=1)
        self.l2 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))

model = PriorDiscriminator()
x = torch.Tensor(1,32,128,128)
out = model(x)
print(out.shape)  

    
    
class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, gamma=0.8):
        super().__init__()
        self.global_d = GlobalDiscriminator()
        self.local_d  = LocalDiscriminator()
        self.prior_d  = PriorDiscriminator()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    def forward(self, image,M, M_prime):

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        y_M       = torch.cat((M, image), dim=1)
        y_M_prime = torch.cat((M_prime, image), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(y_M)).mean()
        Em = F.softplus(self.global_d(y_M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha
        
        prior = torch.rand_like(M)

        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(M)).mean()
        PRIOR = - (term_a + term_b) * self.gamma

        return LOCAL + GLOBAL + PRIOR
    
def calc_MI1(x, y, bins=30):
    c_xy = np.histogram2d(x, y, bins)[0]
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    return mi

def calc_MI2(x, y, bins=30):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi