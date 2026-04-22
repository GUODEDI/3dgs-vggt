#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    elif size_average is None:
        return ssim_map  # [C, H, W]，逐像素图，供加权使用
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()

def weighted_l1_loss(network_output, gt, weight_map=None):
    """加权 L1 损失函数
    
    Args:
        network_output: 网络输出 [3, H, W] 或 [C, H, W]
        gt: 真实值 [3, H, W] 或 [C, H, W]
        weight_map: 权重图 [1, H, W] 或 [H, W]，可选
    
    Returns:
        加权 L1 损失
    """
    if weight_map is None:
        return torch.abs((network_output - gt)).mean()
    
    # 确保 weight_map 是 [1, H, W] 格式
    if weight_map.ndim == 2:
        weight_map = weight_map[None]  # [1, H, W]
    
    # 扩展到与 network_output 相同的通道数
    if network_output.shape[0] == 3:
        weight_map = weight_map.repeat(3, 1, 1)  # [3, H, W]
    
    # 计算加权 L1 损失
    weighted_diff = torch.abs(network_output - gt) * weight_map
    weight_sum = weight_map.sum()
    
    if weight_sum > 1e-8:
        return weighted_diff.sum() / weight_sum
    else:
        return torch.abs((network_output - gt)).mean()

def weighted_ssim(img1, img2, weight_map=None, window_size=11):
    """加权 SSIM 损失函数
    
    Args:
        img1: 图像1 [3, H, W]
        img2: 图像2 [3, H, W]
        weight_map: 权重图 [1, H, W] 或 [H, W]，可选
        window_size: SSIM 窗口大小
    
    Returns:
        加权 SSIM 值
    """
    if weight_map is None:
        return ssim(img1, img2, window_size)
    
    # 计算逐像素 SSIM map（size_average=None 返回 [C, H, W]）
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    ssim_map = _ssim(img1, img2, window, window_size, channel, size_average=None)  # [C, H, W]

    # 准备权重图
    if weight_map.ndim == 2:
        weight_map = weight_map[None]  # [1, H, W]

    # 扩展到与 ssim_map 相同的通道数
    if ssim_map.shape[0] == 3:
        weight_map = weight_map.repeat(3, 1, 1)  # [3, H, W]

    # 加权平均
    weight_sum = weight_map.sum()
    if weight_sum > 1e-8:
        return (ssim_map * weight_map).sum() / weight_sum
    else:
        return ssim_map.mean()
