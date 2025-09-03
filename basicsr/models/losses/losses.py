# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class FFTLoss(nn.Module):
    """L1 loss in frequency domain with FFT.

    Args:
        loss_weight (float): Loss weight for FFT loss. Default: 1.0.
        reductio (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.n
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (..., C, H, W). Predicted tensor.
            target (Tensor): of shape (..., C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (..., C, H, W). Element-wise
                weights. Default: None.
        """

        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        return self.loss_weight * l1_loss(pred_fft, target_fft, weight, reduction=self.reduction)
    

class CharbonnierLoss(nn.Module):
    """Charbonnier loss (robust L1 loss with epsilon smoothing).

    Args:
        loss_weight (float): Loss weight for Charbonnier loss. Default: 1.0.
        eps (float): A small value to avoid division by zero in the loss calculation. Default: 1e-3.
        reductio (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.n
    """
    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Unsupported reduction mode: {reduction}")
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (..., C, H, W). Predicted tensor.
            target (Tensor): of shape (..., C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (..., C, H, W). Element-wise
                weights. Default: None.
        """
        loss = charbonnier_loss(pred, target, weight=weight, reduction=self.reduction, eps=self.eps)
        return self.loss_weight * loss

    
class EdgeLoss(nn.Module):
    """
    Edge loss: Charbonnier loss on Laplacian (edge) maps of images.

    Args:
        loss_weight (float): Loss weight for Charbonnier loss. Default: 0.05
        eps (float): A small value to avoid division by zero in the loss calculation. Default: 1e-3.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """
    def __init__(self, loss_weight=0.05, reduction='mean', eps=1e-3):
        super(EdgeLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Unsupported reduction mode: {reduction}")
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps
        kernel_1d = torch.tensor([0.05, 0.25, 0.4, 0.25, 0.05], dtype=torch.float32)
        gauss2d = torch.outer(kernel_1d, kernel_1d)  # shape (5,5)
        self.register_buffer('gauss_kernel', gauss2d.unsqueeze(0).unsqueeze(0))  # shape (1,1,5,5)

    def _laplacian_edges(self, x):
        N, C, H, W = x.shape
        pad = 2  
        kernel = self.gauss_kernel.expand(C, 1, 5, 5).to(x.device, x.dtype)

        x_blur = F.conv2d(F.pad(x, (pad, pad, pad, pad), mode='replicate'), kernel, groups=C)
        
        x_down = x_blur[:, :, ::2, ::2]# Downsample by 2
        up_H, up_W = x_blur.shape[2], x_blur.shape[3]  # should be H, W
        x_up = torch.zeros_like(x_blur)# Upsample by 2 (insert zeros)
        x_up[:, :, ::2, ::2] = x_down * 4.0  # scale by 4 to compensate for area reduction
        
        x_blur2 = F.conv2d(F.pad(x_up, (pad, pad, pad, pad), mode='replicate'), kernel, groups=C)
        lap = x - x_blur2
        return lap

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (..., C, H, W). Predicted tensor.
            target (Tensor): of shape (..., C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (..., C, H, W). Element-wise
                weights. Default: None.
        """
        pred_edge = self._laplacian_edges(pred)
        gt_edge = self._laplacian_edges(target)
        loss = charbonnier_loss(pred_edge, gt_edge, weight=weight, reduction=self.reduction, eps=self.eps)
        return self.loss_weight * loss