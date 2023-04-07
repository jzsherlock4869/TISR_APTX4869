import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY

from .pytorch_msssim import MSSSIM, SSIM


@LOSS_REGISTRY.register()
class MSSSIMLoss(nn.Module):
    """MSSSIM Loss from https://github.com/jorge-pessoa/pytorch-msssim/blob/master/max_ssim.py
    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0):
        super(MSSSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss_func = MSSSIM()
        
    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * (1 - self.loss_func(target, pred))


@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    """SSIM Loss from https://github.com/jorge-pessoa/pytorch-msssim/blob/master/max_ssim.py
    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss_func = SSIM()
        
    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * (1 - self.loss_func(target, pred))
