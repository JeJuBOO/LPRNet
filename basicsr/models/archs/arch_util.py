# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from basicsr.utils import get_root_logger

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data = m.weight.data * scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data = m.weight.data *scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x,
              flow,
              interp_mode='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(x),
        torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow,
                size_type,
                sizes,
                interp_mode='bilinear',
                align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(
            f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow,
        size=(output_h, output_w),
        mode=interp_mode,
        align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

# handle multiple input
class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

import time
def measure_inference_speed(model, data, max_iter=200, log_interval=50):
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i in range(max_iter):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(*data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    return fps

def check_image_size(x, window_size):        
    _, _, H, W = x.shape
    pad_h = (window_size[0] - H % window_size[0]) % window_size[0]
    pad_w = (window_size[1] - W % window_size[1]) % window_size[1]
    x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
    return x

def patch_partition(x, patch_size, overlap=0):
    """
    Partition image into patches. Automatically append edge patches if needed.

    Returns:
        patches: (B * N, C, ph, pw)
        B: Batch size
        num_main_patches: number of patches per image from unfold (used for fold)
    """
    B, C, H, W = x.shape
    ph, pw = patch_size
    stride_h, stride_w = ph - overlap, pw - overlap

    # 기본 unfold 패치
    patches_unfold = F.unfold(x, kernel_size=(ph, pw), stride=(stride_h, stride_w))
    num_main_patches = patches_unfold.shape[-1]
    patches = patches_unfold.transpose(1, 2).reshape(B * num_main_patches, C, ph, pw)

    # H와 W가 stride로 나눠지는지 확인
    need_h = (H - ph) % stride_h != 0
    need_w = (W - pw) % stride_w != 0

    # 경계 패치 추가
    if need_h or need_w:
        extra_patches = []

        for b in range(B):
            img = x[b]

            # 오른쪽 경계
            if need_w:
                for top in range(0, H - ph + 1, stride_h):
                    patch = img[:, top:top+ph, W - pw:W]
                    extra_patches.append(patch)

            # 아래쪽 경계
            if need_h:
                for left in range(0, W - pw + 1, stride_w):
                    patch = img[:, H - ph:H, left:left+pw]
                    extra_patches.append(patch)

            # 오른쪽 아래 모서리
            if need_h and need_w:
                patch = img[:, H - ph:H, W - pw:W]
                extra_patches.append(patch)

        if extra_patches:
            extra_patches = torch.stack(extra_patches, dim=0)
            patches = torch.cat([patches, extra_patches], dim=0)

    return patches, B, num_main_patches

def patch_reverse(patches, H, W, B, num_main_patches, overlap=0):
    """
    Reverse patches into full image. Supports boundary patches appended separately.
    """
    B_C_patches, C, ph, pw = patches.shape
    stride_h, stride_w = ph - overlap, pw - overlap

    patches_main = patches[:B * num_main_patches]
    patches_reshaped = patches_main.view(B, num_main_patches, C * ph * pw).transpose(1, 2)

    x = F.fold(patches_reshaped, output_size=(H, W),
            kernel_size=(ph, pw), stride=(stride_h, stride_w))
    norm_map = F.fold(torch.ones_like(patches_reshaped), output_size=(H, W),
                    kernel_size=(ph, pw), stride=(stride_h, stride_w))


    # 남은 경계 패치
    patches_extra = patches[B * num_main_patches:]
    ptr = 0

    for b in range(B):
        # 오른쪽 경계
        if (W - pw) % stride_w != 0:
            for top in range(0, H - ph + 1, stride_h):
                x[b, :, top:top+ph, W - pw:W] += patches_extra[ptr]
                norm_map[b, :, top:top+ph, W - pw:W] += 1.0
                ptr += 1

        # 아래쪽 경계
        if (H - ph) % stride_h != 0:
            for left in range(0, W - pw + 1, stride_w):
                x[b, :, H - ph:H, left:left+pw] += patches_extra[ptr]
                norm_map[b, :, H - ph:H, left:left+pw] += 1.0
                ptr += 1

        # 오른쪽 아래 모서리
        if (H - ph) % stride_h != 0 and (W - pw) % stride_w != 0:
            x[b, :, H - ph:H, W - pw:W] += patches_extra[ptr]
            norm_map[b, :, H - ph:H, W - pw:W] += 1.0
            ptr += 1

    x = x / norm_map.clamp(min=1e-6)
    x = x[:, :, :H, :W]
    return x


class fft_bench_complex_conv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False):
        super(fft_bench_complex_conv, self).__init__()
        self.act_fft = act_method

        hid_dim = int(dim * dw)

        self.complex_conv1 = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class CustomAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super(CustomAdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        out_height, out_width = self.output_size

        # 스트라이드 계산
        stride_h = height // out_height
        stride_w = width // out_width

        # 커널 사이즈 계산
        kernel_h = height - (out_height - 1) * stride_h
        kernel_w = width - (out_width - 1) * stride_w

        # 평균 풀링 수행
        out = F.avg_pool2d(x, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))

        return out

    def extra_repr(self):
        return f'output_size={self.output_size}'
    