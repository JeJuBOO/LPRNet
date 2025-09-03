# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d, patch_partition, patch_reverse, check_image_size, \
    CustomAdaptiveAvgPool2d, fft_bench_complex_conv
from basicsr.models.archs.local_arch import Local_Base
from basicsr.models.archs.Flow_arch import KernelPrior
from basicsr.models.archs.my_module import code_extra_mean_var

import numpy as np

# -------------------------   First Stage Block -----------------------------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1))

    def forward(self, input):
        res = self.body(input)
        output = res + input
        return output

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class NAFBlock(nn.Module):
    def __init__(self, c, patch_size=None, FFTmode=False, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.patch_size = patch_size
        self.fftmode = FFTmode

        if FFTmode:
            self.fft1 = fft_bench_complex_conv(c, dw=2, act_method=nn.GELU(),bias=True)
            self.fft2 = fft_bench_complex_conv(c, dw=2, act_method=nn.GELU(),bias=True)
        cf_channel = 2 * c  # Combine Feature Channel
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        _, _, H, W = x.shape
        x = self.norm1(x)

        if self.patch_size is not None:
            x, B_out, num_main_patches = patch_partition(x, (self.patch_size, self.patch_size)) 

        if self.fftmode:
            fft_feature = self.fft1(x)
            if self.patch_size is not None:
                fft_feature = patch_reverse(fft_feature, H, W, B_out, num_main_patches)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        if self.patch_size is not None:
            x = patch_reverse(x, H, W, B_out, num_main_patches)

        if self.fftmode:
            x = fft_feature + x

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        if self.fftmode:
            fft_feature2 = self.fft2(x)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        if self.fftmode:
            x = x + fft_feature2

        x = self.dropout2(x)

        return y + x * self.gamma


class kernel_attention(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_attention, self).__init__()

        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_kernel = nn.Sequential(
                        nn.Conv2d(kernel_size*kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                        )

    def forward(self, input, kernel):
        x = self.conv_1(input)
        kernel = self.conv_kernel(kernel)
        att = torch.cat([x, kernel], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output


class NAFBlock_kernel(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=21):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_atttion = kernel_attention(kernel_size, in_ch=c, out_ch=c)

    def forward(self, inp, kernel):
        x = inp

        # kernel [B, 19*19, H, W]
        x = self.kernel_atttion(x, kernel)

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
    
class BaselineBlock(nn.Module):
    def __init__(self, c, patch_size=None, FFTmode=False, DW_Expand=1, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.patch_size = patch_size
        self.fftmode = FFTmode
        if FFTmode:
            self.fft1 = fft_bench_complex_conv(c, dw=2, act_method=nn.GELU(),bias=True)
            self.fft2 = fft_bench_complex_conv(c, dw=2, act_method=nn.GELU(),bias=True)
            
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Channel Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )

        # GELU
        self.gelu = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        _, _, H, W = x.shape
        x = self.norm1(x)

        if self.patch_size is not None:
            x, B_out, num_main_patches = patch_partition(x, (self.patch_size, self.patch_size))  # (B*P, 2*C, pH, pW)

        if self.fftmode:
            fft_feature = self.fft1(x)
            if self.patch_size is not None:
                fft_feature = patch_reverse(fft_feature, H, W, B_out, num_main_patches)

        x = self.conv1(x)

        x = self.conv2(x)
        x = self.gelu(x)
        x = x * self.se(x)
        x = self.conv3(x)

        if self.patch_size is not None:
            x = patch_reverse(x, H, W, B_out, num_main_patches)

        if self.fftmode:
            x = fft_feature + x

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        if self.fftmode:
            fft_feature2 = self.fft2(x)
        x = self.conv4(x)
        x = self.gelu(x)
        x = self.conv5(x)

        if self.fftmode:
            x = x + fft_feature2

        x = self.dropout2(x)

        return y + x * self.gamma
    

class Global_BaselineBlock(nn.Module):
    def __init__(self, c, patch_size=None, FFTmode=False, DW_Expand=1, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.patch_size = patch_size
        self.fftmode = FFTmode
        if FFTmode:
            self.fft1 = fft_bench_complex_conv(c, dw=2, act_method=nn.GELU(),bias=True)
            self.fft2 = fft_bench_complex_conv(c, dw=2, act_method=nn.GELU(),bias=True)
            
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Channel Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )

        # GELU
        self.gelu = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, g_feat):
        # device = next(self.parameters()).device
        # inp = inp.to(device)
        # g_feat = g_feat.to(device)

        x = inp
        B, C, H, W = x.shape
        x = self.norm1(x)
        
        x, B_out, num_main_patches = patch_partition(x, (self.patch_size, self.patch_size)) 
        BP, _, pH, pW = x.shape
        fft_feature = self.fft1(x)
        fft_feature = patch_reverse(fft_feature, H, W, B_out, num_main_patches)


        x = self.conv1(x)

        x = self.conv2(x)
        x = self.gelu(x)

        x = x * self.se(g_feat).repeat(BP//B, 1, 1, 1)
        x = self.conv3(x)

        if self.patch_size is not None:
            x = patch_reverse(x, H, W, B_out, num_main_patches)

        if self.fftmode:
            x = fft_feature + x

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        if self.fftmode:
            fft_feature2 = self.fft2(x)
        x = self.conv4(x)
        x = self.gelu(x)
        x = self.conv5(x)

        if self.fftmode:
            x = x + fft_feature2

        x = self.dropout2(x)

        return y + x * self.gamma
    

def generate_k(model, code, n_row=1):
    model.eval()

    # unconditional model
    # for a random Gaussian vector, its l2norm is always close to 1.
    # therefore, in optimization, we can constrain the optimization space to be on the sphere with radius of 1

    u = code  # [B, 19*19]
    samples, _ = model.inverse(u)

    samples = model.post_process(samples)

    return samples

class UpperNAFBlock(nn.Module):
    def __init__(self, c, k, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1,
                               padding=0, stride=1, groups=1, bias=True)
        self.dconv1 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=k,
                               padding=k // 2, stride=1, groups=dw_channel, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1,
                               padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1,
                      padding=0, stride=1, groups=1, bias=True),
        )
        self.sg = SimpleGate()

    def forward(self, x):
        x = self.conv1(x)
        x = self.dconv1(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv2(x)
        return x

class MDBlock(nn.Module):
    def __init__(self, c, dwconv_kernel_size, 
                 patch_size=None, 
                 NAFNetmode=False, 
                 Patchmode=False, 
                 FFTmode=False, 
                 FFN_Expand=2, drop_out_rate=0.):
        
        super().__init__()
        self.nafnetmode = NAFNetmode
        self.patchmode = Patchmode
        self.fftmode = FFTmode
        self.patch_size = patch_size

        self.norm1 = LayerNorm2d(c)
        
        if FFTmode:
            self.fft1 = fft_bench_complex_conv(c, dw=2, act_method=nn.GELU(),bias=True)
            self.fft2 = fft_bench_complex_conv(c, dw=2, act_method=nn.GELU(),bias=True)
        
        self.local_path = nn.Sequential(
            UpperNAFBlock(c, 3),
        )
    
        if not self.nafnetmode:
            cf_channel = 2 * c  # Combined Feature Channel

            self.global_path = nn.Sequential(
                UpperNAFBlock(c, dwconv_kernel_size)
            )
            self.norm2 = LayerNorm2d(cf_channel)
            self.conv1 = nn.Conv2d(in_channels=cf_channel, out_channels=c, kernel_size=1,
                                   padding=0, stride=1, groups=1, bias=True)

        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
       
        self.norm3 = LayerNorm2d(c)
        if self.patchmode:
            kn_size = 3
        else:
            kn_size = 1

        self.conv2 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=kn_size,
                               padding=kn_size//2, stride=1, groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=kn_size,
                               padding=kn_size//2, stride=1, groups=1, bias=True)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):

        x = inp
        _, _, H, W = x.shape

        x = self.norm1(x)

        if self.patchmode:
            # x = check_image_size(x, [self.patch_size, self.patch_size])  # (B, 2*C, H_pad, W_pad) 이후 H_pad=H
            x, B_out, num_main_patches = patch_partition(x, (self.patch_size, self.patch_size))  # (B*P, 2*C, pH, pW)
            # inp = x
        
        # x = self.norm1(x)

        if self.fftmode:
            fft_ft = self.fft1(x)
            if self.patchmode:
                fft_ft = patch_reverse(fft_ft, H, W, B_out, num_main_patches)

        if self.nafnetmode:
            local_features = self.local_path(x)
            feature = local_features
        else:
            local_features = self.local_path(x)
            global_features = self.global_path(x)

            combined_feature = torch.cat([local_features, global_features], dim=1)
            combined_feature = self.norm2(combined_feature)
            feature = self.conv1(combined_feature)

        if self.patchmode:
            feature = patch_reverse(feature, H, W, B_out, num_main_patches)

        if self.fftmode:
            feature = feature + fft_ft

        feature = self.dropout1(feature)
        x = inp + feature * self.beta

        x1 = self.norm3(x)
        if self.fftmode:
            fft_ft1 = self.fft2(x)
        x1 = self.conv2(x1)
        x1 = self.sg(x1)
        x1 = self.conv3(x1)

        if self.fftmode:
            x1 = x1 + fft_ft1

        x1 = self.dropout2(x1)
        output = x + x1 * self.gamma

        return output


# -------------------------   Second Stage Block -----------------------------
class SAM(nn.Module):
    def __init__(self, chan, out_chan,padder_size=64):
        super().__init__()
        img_channel = 3
        self.padder_size = padder_size
        self.sam_conv1 = nn.Conv2d(in_channels=chan, out_channels=out_chan, kernel_size=3,
                                   padding=1, stride=1, groups=1, bias=True)
        self.sam_conv3 = nn.Conv2d(in_channels=img_channel, out_channels=out_chan, kernel_size=1, bias=True)
        self.ending = nn.Conv2d(in_channels=chan, out_channels=img_channel, kernel_size=3,
                                padding=1, stride=1, groups=1, bias=True)

    def forward(self, x, x_img):
        B,_,h,w = x_img.shape
        x1 = self.sam_conv1(x)

        first_out_img = self.ending(x)[:,:,:h,:w] + x_img.detach()  # (Batch,C,H,W)

        x2 = check_image_size(first_out_img, [self.padder_size, self.padder_size])
        x2 = torch.sigmoid(self.sam_conv3(x2))  # (Batch,C,H,W)

        first_feature = x + x1 * x2

        return first_feature, first_out_img


class PatchBlock(nn.Module):
    def __init__(self, c, patch_size, FFTmode=False, drop_out_rate=0.):
        super().__init__()

        self.channel = c
        self.patch_size = patch_size
        self.fftmode = FFTmode

        if FFTmode:
            self.fft1 = fft_bench_complex_conv(c, dw=2, act_method=nn.GELU(),bias=True)
            self.fft2 = fft_bench_complex_conv(c, dw=2, act_method=nn.GELU(),bias=True)
        cf_channel = 2 * c  # Combine Feature Channel

        self.pat1_conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                                    groups=1, bias=True)
        self.pat1_dconv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=3 // 2, stride=1,
                                     groups=c, bias=True)
        self.pat1_conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                                    groups=1, bias=True)
        

        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=3 // 2, stride=1,
                               groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=3 // 2, stride=1,
                               groups=1, bias=True)

        self.combconv = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1)
        self.combconv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                                    groups=1, bias=True)
        self.combnorm = LayerNorm2d(c)

        self.norm = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.identity = nn.Identity()
        self.gelu = nn.GELU()
        # patch channel feature
        self.p_channel_interaction = nn.Sequential(
            CustomAdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 2, kernel_size=1),
            nn.BatchNorm2d(c // 2),
            nn.GELU(),
            nn.Conv2d(c // 2, c, kernel_size=1),
            nn.Sigmoid()
        )
        # 1stage global channel feature
        self.g_channel_interaction = nn.Sequential(
            CustomAdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 2, kernel_size=1),
            nn.BatchNorm2d(c // 2),
            nn.GELU(),
            nn.Conv2d(c // 2, c, kernel_size=1),
            nn.Sigmoid()
        )

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    # def forward(self, inp, first_stage_feat):
    def forward(self, x, first_stage_feat):
        device = next(self.parameters()).device
        x = x.to(device)
        first_stage_feat = first_stage_feat.to(device)
        B, _, H, W = x.shape

        inp = x

        x = self.norm(x)
        
        x, B_out, num_main_patches = patch_partition(x, (self.patch_size, self.patch_size))  # (B*P, 2*C, pH, pW)

        # inp = x

        # x = self.norm(x)

        B_P, C, pH, pW = x.shape


        if self.fftmode:
            fft_feature = self.fft1(x)
            fft_feature = patch_reverse(fft_feature, H, W, B_out, num_main_patches)

        x_pat1 = self.pat1_conv1(x)
        x_pat1 = self.pat1_dconv2(x_pat1)
        x_pat1_gelu = self.gelu(x_pat1)
        x_pat1 = self.pat1_conv3(x_pat1_gelu)
        
        patch_chan_attention = self.p_channel_interaction(x_pat1_gelu)
        global_chan = self.g_channel_interaction(first_stage_feat)

        first_stage_feat, _, _ = patch_partition(first_stage_feat, (self.patch_size, self.patch_size)) 

        first_stage_chan_result = first_stage_feat * patch_chan_attention
        patch_chan_result = x_pat1.view(B,-1,C,pH,pW) * global_chan.view(B,1,C,1,1)

        combined_ca_feature = first_stage_chan_result + patch_chan_result.view(-1,C,pH,pW)
        
        combined_ca_feature = self.combnorm(combined_ca_feature)
        combined_ca_feature = self.combconv(combined_ca_feature)
        
        combined_feature = combined_ca_feature * x_pat1_gelu

        combined_feature = self.combconv2(combined_feature)

        combined_feature = patch_reverse(combined_feature, H, W, B_out, num_main_patches)
        
        if self.fftmode:
            combined_feature = fft_feature + combined_feature

        combined_feature = self.dropout1(combined_feature)
        x = inp + combined_feature * self.beta

        x1 = self.norm2(x)
        if self.fftmode:
            fft_feature2 = self.fft2(x1)
        x1 = self.conv2(x1)
        x1 = self.gelu(x1)
        x1 = self.conv3(x1)

        if self.fftmode:
            x1 = x1 + fft_feature2

        x1 = self.dropout2(x1)
        output = x + x1 * self.gamma

        # output = patch_reverse(output, H, W)

        return output

class CA_PatchBlock(nn.Module):
    def __init__(self, c, patch_size, FFTmode=False, drop_out_rate=0.):
        super().__init__()

        self.channel = c
        self.patch_size = patch_size
        self.fftmode = FFTmode

        if FFTmode:
            self.fft1 = fft_bench_complex_conv(c, dw=2, act_method=nn.GELU(),bias=True)
            self.fft2 = fft_bench_complex_conv(c, dw=2, act_method=nn.GELU(),bias=True)
        cf_channel = 2 * c  # Combine Feature Channel

        self.pat1_conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                                    groups=1, bias=True)
        self.pat1_dconv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=3 // 2, stride=1,
                                     groups=c, bias=True)
        self.pat1_conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                                    groups=1, bias=True)
        

        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=3 // 2, stride=1,
                               groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=3 // 2, stride=1,
                               groups=1, bias=True)

        self.combconv = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1)
        self.combconv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                                    groups=1, bias=True)
        self.combnorm = LayerNorm2d(c)

        self.norm = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.identity = nn.Identity()
        self.gelu = nn.GELU()
        # patch channel feature
        self.p_channel_interaction = nn.Sequential(
            CustomAdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 2, kernel_size=1),
            nn.BatchNorm2d(c // 2),
            nn.GELU(),
            nn.Conv2d(c // 2, c, kernel_size=1),
            nn.Sigmoid()
        )
        # 1stage global channel feature
        self.g_channel_interaction = nn.Sequential(
            CustomAdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 2, kernel_size=1),
            nn.BatchNorm2d(c // 2),
            nn.GELU(),
            nn.Conv2d(c // 2, c, kernel_size=1),
            nn.Sigmoid()
        )

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    # def forward(self, inp, first_stage_feat):
    def forward(self, x, first_stage_feat):
        device = next(self.parameters()).device
        x = x.to(device)
        first_stage_feat = first_stage_feat.to(device)
        B, _, H, W = x.shape

        inp = x

        x = self.norm(x)
        
        x, B_out, num_main_patches = patch_partition(x, (self.patch_size, self.patch_size))  # (B*P, 2*C, pH, pW)

        # inp = x

        # x = self.norm(x)

        B_P, C, pH, pW = x.shape


        if self.fftmode:
            fft_feature = self.fft1(x)
            fft_feature = patch_reverse(fft_feature, H, W, B_out, num_main_patches)

        x_pat1 = self.pat1_conv1(x)
        x_pat1 = self.pat1_dconv2(x_pat1)
        x_pat1_gelu = self.gelu(x_pat1)

        x_pat1 = self.pat1_conv3(x_pat1_gelu)
        
        patch_chan_attention = self.p_channel_interaction(x_pat1_gelu)
        patch_chan_result = x_pat1 * patch_chan_attention

        global_chan = self.g_channel_interaction(first_stage_feat)
        first_stage_feat, _, _ = patch_partition(first_stage_feat, (self.patch_size, self.patch_size)) 
        first_stage_chan_result = first_stage_feat.view(B,-1,C,pH,pW) * global_chan.view(B,1,C,1,1)

        combined_ca_feature = first_stage_chan_result.view(-1,C,pH,pW) + patch_chan_result
        
        combined_ca_feature = self.combnorm(combined_ca_feature)
        combined_ca_feature = self.combconv(combined_ca_feature)
        
        combined_feature = combined_ca_feature * x_pat1_gelu

        combined_feature = self.combconv2(combined_feature)

        combined_feature = patch_reverse(combined_feature, H, W, B_out, num_main_patches)
        
        if self.fftmode:
            combined_feature = fft_feature + combined_feature

        combined_feature = self.dropout1(combined_feature)
        x = inp + combined_feature * self.beta

        x1 = self.norm2(x)
        if self.fftmode:
            fft_feature2 = self.fft2(x1)
        x1 = self.conv2(x1)
        x1 = self.gelu(x1)
        x1 = self.conv3(x1)

        if self.fftmode:
            x1 = x1 + fft_feature2

        x1 = self.dropout2(x1)
        output = x + x1 * self.gamma

        # output = patch_reverse(output, H, W)

        return output
# -------------------------      Network Part     ------------------------------

class TSMKDNet(nn.Module):

    def __init__(self, img_channel=3, width=32, first_middle_blk_num=1, first_enc_blk_nums=[], first_dec_blk_nums=[],
                 second_enc_blk_nums=[], second_dec_blk_nums=[], dwconv_kernel_size=7,
                 patch_size=64, ufp_pretrain=False, kernel_size=19):
        super().__init__()

        self.patch_size = patch_size

        # for ufp_pretrain
        self.ufp_pretrain = ufp_pretrain
        if ufp_pretrain:
            self.kernel_size = kernel_size
            self.kernel_extra = code_extra_mean_var(kernel_size)

            self.flow = KernelPrior(n_blocks=5, input_size=19 ** 2, hidden_size=25, n_hidden=1, kernel_size=19)
        self.kernel_down = nn.ModuleList()

        # run_이 붙은 모듈들은 학습이 진행되는 모듈들
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)
        self.run_intro2 = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)
        self.run_ending2 = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                 groups=1, bias=True)

        # 2stage 입력와 SAM특징 맵을 concat하기 위한 conv,norm
        self.run_feat_conv = nn.Conv2d(in_channels=width * 2, out_channels=width, kernel_size=1, padding=0, stride=1,
                                   bias=True)
        self.run_feat_norm = LayerNorm2d(width * 2)

        self.encoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.run_first_stage_decoders = nn.ModuleList()
        
        self.run_second_stage_encoders = nn.ModuleList()
        self.run_second_stage_decoders = nn.ModuleList()
        
        self.run_enc_channel_interaction = nn.ModuleList()
        self.run_dec_channel_interaction = nn.ModuleList()

        # First stage
        self.downs = nn.ModuleList()
        self.run_first_stage_ups = nn.ModuleList()

        # Second stage
        self.run_second_stage_downs = nn.ModuleList()
        self.run_second_stage_ups = nn.ModuleList()

        self.run_fft_interaction = nn.ModuleList()
        

        chan = width

        # -------------------    First stage  ---------------------
        for i in range(len(first_enc_blk_nums)):
            if first_enc_blk_nums[i]==1 and self.ufp_pretrain:
                self.encoders.append(nn.Sequential(*[NAFBlock_kernel(chan, kernel_size=kernel_size) for _ in range(first_enc_blk_nums[i])]))
            else:
                self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(first_enc_blk_nums[i])]))
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            if self.ufp_pretrain:
                self.kernel_down.append(nn.Conv2d(kernel_size * kernel_size, kernel_size * kernel_size, 2, 2))
            else:
                self.kernel_down.append(nn.Identity())
            chan = chan * 2
            self.patch_size = self.patch_size // 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(first_middle_blk_num)]
            )

        for i in range(len(first_dec_blk_nums)):
            self.run_first_stage_ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.patch_size = self.patch_size * 2
            self.run_first_stage_decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(first_dec_blk_nums[i])]
                )
            )
        self.padder_size = 2 ** len(self.encoders)
        self.run_sam = SAM(chan, width, padder_size=self.padder_size)

        # -------------------    Second stage  ---------------------
        chan = width

        for i in range(len(second_enc_blk_nums)):
            self.run_second_stage_encoders.append(
                nn.Sequential(
                    *[PatchBlock(chan, 
                                patch_size=self.patch_size,
                                FFTmode=True) for _ in range(second_enc_blk_nums[i])]
                )
            )

            self.run_second_stage_downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2
            self.patch_size = self.patch_size // 2

        for i in range(len(second_dec_blk_nums)):
            self.run_second_stage_ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.patch_size = self.patch_size * 2

            self.run_second_stage_decoders.append(
                nn.Sequential(
                    *[MDBlock(chan, 
                               dwconv_kernel_size=dwconv_kernel_size,
                               patch_size=self.patch_size, 
                               Patchmode=True, 
                               FFTmode=True) for _ in range(second_dec_blk_nums[i])]
                )
            )

    # -------------------    Foward stage  ---------------------
    def forward(self, input_img):
        B, in_C, in_H, in_W = input_img.shape

        inp = check_image_size(input_img, [self.padder_size, self.padder_size])
        if self.ufp_pretrain:
            with torch.no_grad():
                # kernel estimation: size [B, H*W, 19, 19]
                kernel_code, kernel_var = self.kernel_extra(inp)
                kernel_code = (kernel_code - torch.mean(kernel_code, dim=[2, 3], keepdim=True)) / torch.std(kernel_code,
                                                                                                            dim=[2, 3],
                                                                                                            keepdim=True)
                # code uncertainty
                sigma = kernel_var
                kernel_code_uncertain = kernel_code * torch.sqrt(1 - torch.square(sigma)) + torch.randn_like(kernel_code) * sigma

                kernel = generate_k(self.flow, kernel_code_uncertain.reshape(kernel_code.shape[0]*kernel_code.shape[1], -1))
                kernel = kernel.reshape(kernel_code.shape[0], kernel_code.shape[1], self.kernel_size, self.kernel_size)
                kernel_blur = kernel

                kernel = kernel.permute(0, 2, 3, 1).reshape(B, self.kernel_size*self.kernel_size, inp.shape[2], inp.shape[3])

        # ------------ First stage ------------
        x = inp
        x = self.intro(x.clone())
        first_enc_features = []  # 첫번째 enc stage의 특징맵 저장
        first_dec_features = []  # 첫번째 dec stage의 특징맵 저장

        for encoder, down, kernel_down in zip(self.encoders, self.downs, self.kernel_down):
            if len(encoder) == 1 and self.ufp_pretrain:
                x = encoder[0](x, kernel)
                kernel = kernel_down(kernel)
            else:
                x = encoder(x)
            first_enc_features.append(x.clone())  # 특징맵 복사본 저장
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.run_first_stage_decoders, self.run_first_stage_ups, first_enc_features[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

            first_dec_features.append(x.clone())

        first_feature, first_out_img = self.run_sam(x, input_img)

        # ------------ Second stage ------------
        x = self.run_intro2(inp)
        x = torch.concat((x, first_feature), 1)
        
        x = self.run_feat_norm(x)
        x = self.run_feat_conv(x)

        encs2 = []

        for s_encoder, down, first_skip_feature in zip(self.run_second_stage_encoders, self.run_second_stage_downs,
                                                       first_dec_features[::-1]
                                                       ):
            for block in s_encoder:
                x = block(x,first_skip_feature)
            encs2.append(x)
            x = down(x)

        i=True
        for decoder, up, enc_skip in zip(self.run_second_stage_decoders, self.run_second_stage_ups, encs2[::-1]):
            x = up(x)
            if i:
                i=False
            else:
                x = x + enc_skip

            x = decoder(x)

        x = self.run_ending2(x)

        second_out_img = x[:, :, :in_H, :in_W] + first_out_img

        return second_out_img, first_out_img


class TSMKDNet_Local(Local_Base, TSMKDNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        TSMKDNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        self.eval()

        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


from ptflops import get_model_complexity_info
def get_parameter_number(net):
    total_num = sum(np.prod(p.size()) for p in net.parameters())
    trainable_num = sum(np.prod(p.size()) for p in net.parameters() if p.requires_grad)
    print('Total: ', total_num)
    print('Trainable: ', trainable_num)
def apply_freezing_rule(model):
    for name, param in model.named_parameters():
        parts = name.split('.')
        stringLine = parts[0].split('_')[0]
        if stringLine != 'run':
            param.requires_grad = False

        else:
            param.requires_grad = True 

def summarize_params(model):
    total, trainable = 0, 0
    for name, param in model.named_parameters():
        p = param.numel()
        total += p
        if param.requires_grad:
            trainable += p
        print(f"{'[Train]' if param.requires_grad else '[Frozen]'} {name:60s} | shape: {tuple(param.shape)} | params: {p}")
    print(f"\n✅ Total Params: {total/1e6:.2f} M, Trainable: {trainable/1e6:.2f} M, Ratio: {trainable/total*100:.2f}%")

if __name__ == '__main__':
    img_channel = 3
    width = 64
    patch_size = 64
    first_enc_blks = [1, 1, 1, 28]
    first_middle_blk_num= 1
    first_dec_blks = [1, 1, 1, 1]

    second_enc_blks = [1, 1, 1, 4]
    second_dec_blks = [4, 1, 1, 1]

    net = TSMKDNet(img_channel=img_channel, width=width, first_middle_blk_num=first_middle_blk_num,
                  first_enc_blk_nums=first_enc_blks, first_dec_blk_nums=first_dec_blks,
                  second_enc_blk_nums=second_enc_blks, second_dec_blk_nums=second_dec_blks,
                  ufp_pretrain=True, dwconv_kernel_size=7, 
                  patch_size=patch_size, kernel_size=19)
    apply_freezing_rule(net)
    load_net = torch.load('experiments/New_OursB/models/net_g_latest.pth', map_location='cpu')

    for a,b in zip(net.state_dict().keys(), load_net['params'].keys()):
        if a != b:
            print(f"Mismatch: {a} != {b}")
            input()
            
        else:
            net.state_dict()[a].copy_(load_net['params'][b])
            # print(f"match: {a} == {b}")
    
    inp_shape = (3, 1280, 720)

    from torchinfo import summary

    summary(net, input_size=(1, 3, 1280, 720), depth=10)
    
    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    # params = float(params[:-3])
    # macs = float(macs[:-4])
    
    print(macs, params)
    get_parameter_number(net)