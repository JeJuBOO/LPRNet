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
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
# from basicsr.metrics.psnr_ssim import calculate_psnr

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# -------------------------   First Stage Block -----------------------------

class UpperNAFBlock(nn.Module):
    def __init__(self, c, k, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=k, padding=k // 2,
                               stride=1, groups=dw_channel,
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

        self.dropout = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        return x


class MDBlock(nn.Module):
    def __init__(self, c, dwconv_kernel_size, NAFNetmode=False, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.nafnetmode = NAFNetmode

        self.norm1 = LayerNorm2d(c)

        self.local_path = nn.Sequential(
            UpperNAFBlock(c, 3),
        )
        self.global_path = nn.Sequential(
            UpperNAFBlock(c, dwconv_kernel_size),
        )

        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c

        self.norm2 = LayerNorm2d(ffn_channel)
        self.norm3 = LayerNorm2d(c)

        self.conv1 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        if self.nafnetmode:
            local_features = self.local_path(x)
            combined_feature = local_features
        else:
            local_features = self.local_path(x)
            global_features = self.global_path(x)

            combined_feature = torch.cat([local_features, global_features], dim=1)
            combined_feature = self.norm2(combined_feature)
            combined_feature = self.conv1(combined_feature)

        combined_feature = self.dropout1(combined_feature)

        x = inp + combined_feature * self.beta

        x1 = self.norm3(x)

        x1 = self.conv2(x1)
        x1 = self.sg(x1)
        x1 = self.conv3(x1)

        x1 = self.dropout2(x1)

        output = x + x1 * self.gamma

        return output


# -------------------------   Second Stage Block -----------------------------

class PatchBlock(nn.Module):
    def __init__(self, c, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.channel = c
        dw_channel = c * 2
        self.norm = LayerNorm2d(c)

        self.pat1_conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                                    bias=True)
        self.pat1_conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=3 // 2,
                                    stride=1, groups=c,
                                    bias=True)

        self.pat1_conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                                    groups=1, bias=True)

        self.pat2_conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                                    bias=True)
        self.pat2_conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=3 // 2,
                                    stride=1, groups=c,
                                    bias=True)

        self.pat2_conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                                    groups=1, bias=True)

        ffn_channel = FFN_Expand * c

        self.norm1 = LayerNorm2d(ffn_channel)
        self.norm2 = LayerNorm2d(c)

        self.gelu = nn.GELU()

        self.conv1 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 2, kernel_size=1),
            nn.BatchNorm2d(c // 2),
            nn.GELU(),
            nn.Conv2d(c // 2, c, kernel_size=1),
        )

        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 2, kernel_size=1),
            nn.BatchNorm2d(c // 2),
            nn.GELU(),
            nn.Conv2d(c // 2, c, kernel_size=1),
        )

    # def forward(self, inp, first_stage_feat):
    def forward(self, combined_input):
        combine_B, _, H, W = combined_input.shape
        
        if self.training:
            inp_batch = 4  # train mode
        else:
            inp_batch = 1 

        patch_num = (combine_B // inp_batch) -1
        inp_B = combine_B // (patch_num + 1)

        inp, first_stage_feat = torch.split(combined_input, [combine_B - inp_B, inp_B], dim=0)
        B_P, C, H, W = inp.size()
        first_stage_feat = first_stage_feat[:, :, 0:1, 0:1]
        x = inp

        x = self.norm(x)

        x_pat1 = self.pat1_conv1(x)
        x_pat1 = self.pat1_conv2(x_pat1)
        x_pat1 = self.gelu(x_pat1)
        x_pat1 = x_pat1.reshape(B_P // patch_num, patch_num, C, H, W)
        x_pat1 = x_pat1 * first_stage_feat.unsqueeze(1)  # (B_P//patch_num, patch_num, C, pH, pW) * (B_P//patch_num, 1, C, 1, 1)
        x_pat1 = x_pat1.view(-1, C, H, W)

        x_pat1 = self.pat1_conv3(x_pat1)

        x_pat2 = self.pat2_conv1(x)
        x_pat2 = self.pat2_conv2(x_pat2)
        x_pat2 = self.gelu(x_pat2)
        x_pat2 = x_pat2 * self.channel_attention(x_pat2)
        x_pat2 = self.pat2_conv3(x_pat2)

        combined_feature = torch.cat([x_pat1, x_pat2], dim=1)
        combined_feature = self.norm1(combined_feature)
        combined_feature = self.conv1(combined_feature)

        combined_feature = self.dropout1(combined_feature)

        x = inp + combined_feature * self.beta

        x1 = self.norm2(x)

        x1 = self.conv2(x1)
        x1 = self.gelu(x1)
        x1 = self.conv3(x1)

        x1 = self.dropout2(x1)

        output = x + x1 * self.gamma

        return output

# class SAM(nn.Module):
#     def __init__(self,channel=3, kernel_size=1):
#         super(SAM, self).__init__()
#         self.conv1 = nn.Conv2d(channel, channel, kernel_size, bias=True)
#         self.conv2 = nn.Conv2d(channel, channel, kernel_size, bias=True)
#         self.conv3 = nn.Conv2d(channel, channel, kernel_size, bias=True)

#     def forward(self, x, x_img, psnr_weight=None,qualities_weight=None):
#         B,_,_,_ = x.shape
#         x1 = self.conv1(x)
#         img = self.conv2(x) + x_img
        
#         out = self.check_patch_size(img, [32,32])
#         out = self.patch_partition(out, [32,32])

#         x2 = torch.sigmoid(self.conv3(img))
#         if psnr_weight is not None:
#             psnr_weight = psnr_weight.unsqueeze(-1).unsqueeze(-1)
#             attention_weight = (1-psnr_weight) * x2
#         elif qualities_weight is not None:
#             qualities_weight = torch.sigmoid(qualities_weight)
#             qualities_weight = qualities_weight.unsqueeze(-1).unsqueeze(-1).view(B,-1,1,1,1)
#             print(qualities_weight.shape,x2.shape)
#             attention_weight = (1-qualities_weight) * x2
#         else:
#             attention_weight = x2
#         x1 = x1*attention_weight
#         x1 = x1+x
#         return x1
    
# -------------------------      Network Part     ------------------------------

class TSMPNet(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],
                 patch_size=[32, 32], dwconv_kernel_size=7):
        super().__init__()
        self.psnr_attention = False
        self.qualities_attention = True

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        # self.onetwo = SAM()
        self.sam_conv1 = nn.Conv2d(img_channel, width, 1, bias=True)
        self.sam_conv2 = nn.Conv2d(width, img_channel, 1, bias=True)
        self.sam_conv3 = nn.Conv2d(img_channel, width, 1, bias=True)


        self.intro2 = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        
        self.ending2 = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)
        
        self.feat_conv = nn.Conv2d(in_channels=width*2, out_channels=width, kernel_size=1, padding=0, stride=1, bias=True)

        self.feat_norm = LayerNorm2d(width*2)

        self.first_stage_encoders = nn.ModuleList()
        self.first_stage_decoders = nn.ModuleList()
        self.first_stage_middle_blks = nn.ModuleList()

        self.second_stage_encoders = nn.ModuleList()
        self.second_stage_decoders = nn.ModuleList()
        self.second_stage_middle_blks = nn.ModuleList()
        
        self.enc_channel_interaction = nn.ModuleList()
        self.dec_channel_interaction = nn.ModuleList()

        # First stage
        self.first_stage_downs = nn.ModuleList()
        self.first_stage_ups = nn.ModuleList()

        # Second stage
        self.second_stage_downs = nn.ModuleList()
        self.second_stage_ups = nn.ModuleList()

        chan = width

        # -------------------    First stage  ---------------------
        for i in range(len(enc_blk_nums)):
            self.first_stage_encoders.append(
                nn.Sequential(
                    *[MDBlock(chan, dwconv_kernel_size) for _ in range(enc_blk_nums[i])]
                )
            )
            self.first_stage_downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.first_stage_middle_blks = \
            nn.Sequential(
                *[MDBlock(chan, dwconv_kernel_size) for _ in range(middle_blk_num)]
            )

        for i in range(len(dec_blk_nums)):
            self.first_stage_ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.first_stage_decoders.append(
                nn.Sequential(
                    *[MDBlock(chan, dwconv_kernel_size) for _ in range(dec_blk_nums[i])]
                )
            )

        # -------------------    Second stage  ---------------------
        chan = width

        for i in range(len(enc_blk_nums)):
            self.second_stage_encoders.append(
                nn.Sequential(
                    *[PatchBlock(chan) for _ in range(enc_blk_nums[i])]
                )
            )
            self.enc_channel_interaction.append(
                nn.Sequential(
                    *[nn.AdaptiveAvgPool2d(1),
                      nn.Conv2d(chan, chan // 2, kernel_size=1),
                      nn.BatchNorm2d(chan // 2),
                      nn.GELU(),
                      nn.Conv2d(chan // 2, chan, kernel_size=1), ]
                )
            )
            self.second_stage_downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.second_stage_middle_blks = \
            nn.Sequential(
                *[PatchBlock(chan) for _ in range(middle_blk_num)]
            )
        self.mid_channel_interaction = \
            nn.Sequential(
                *[nn.AdaptiveAvgPool2d(1),
                  nn.Conv2d(chan, chan // 2, kernel_size=1),
                  nn.BatchNorm2d(chan // 2),
                  nn.GELU(),
                  nn.Conv2d(chan // 2, chan, kernel_size=1), ])

        for i in range(len(dec_blk_nums)):
            self.second_stage_ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.second_stage_decoders.append(
                nn.Sequential(
                    *[PatchBlock(chan) for _ in range(dec_blk_nums[i])]
                )
            )
            self.dec_channel_interaction.append(
                nn.Sequential(
                    *[nn.AdaptiveAvgPool2d(1),
                      nn.Conv2d(chan, chan // 2, kernel_size=1),
                      nn.BatchNorm2d(chan // 2),
                      nn.GELU(),
                      nn.Conv2d(chan // 2, chan, kernel_size=1), ]
                )
            )

        self.padder_size = 2 ** len(self.first_stage_encoders)

    # -------------------    Patch Define  ---------------------
    @staticmethod
    def patch_partition(x, patch_size):
        """
        Partition a tensor into non-overlapping patchs of the given size.

        Args:
            x: (B, C, H, W) input tensor in the PyTorch format.
            patch_size (tuple[int]): patch size (height, width).
        Returns:
            patchs: (B, num_patch*C, patch_size[0], patch_size[1]) partitioned patchs.
        """
        B, C, H, W = x.shape
        x = x.view(B, C, H // patch_size[0], patch_size[0], W // patch_size[1], patch_size[1])

        # patchs = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        # patchs = patchs.view(B, -1, patch_size[0], patch_size[1])

        # (num_patch*B, C, patch_size[0], patch_size[1])
        patchs = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        patchs = patchs.view(-1, C, patch_size[0], patch_size[1])

        return patchs

    @staticmethod
    def patch_reverse(patches, H: int, W: int):
        """ Reverse patches to feature map.
        [B, C * (H // patch_size * W // patch_size), patch_size, patch_size] --> [B, C, H, W]
        Args:
            patches: (B, num_patches * C, patch_size, patch_size)
            H (int): Height of original image
            W (int): Width of original image
        Returns:
            x: (B, C, H, W)
        """
        B_N, C, pH, pW = patches.shape
        num_patches = (H // pH) * (W // pW)
        B = B_N // num_patches

        H_padded = (H + pH - 1) // pH * pH
        W_padded = (W + pW - 1) // pW * pW

        # Reshape: [num_patches * B, C, pH, pW] -> [B, H // pH, W // pW, C, pH, pW]
        x = patches.view(B, H_padded // pH, W_padded // pW, C, pH, pW)

        # Permute: [B, H // pH, W // pW, C, pH, pW] -> [B, C, H // pH, pH, W // pW, pW]
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()

        # Reshape: [B, C, H // pH, pH, W // pW, pW] -> [B, C, H_padded, W_padded]
        x = x.view(B, C, H_padded, W_padded)

        # Crop to original size
        x = x[:, :, :H, :W]
        return x
    
    @staticmethod
    def calculate_patch_psnr(img1, img2):
        """
        Calculate PSNR for each patch in a batch of patched images.

        Args:
            img1 (torch.Tensor): Ground truth image patches (B, P, C, patch_H, patch_W).
            img2 (torch.Tensor): Predicted image patches (B, P, C, patch_H, patch_W).

        Returns:
            torch.Tensor: PSNR values for each patch (B, P, 1).
        """
        B, P, C, patch_H, patch_W = img1.shape

        # Ensure both images have the same shape
        assert img1.shape == img2.shape, "Images must have the same shape"

        # Calculate MSE for each patch
        mse = torch.mean((img1 - img2) ** 2, dim=(2,3,4), keepdim=True)  # (B, P, 1, 1, 1)

        # Determine the maximum pixel value
        max_value = torch.max(img1, img2).max()
        max_value = 1.0 #if max_value <= 1 else 255.0

        # Calculate PSNR for each patch
        psnr = 20. * torch.log10(max_value / torch.sqrt(mse))

        return torch.tensor(psnr).view(-1,1)
    
    @staticmethod
    def calculate_patch_qualities(x, learnable=True):
        if learnable:
            std_weight = nn.Parameter(torch.tensor(0.4))
            contrast_weight = nn.Parameter(torch.tensor(0.3))
            sharpness_weight = nn.Parameter(torch.tensor(0.3))

        qualities = []
        for patch in x:
            std = torch.std(patch)
            contrast = torch.max(patch) - torch.min(patch)
            sharpness = (torch.mean(torch.abs(patch[:, 1:, :] - patch[:, :-1, :]))+\
                         torch.mean(torch.abs(patch[:, :, 1:] - patch[:, :, :-1])))/2
            quality = (std*std_weight + contrast*contrast_weight + sharpness*sharpness_weight)
            qualities.append(quality)

        return torch.tensor(qualities).view(-1,1) # (Batch*patchnum,1)
    # @staticmethod
    # def check_patch_size(x, patch_size):
    #     B, C, H, W = x.shape
    #     pad_h = (patch_size[0] - H % patch_size[0]) % patch_size[0]
    #     pad_w = (patch_size[1] - W % patch_size[1]) % patch_size[1]
    #     x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
    #     return x, H, W

    # -------------------    Foward stage  ---------------------
    def forward(self, inp, target):
        in_B, in_C, in_H, in_W = inp.shape
        inp = self.check_image_size(inp)

        # ------------ First stage ------------
        x = inp
        x = self.intro(x)
        first_enc_features = []  # 첫번째 stage의 특징맵 저장
        first_dec_features = []

        for encoder, down in zip(self.first_stage_encoders, self.first_stage_downs):
            x = encoder(x)
            first_enc_features.append(x.clone())  # 특징맵 복사본 저장
            x = down(x)

        x = self.first_stage_middle_blks(x)
        mid_feature = x.clone()

        for decoder, up, enc_skip in zip(self.first_stage_decoders, self.first_stage_ups, first_enc_features[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            first_dec_features.append(x.clone())
            
        end_x = self.ending(x)
        first_stage_out = end_x + inp
        B,_,_,_ = first_stage_out.shape

        out = self.check_patch_size(first_stage_out, [32,32])
        out = self.patch_partition(out, [32,32])
        
        x1 = self.sam_conv1(out)
        out_img = self.sam_conv2(x) + inp
 
        out_patch = self.check_patch_size(out_img, [32,32])
        out_patch = self.patch_partition(out_patch, [32,32])

        _,pC,pH,pW = out_patch.shape
        target_patch = self.check_patch_size(target, [32,32])
        target_patch = self.patch_partition(target_patch, [32,32])

        x2 = torch.sigmoid(self.sam_conv3(out_patch))# (Batch*patchnum,pC,pH,pW)

        if self.psnr_attention:  
            psnr_weight = self.calculate_patch_psnr(out_patch.view(B,-1,pC,pH,pW),target_patch.view(B,-1,pC,pH,pW)).to(x2.device) # (Batch*patchnum,1)
            psnr_weight = torch.sigmoid(psnr_weight)
            psnr_weight = psnr_weight.unsqueeze(-1).unsqueeze(-1)# (Batch*patchnum,1,1,1)
            attention_weight = (1-psnr_weight) * x2

        elif self.qualities_attention:
            qualities_weight = self.calculate_patch_qualities(out_patch).to(x2.device)
            qualities_weight = torch.sigmoid(qualities_weight)
            qualities_weight = qualities_weight.unsqueeze(-1).unsqueeze(-1)
            attention_weight = (1-qualities_weight) * x2
        else:
            attention_weight = x2
        

        first_feature = x1*attention_weight#(Batch*patchnum,C,pH,pW) (Batch*patchnum,pC,pH,pW)
        # first_feature = x1+out

        # inp = torch.concat((first_feature, inp), 1)

        # ------------ Second stage ------------
        inp = self.check_patch_size(inp, [32, 32])  # (B, 2*C, H_pad, W_pad) 이후 H_pad=H
        inp = self.patch_partition(inp, [32, 32])  # (B*P, 2*C, pH, pW)
        x = self.intro2(inp)
        x = torch.concat((x, first_feature), 1)

        x = self.feat_norm(x)
        x = self.feat_conv(x)
        encs2 = []
        
        # 첫번째 stage의 특징맵을 패치 형태로 변환
        # first_stage_features = [self.patch_partition(feat, (32, 32)) for feat in first_stage_features] # (B, C, H, W) ->

        for s_encoder, down, skip_feature, ch_inter in zip(self.second_stage_encoders, self.second_stage_downs, first_dec_features[::-1],
                                                           self.enc_channel_interaction):
            B, C, H, W = x.shape

            skip_feature = ch_inter(skip_feature)  # (B,C,1,1)
            skip_feature_resized = F.interpolate(skip_feature, size=(H, W), mode='nearest')  # (B,C,pH,pW)
            combined_input = torch.cat([x, skip_feature_resized], dim=0)  # ((B*P)+B, C, pH, pW)
            x = s_encoder(combined_input)  # 첫번째 stage의 특징맵 전달
            encs2.append(x)
            x = down(x)

        B, C, H, W = x.shape

        skip_feature = self.mid_channel_interaction(mid_feature)
        skip_feature_resized = F.interpolate(skip_feature, size=(H, W), mode='nearest')
        combined_input = torch.cat([x, skip_feature_resized], dim=0)
        del skip_feature_resized 

        x = self.second_stage_middle_blks(combined_input)

        for decoder, up, enc_skip, skip_feature_dec, dec_ch_inter in zip(self.second_stage_decoders, self.second_stage_ups, encs2[::-1], first_dec_features,
                                                           self.dec_channel_interaction):
            
            x = up(x)
            x = x + enc_skip
            B, C, H, W = x.shape
            skip_feature = dec_ch_inter(skip_feature_dec)  # (B,C,1,1)
            skip_feature_resized = F.interpolate(skip_feature, size=(H, W), mode='nearest')  # (B,C,pH,pW)
            combined_input = torch.cat([x, skip_feature_resized], dim=0)  # ((B*P)+B, C, pH, pW)
            x = decoder(combined_input)

        x = self.ending2(x)

        x = self.patch_reverse(x, in_H, in_W)
        x = x + first_stage_out
        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def check_patch_size(self, x, patch_size):
        _, _, h, w = x.shape
        pad_h = (patch_size[0] - h % patch_size[0]) % patch_size[0]
        pad_w = (patch_size[1] - w % patch_size[1]) % patch_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
        return x


class TSMPNetLocal(Local_Base, TSMPNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        TSMPNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

from ptflops import get_model_complexity_info

if __name__ == '__main__':
    img_channel = 3
    width = 32

    enc_blks = [1, 1, 1, 1]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]

    # 모델을 래퍼로 감싸서 두 입력을 처리할 수 있게 수정
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            # 두 번째 입력을 생성
            target = torch.randn_like(x)
            return self.model(x, target)

    net = TSMPNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                  enc_blk_nums=enc_blks, dec_blk_nums=dec_blks, dwconv_kernel_size=7)
    
    wrapped_net = ModelWrapper(net)  # 모델을 래퍼로 감싸기
    inp_shape = (3, 256, 256)

    macs, params = get_model_complexity_info(wrapped_net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])
    print(wrapped_net)
    print(macs, params)
