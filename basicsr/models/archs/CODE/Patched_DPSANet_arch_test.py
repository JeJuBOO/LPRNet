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

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

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


def check_patch_size(x, patch_size):
    B, C, H, W = x.shape
    pad_h = (patch_size[0] - H % patch_size[0]) % patch_size[0]
    pad_w = (patch_size[1] - W % patch_size[1]) % patch_size[1]
    x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
    return x, H, W

def split_image_into_patches_with_position_encoding(image, patch_size):
    """
    이미지를 패치로 나누고, 포지셔널 인코딩을 추가한 뒤, 
    (Batch * num_patches, C, patch_h, patch_w) 형식으로 반환하는 함수.

    Args:
        image (torch.Tensor): 입력 이미지 텐서 (B, C, H, W)
        patch_size (int): 패치의 높이와 너비 (정사각형 패치 기준)

    Returns:
        patches_with_pos (torch.Tensor): (Batch * num_patches, C, patch_h, patch_w)
    """
    # 입력 이미지 크기 가져오기
    B, C, H, W = image.shape

    # 패치 크기 검증
    assert H % patch_size[0]== 0 and W % patch_size[1] == 0, "H, W는 patch_size로 나누어 떨어져야 합니다."

    # 패치 크기 계산
    h_patches = H // patch_size[0]  # 세로 패치 개수
    w_patches = W // patch_size[1]  # 가로 패치 개수
    num_patches = h_patches * w_patches  # 전체 패치 개수

    # 이미지를 패치로 나누기 (torch.unfold 사용)
    patches = image.unfold(2, patch_size[0], patch_size[1]).unfold(3, patch_size[0], patch_size[1])
    patches = patches.permute(0, 2, 3, 1, 4, 5)  # (B, h_patches, w_patches, C, patch_h, patch_w)

    # 포지셔널 인코딩 생성
    B, h_patches, w_patches, C, patch_h, patch_w = patches.shape
    pos_encoding = generate_sinusoidal_position_encoding(h_patches, w_patches, C, patch_size).to(patches.device)  

    # 패치에 포지셔널 인코딩 추가
    patches_with_pos = patches + pos_encoding.unsqueeze(0)  # Broadcasting (B, h_patches, w_patches, C, patch_h, patch_w)

    # 패치를 평탄화하여 (Batch * num_patches, C, patch_h, patch_w)로 변환
    patches_with_pos = patches_with_pos.reshape(B * num_patches, C, patch_size[0], patch_size[1])

    return patches_with_pos

def generate_sinusoidal_position_encoding(h_patches, w_patches, C, patch_size):
    """
    2D 사인-코사인 포지셔널 인코딩 생성.

    Args:
        h_patches (int): 세로 패치 개수
        w_patches (int): 가로 패치 개수
        C (int): 입력 채널 수
        patch_size (tuple): 패치 크기 

    Returns:
        pos_encoding (torch.Tensor): 포지셔널 인코딩 텐서 (1, h_patches, w_patches, C, patch_size, patch_size)
    """

    # 포지션 값을 생성
    position_h = torch.arange(h_patches).unsqueeze(1).repeat(1, w_patches)  # (h_patches, w_patches)
    position_w = torch.arange(w_patches).unsqueeze(0).repeat(h_patches, 1)  # (h_patches, w_patches)

    # 채널에 따라 사인-코사인 인코딩 생성 (짝수 채널 분리)
    half_C = C // 2
    pos_h = torch.sin(position_h.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, half_C, patch_size[0], patch_size[1]))
    pos_w = torch.cos(position_w.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, C-half_C, patch_size[0], patch_size[1]))

    # 채널 방향으로 합치기
    pos_encoding = torch.cat([pos_h, pos_w], dim=2)  # (h_patches, w_patches, C, patch_size, patch_size)

    # 배치 차원 추가
    return pos_encoding.unsqueeze(0)

class UpperNAFBlock(nn.Module):
    def __init__(self, c, k, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., dilation=0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        if dilation:
            self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=k, dilation=dilation, padding=k//2, stride=1, groups=dw_channel,
                               bias=True)
        else:
            self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=k, padding=k//2, stride=1, groups=dw_channel,
                               bias=True)
        
        
        
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
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
    
class DPSABlock(nn.Module):
    def __init__(self, c, NAFNetmode=False ,FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.nafnetmode = NAFNetmode
        self.norm1 = LayerNorm2d(c)

        self.local_path = nn.Sequential(
            UpperNAFBlock(c, 3),
        )
        self.global_path = nn.Sequential(
            UpperNAFBlock(c, 7),
        )
        
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c

        self.norm2 = LayerNorm2d(ffn_channel)
        self.norm3 = LayerNorm2d(c)

        self.conv1 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)


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
    
class Patched_DPSABlock(nn.Module):
    def __init__(self, channel, patch_size, dwconv_kernel_size=7, NAFNetmode=False, FirstMode=False, patch_skip = False, global_skip = False, FFN_Expand=2, drop_out_rate=0., attn_drop=0.):
        super().__init__()
        self.dim = channel
        c = channel
        self.patch_size = patch_size
        self.dwconv_kernel_size = dwconv_kernel_size
        self.nafnetmode = NAFNetmode
        self.FirstMode = FirstMode
        self.patch_skip = patch_skip
        self.global_skip = global_skip
        ## -----------------------NAFNet branch------------------------
        self.norm1 = LayerNorm2d(c)

        self.local_path = nn.Sequential(
            UpperNAFBlock(c, 3),
        )
        self.global_path = nn.Sequential(
            UpperNAFBlock(c, self.dwconv_kernel_size),
        )

        self.norm_dw = LayerNorm2d(2*c)
        self.conv_dw = nn.Conv2d(in_channels=2*c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        ## -----------------------patch branch-----------------------

        self.conv1_1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2_1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.dconv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=3//2, stride=1, groups=c,
                               bias=True)
        self.dconv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=3//2, stride=1, groups=c,
                               bias=True)
        
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        

        self.sg = SimpleGate()
        self.gelu = nn.GELU()
        # # Simplified Channel Attention
        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=c // 2, out_channels=c // 2, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )

        self.patch_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//2, kernel_size=1),
            nn.BatchNorm2d(c//2),
            nn.GELU(),
            nn.Conv2d(c//2, c, kernel_size=1)
        )

        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//2, kernel_size=1),
            nn.BatchNorm2d(c//2),
            nn.GELU(),
            nn.Conv2d(c//2, c, kernel_size=1),
        )
        # self.channel_interaction = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(c, c//2, kernel_size=1),
        # )
        self.norm3 = LayerNorm2d(2*c)
        self.conv3 = nn.Conv2d(in_channels=2*c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(c, c // 2, kernel_size=1),
            nn.BatchNorm2d(c // 2),
            nn.GELU(),
            nn.Conv2d(c // 2, 1, kernel_size=1)
        )
        # self.spatial_interaction = nn.Sequential(
        #     nn.Conv2d(c, 1, kernel_size=1),
        # )

        self.norm4 = LayerNorm2d(2*c)
        self.conv4 = nn.Conv2d(in_channels=2*c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        ## ----------------------------last branch-----------------------
        ffn_channel = FFN_Expand * c
        self.norm5 = LayerNorm2d(c)

        self.conv5 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # simplegate
        self.conv6 = nn.Conv2d(in_channels=ffn_channel//2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        if self.patch_skip:
            self.p_skip = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        if self.global_skip:
            self.g_skip = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.position_encoding = nn.Parameter(torch.randn((1, c, 1, 1)), requires_grad=True)
    
    def forward(self, inp):
        """
        Args:
            x: input features with shape of (B, C, H, W)
        """
        B, C, H, W = inp.size()
        N = H//self.patch_size[0] * W//self.patch_size[1]
        x = inp
        x = self.norm1(x) # x: (B, C, H, W)
        debug_print = True
        
        ## ------------ Conv branch ----------------

        if self.nafnetmode:
            local_features = self.local_path(x)
            combined_feature = local_features
        else:
            local_features = self.local_path(x)
            global_features = self.global_path(x)

            combined_feature = torch.cat([local_features, global_features], dim=1)
            combined_feature = self.norm_dw(combined_feature)
            combined_feature = self.conv_dw(combined_feature) 

        combined_feature = self.dropout1(combined_feature)
        if debug_print:print('\n\nConv branch shape : ',combined_feature.shape)


        channel_interaction = self.channel_interaction(combined_feature) # x: (B, C, 1, 1)
        if debug_print:print('channel_interaction shape : ',channel_interaction.shape)                  

        ## ------------ patch branch -----------
        x_pat, original_h, original_w = check_patch_size(x, self.patch_size)
        if debug_print:print('x_pat(check_patch_size) shape : ',x_pat.shape) 

        if self.FirstMode:
            x_pat = split_image_into_patches_with_position_encoding(x_pat, self.patch_size)
        else:
            x_pat = patch_partition(x_pat, self.patch_size) # (num_patches*B, C, pH, pW)

        if debug_print:print('x_pat(patch_partition) shape : ',x_pat.shape)

        NB, _, wH, wW = x_pat.shape

        x_pat1 = self.conv1_1(x_pat)  # (B*N, C, wH, wW)
        x_pat1 = self.dconv1(x_pat1)  # (B*N, C, wH, wW)
        x_pat1 = self.gelu(x_pat1) # (B*N, C, 1, 1)
        if debug_print:print('x_pat(simple gate) shape : ',x_pat1.shape)
        
        channel_interaction = torch.sigmoid(channel_interaction)  # (B*N, C, 1, 1)

        x_pat1 = x_pat1.view(B,-1,C,wH,wW) # (B, N, C, 1, 1)
        channel_interaction = channel_interaction.view(B,1,C,1,1) # (B, 1, C, 1, 1)

        if debug_print:print('x_pat, channel_interaction shape : ',x_pat1.shape,channel_interaction.shape)
        x_pat1 = x_pat1 * channel_interaction
        if debug_print:print('x_pat * channel shape : ',x_pat1.shape)

        x_pat1 = x_pat1.view(-1,C,wH,wW)
        x_pat1 = self.conv1(x_pat1)  # (B*N, C, wH, wW)
        if debug_print:print('x_pat1 shape : ',x_pat1.shape)

        x_pat2 = self.conv2_1(x_pat)
        x_pat2 = self.dconv2(x_pat2)  # (B*N, C, wH, wW)
        x_pat2 = self.gelu(x_pat2)
        x_pat2 = x_pat2 * torch.sigmoid(self.patch_channel_attention(x_pat2))  # (B*N, C, wH, wW)
        x_pat2 = self.conv2(x_pat2)  # (B*N, C, wH, wW)

        x_pat = torch.cat([x_pat1, x_pat2], dim=1)  # (B*N, 2C, wH, wW)
        x_pat = self.norm3(x_pat)
        x_pat = self.conv3(x_pat)  # (B*N, C, wH, wW)

        # 원래 크기로 복원
        patch_feature = patch_reverse(x_pat, original_h, original_w) # (B, C, H, W)
        if debug_print:print('patch_feature(patch_reverse) shape : ',patch_feature.shape)

        # patch branch skipconetion
        if self.patch_skip:
            patch_feature = inp + patch_feature * self.p_skip

        # 나머지 부분은 동일
        spatial_interaction = self.spatial_interaction(patch_feature)
        if debug_print:print('spatial_interaction shape : ',spatial_interaction.shape)

        naf_feature = torch.sigmoid(spatial_interaction) * combined_feature

        # global branch skipconetion
        if self.global_skip:
            naf_feature = inp + naf_feature * self.g_skip

        x = torch.cat([naf_feature, patch_feature], dim=1)
        x = self.norm4(x)
        x = self.conv4(x)
        x = inp + x * self.beta


        ## ------------ last branch ----------------
        x1 = self.norm5(x)
        x1 = self.conv5(x1)
        x1 = self.sg(x1)
        x1 = self.conv6(x1)

        x1 = self.dropout2(x1)

        output = x + x1 * self.gamma

        return output 

class Patched_DPSANet(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], patch_size=[16, 16], dwconv_kernel_size=7):
        super().__init__()
        self.patch_size = patch_size

        # self.preprocessing = Patched_DPSABlock(img_channel, (16, 16), dwconv_kernel_size=7, NAFNetmode=True, FirstMode=True)

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for i in range(len(enc_blk_nums)):

            self.encoders.append(
                nn.Sequential(
                    # *[DPSABlock(chan,NAFNetmode=True) for _ in range(dec_blk_nums[i])]
                    *[Patched_DPSABlock(chan,
                                        self.patch_size,
                                        dwconv_kernel_size,
                                        NAFNetmode=True,
                                        patch_skip=True,
                                        global_skip=True,
                                        FirstMode=False) for _ in range(enc_blk_nums[i])]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2
            self.patch_size = [int(self.patch_size[0]/2), int(self.patch_size[1]/2)]

        self.middle_blks = \
            nn.Sequential(
                # *[DPSABlock(chan,NAFNetmode=True) for _ in range(dec_blk_nums[i])]
                *[Patched_DPSABlock(chan,
                                    self.patch_size,
                                    dwconv_kernel_size,
                                    NAFNetmode=True,
                                    patch_skip=True,
                                    global_skip=True,
                                    FirstMode=False) for _ in range(middle_blk_num)]
            )

        for i in range(len(dec_blk_nums)):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[DPSABlock(chan,NAFNetmode=True) for _ in range(dec_blk_nums[i])]
                    # *[Patched_DPSABlock(chan,
                    #                       patch_size,
                    #                       patch_num,
                    #                       dwconv_kernel_size) for _ in range(dec_blk_nums[i])]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape

        inp = self.check_image_size(inp)     

        x = inp
        # x = self.preprocessing(inp)

        x = self.intro(x)
        encs = []
        # print('intro x',x.shape)
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            # print('enc x',x.shape)
            x = down(x)
            # print('down x',x.shape)

        x = self.middle_blks(x)
        # print('middle x',x.shape)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            # print('up x',x.shape)
            x = x + enc_skip
            x = decoder(x)
            # print('dec x',x.shape)

        x = self.ending(x)
        # print('ending x',x.shape)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    
class Patched_DPSANetLocal(Local_Base, Patched_DPSANet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        Patched_DPSANet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 1]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    net = Patched_DPSANet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
    