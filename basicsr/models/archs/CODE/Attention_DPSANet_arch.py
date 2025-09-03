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


def window_partition(x, window_size):
    """
    Partition a tensor into non-overlapping windows of the given size.

    Args:
        x: (B, C, H, W) input tensor in the PyTorch format.
        window_size (tuple[int]): window size (height, width).
    Returns:
        windows: (num_windows*B, window_size[0]*window_size[1], C) partitioned windows.
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])

    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = windows.view(-1, window_size[0] * window_size[1], C)

    return windows

def window_reverse(windows, window_size, B: int, H: int, W: int):
    """ Windows reverse to feature map.
    [B * H // win * W // win , win*win , C] --> [B, C, H, W]
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """

    H_padded = (H + window_size[0] - 1) // window_size[0] * window_size[0]
    W_padded = (W + window_size[1] - 1) // window_size[1] * window_size[1]

    # view: [B*num_windows, N, C] -> [B, H//window_size, W//window_size, window_size, window_size, C]
    x = windows.view(B, H_padded // window_size[0], W_padded // window_size[1],
                     window_size[0], window_size[1], -1)
    # permute: [B, H//Wh, W//Ww, Wh, Ww, C] -> [B, C, H//Wh, Wh, W//Ww, Ww]
    # view: [B, C, H//Wh, Wh, W//Ww, Ww] -> [B, C, H, W]
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, -1, H_padded, W_padded)
    x = x[:, :, :H, :W]
    return x

def check_window_size(x, window_size):
    B, C, H, W = x.shape
    pad_h = (window_size[0] - H % window_size[0]) % window_size[0]
    pad_w = (window_size[1] - W % window_size[1]) % window_size[1]
    x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
    return x, H, W


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

    def forward(self, inp):
        x = inp

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        #x = self.dropout(x)

        #y = inp + x * self.beta
        y = x
        return y
    
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
    
class Attention_DPSABlock(nn.Module):
    def __init__(self, c, window_size, dwconv_kernel_size, num_heads, qkv_bias=True, qk_scale=None, NAFNetmode=False, FFN_Expand=2, drop_out_rate=0., attn_drop=0.):
        super().__init__()
        self.dim = c
        attn_dim = c
        
        self.window_size = window_size 
        self.num_heads = num_heads  # using multi head attention
        head_dim = attn_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        relative_coords = self._get_rel_pos()
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.proj_attn = nn.Linear(c, c)
        self.proj_attn_norm = nn.LayerNorm(c)

        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 2, kernel_size=1),
            nn.BatchNorm2d(c // 2),
            nn.GELU(),
            nn.Conv2d(c // 2, c, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(c, c // 2, kernel_size=1),
            nn.BatchNorm2d(c // 2),
            nn.GELU(),
            nn.Conv2d(c // 2, 1, kernel_size=1)
        )
        self.projection_cat = nn.Linear(c * 2, c)

        self.attn_norm = nn.LayerNorm(c)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.qkv = nn.Linear(c, c * 3, bias=qkv_bias)

        self.dwconv_kernel_size = dwconv_kernel_size
        self.nafnetmode = NAFNetmode
        self.norm1 = LayerNorm2d(c)

        self.local_path = nn.Sequential(
            UpperNAFBlock(c, 3),
        )
        self.global_path = nn.Sequential(
            UpperNAFBlock(c, self.dwconv_kernel_size),
        )
        
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c

        self.norm2 = LayerNorm2d(ffn_channel)
        self.norm3 = LayerNorm2d(c)
        self.norm4 = LayerNorm2d(ffn_channel)

        self.conv1 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=2*c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def _get_rel_pos(self):
        """
            Get pair-wise relative position index for each token inside the window.
            Args:
                window_size (tuple[int]): window size
        """
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        return relative_coords
    
    def forward(self, inp):
        """
        Args:
            x: input features with shape of (B, C, H, W)
        """
        IB, _, H, W = inp.size()
        x = inp
        x = self.norm1(x) # x: (B, C, H, W)
        
        ## ------------ Conv branch ----------------

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

        channel_interaction = self.channel_interaction(combined_feature) # x: (B, C, H, W)
                                                    
        ## ------------ Attention branch -----------
        x_atten, original_h, original_w = check_window_size(x, self.window_size)
        x_atten = window_partition(x_atten, self.window_size) # x: num_windows*B, N, C) N: size_window^2
        # print('window x',x_atten.shape)
        x_atten = self.proj_attn_norm(self.proj_attn(x_atten))
        B_, N, C = x_atten.shape
        qkv = self.qkv(x_atten)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # (3, batch*num_window, num_head, window_size^2, channel/num_head)
        q, k, v = qkv.unbind(0)

        # channel interaction
        x_cnn2v = torch.sigmoid(channel_interaction).reshape(-1, 1, self.num_heads, 1, C // self.num_heads) # (batch, 1         , num_head, 1           , channel/num_head))
        v = v.reshape(x_cnn2v.shape[0], -1, self.num_heads, N, C // self.num_heads)                         # (batch, num_window, num_head, window_size^2, channel/num_head)
        v = v * x_cnn2v
        v = v.reshape(-1, self.num_heads, N, C // self.num_heads)

        q = q * self.scale  # Q/sqrt(dk)
        attn = (q @ k.transpose(-2, -1))  # Q*K^{T} / sqrt(dk)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [num_heads, window_size^2, window_size^2]
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x_atten = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # spatial interaction
        x_spatial = window_reverse(x_atten, self.window_size, IB, original_h, original_w)

        spatial_interaction = self.spatial_interaction(x_spatial)
        

        combined_feature = torch.sigmoid(spatial_interaction) * combined_feature

        x = torch.cat([combined_feature, x_spatial], dim=1)
        x = self.norm4(x)
        x = self.conv4(x) 
        x = inp + x * self.beta
        
        ## ------------ last branch ----------------
        x1 = self.norm3(x)
        x1 = self.conv2(x1)
        x1 = self.sg(x1)
        x1 = self.conv3(x1)

        x1 = self.dropout2(x1)

        output = x + x1 * self.gamma

        return output 

class Attention_DPSANet(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], window_size=[16,16], dwconv_kernel_size=7, num_encheads=[2, 4, 8, 16], num_midheads=16, num_decheads=[16, 8, 4, 2]):
        super().__init__()
        self.window_size = window_size
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
                    *[Attention_DPSABlock(chan,
                                          window_size,
                                          dwconv_kernel_size,
                                          num_encheads[i],
                                          NAFNetmode=True) for _ in range(enc_blk_nums[i])]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[Attention_DPSABlock(chan,
                                      window_size,
                                      dwconv_kernel_size,
                                      num_midheads,
                                      NAFNetmode=True) for _ in range(middle_blk_num)]
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
                    # *[Attention_DPSABlock(chan,
                    #                       window_size,
                    #                       dwconv_kernel_size,
                    #                       num_decheads[i]) for _ in range(dec_blk_nums[i])]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape

        inp = self.check_image_size(inp)        
    
        x = self.intro(inp)
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
    
class Attention_DPSANetLocal(Local_Base, Attention_DPSANet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        Attention_DPSANet.__init__(self, *args, **kwargs)

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

    enc_blks = [1, 1, 1, 32]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    net = Attention_DPSANet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
