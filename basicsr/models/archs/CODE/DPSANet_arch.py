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
    
class BaselineBlock(nn.Module):
    def __init__(self, c, DW_Expand=1, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
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

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = x * self.se(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.gelu(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
    
class Attention_DPSABlock(nn.Module):
    def __init__(self, c, window_size, dwconv_kernel_size, NAFNetmode=False, FFN_Expand=2, drop_out_rate=0., attn_drop=0.):
        super().__init__()
        self.dim = c
        
        self.window_size = window_size 

        self.dwconv_kernel_size = dwconv_kernel_size
        self.nafnetmode = NAFNetmode
        self.norm1 = LayerNorm2d(c)

        self.local_path = nn.Sequential(
            UpperNAFBlock(c, 5),
        )
        self.global_path = nn.Sequential(
            UpperNAFBlock(c, self.dwconv_kernel_size),
        )
        
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c

        self.norm2 = LayerNorm2d(ffn_channel)
        self.norm3 = LayerNorm2d(c)

        self.conv1 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        """
        Args:
            x: input features with shape of (B, C, H, W)
        """
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
        
        ## ------------ last branch ----------------
        x1 = self.norm3(x)
        x1 = self.conv2(x1)
        x1 = self.sg(x1)
        x1 = self.conv3(x1)

        x1 = self.dropout2(x1)

        output = x + x1 * self.gamma

        return output 

class Attention_DPSANet(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], window_size=[3,3], dwconv_kernel_size=7, num_encheads=[2, 4, 8, 16], num_midheads=16, num_decheads=[16, 8, 4, 2]):
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
                                          dwconv_kernel_size) for _ in range(enc_blk_nums[i])]
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
                                      dwconv_kernel_size) for _ in range(middle_blk_num)]
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
                    *[BaselineBlock(chan) for _ in range(dec_blk_nums[i])]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, target=None):
        B, C, H, W = inp.shape

        inp = self.check_image_size(inp)        
    
        x = self.intro(inp)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
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

    enc_blks = [1, 1, 1, 1]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    net = Attention_DPSANet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (3, 256, 256)

    from torchinfo import summary
    summary(net, input_size=(1, 3, 224, 224), depth=10)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
