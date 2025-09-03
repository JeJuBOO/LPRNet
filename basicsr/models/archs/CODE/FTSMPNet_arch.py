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
    CustomAdaptiveAvgPool2d
from basicsr.models.archs.local_arch import Local_Base


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
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

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

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


# -------------------------   Second Stage Block -----------------------------
class SAM(nn.Module):
    def __init__(self, chan, out_chan):
        super().__init__()
        img_channel = 3

        self.sam_conv1 = nn.Conv2d(in_channels=chan, out_channels=out_chan, kernel_size=3,
                                   padding=1, stride=1, groups=1, bias=True)
        self.sam_conv3 = nn.Conv2d(in_channels=img_channel, out_channels=out_chan, kernel_size=1, bias=True)
        self.ending = nn.Conv2d(in_channels=chan, out_channels=img_channel, kernel_size=3,
                                padding=1, stride=1, groups=1, bias=True)

    def forward(self, x, x_img):
        # B,_,_,_ = x.shape
        x1 = self.sam_conv1(x)

        first_out_img = self.ending(x) + x_img.detach()  # (Batch,C,H,W)
        x2 = torch.sigmoid(self.sam_conv3(first_out_img))  # (Batch,C,H,W)
        
        first_feature = x + x1 * x2

        return first_feature, first_out_img


class PatchBlock(nn.Module):
    def __init__(self, c, patch_size, drop_out_rate=0.):
        super().__init__()

        self.channel = c
        self.patch_size = patch_size
        cf_channel = 2 * c  # Combine Feature Channel

        self.pat1_conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                                    groups=1, bias=True)
        self.pat1_dconv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=3 // 2, stride=1,
                                     groups=c, bias=True)
        self.pat1_conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                                    groups=1, bias=True)

        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=3 // 2, stride=1,
                               groups=c, bias=True)
        self.conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=3 // 2, stride=1,
                               groups=c, bias=True)

        self.combconv = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1)
        self.combnorm = LayerNorm2d(c)

        self.norm = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.identity = nn.Identity()
        self.gelu = nn.GELU()
        # patch channel feature
        self.p_channel_interaction = nn.Sequential(
            CustomAdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//2, kernel_size=1),
            nn.BatchNorm2d(c//2),
            nn.GELU(),
            nn.Conv2d(c//2, c, kernel_size=1),
            nn.Sigmoid()
        )
        # 1stage global channel feature
        self.g_channel_interaction = nn.Sequential(
            CustomAdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//2, kernel_size=1),
            nn.BatchNorm2d(c//2),
            nn.GELU(),
            nn.Conv2d(c//2, c, kernel_size=1),
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

        x = check_image_size(x, [self.patch_size, self.patch_size])  # (B, 2*C, H_pad, W_pad) 이후 H_pad=H
        x = patch_partition(x, (self.patch_size, self.patch_size))  # (B*P, 2*C, pH, pW)
        B_P, C, pH, pW = x.shape
        patch_num = (B_P // B)
        x_pat1 = self.pat1_conv1(x)
        x_pat1 = self.pat1_dconv2(x_pat1)
        x_pat1 = self.gelu(x_pat1)

        patch_chan_attention = self.p_channel_interaction(x_pat1)

        first_stage_feat = check_image_size(first_stage_feat, [self.patch_size, self.patch_size])
        first_stage_feat = patch_partition(first_stage_feat, (self.patch_size, self.patch_size))

        first_stage_feat = self.identity(first_stage_feat)
        global_chan = self.g_channel_interaction(first_stage_feat)


        first_stage_chan_result = first_stage_feat * patch_chan_attention
        patch_chan_result = x_pat1 * global_chan

        combined_ca_feature = first_stage_chan_result + patch_chan_result

        combined_ca_feature = self.combconv(combined_ca_feature)
        combined_ca_feature = self.combnorm(combined_ca_feature)

        combined_feature = combined_ca_feature * x_pat1

        combined_feature = self.pat1_conv3(combined_feature)

        combined_feature = self.dropout1(combined_feature)
        combined_feature = patch_reverse(combined_feature, H, W)

        x = inp + combined_feature * self.beta

        x1 = self.norm2(x)
        x1 = self.conv2(x1)
        x1 = self.gelu(x1)
        x1 = self.conv3(x1)

        x1 = self.dropout2(x1)
        output = x + x1 * self.gamma
        return output


# -------------------------      Network Part     ------------------------------

class FTSMPNet(nn.Module):

    def __init__(self, img_channel=3, width=32, first_middle_blk_num=1, first_enc_blk_nums=[], first_dec_blk_nums=[],
                 second_middle_blk_num=1, second_enc_blk_nums=[], second_dec_blk_nums=[], dwconv_kernel_size=7,
                 patch_size=64):
        super().__init__()
        self.patch_size = patch_size

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)

        self.intro2 = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)

        self.ending2 = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                 groups=1, bias=True)
        
        self.feat_conv = nn.Conv2d(in_channels=width * 2, out_channels=width, kernel_size=1, padding=0, stride=1,
                                   bias=True)

        self.feat_norm = LayerNorm2d(width * 2)

        self.encoders = nn.ModuleList()
        self.first_stage_decoders = nn.ModuleList()
        self.first_stage_middle_blks = nn.ModuleList()

        self.second_stage_encoders = nn.ModuleList()
        self.second_stage_decoders = nn.ModuleList()
        self.second_stage_middle_blks = nn.ModuleList()

        self.enc_channel_interaction = nn.ModuleList()
        self.dec_channel_interaction = nn.ModuleList()

        # First stage
        self.downs = nn.ModuleList()
        self.first_stage_ups = nn.ModuleList()

        # Second stage
        self.second_stage_downs = nn.ModuleList()
        self.second_stage_ups = nn.ModuleList()

        chan = width

        # -------------------    First stage  ---------------------
        for i in range(len(first_enc_blk_nums)):
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(first_enc_blk_nums[i])]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.first_stage_middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(first_middle_blk_num)]
            )

        for i in range(len(first_dec_blk_nums)):
            self.first_stage_ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.first_stage_decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(first_dec_blk_nums[i])]
                )
            )

        self.sam = SAM(chan, width)

        # -------------------    Second stage  ---------------------
        chan = width
       
        for i in range(len(second_enc_blk_nums)):
            self.second_stage_encoders.append(
                nn.Sequential(
                    *[PatchBlock(chan, self.patch_size) for _ in range(second_enc_blk_nums[i])]
                )
            )

            self.second_stage_downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2
            self.patch_size = self.patch_size//2

        self.second_stage_middle_blks = \
            nn.Sequential(
                *[PatchBlock(chan, self.patch_size) for _ in range(second_middle_blk_num)]
            )

        for i in range(len(second_dec_blk_nums)):
            self.second_stage_ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.patch_size = self.patch_size*2
            self.second_stage_decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(second_dec_blk_nums[i])]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    # -------------------    Foward stage  ---------------------
    def forward(self, inp):
        in_B, in_C, in_H, in_W = inp.shape
        inp = check_image_size(inp, [self.padder_size, self.padder_size])

        # ------------ First stage ------------
        x = inp
        x = self.intro(x.clone())
        first_enc_features = []  # 첫번째 enc stage의 특징맵 저장
        first_dec_features = []  # 첫번째 dec stage의 특징맵 저장

        for encoder, down in zip(self.encoders, self.downs):
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

        first_feature, first_out_img = self.sam(x, inp)

        # ------------ Second stage ------------

        x = self.intro2(inp)
        x = torch.concat((x, first_feature), 1)

        x = self.feat_norm(x)
        x = self.feat_conv(x)

        encs2 = []

        for s_encoder, down, first_skip_feature in zip(self.second_stage_encoders, self.second_stage_downs,
                                                                   first_dec_features[::-1]
                                                                   ):
            for block in s_encoder:
                x = block(x, first_skip_feature)
            encs2.append(x)
            x = down(x)

        B, C, H, W = x.shape

        for block in self.second_stage_middle_blks:
            x = block(x, mid_feature)

        for decoder, up, enc_skip in zip(self.second_stage_decoders, self.second_stage_ups, encs2[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        
        x = x.clone()
        x = self.ending2(x)

        second_out_img = x + first_out_img.detach()

        return second_out_img, first_out_img


class FTSMPNet_Local(Local_Base, FTSMPNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        FTSMPNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        self.eval()

        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


from ptflops import get_model_complexity_info

if __name__ == '__main__':
    img_channel = 3
    width = 32
    patch_size = 64
    first_enc_blks = [1, 1, 1, 1]
    first_middle_blk_num = 1
    first_dec_blks = [1, 1, 1, 1]

    second_enc_blks = [1, 1, 1, 1]
    second_middle_blk_num = 1
    second_dec_blks = [1, 1, 1, 1]


    # 모델을 래퍼로 감싸서 두 입력을 처리할 수 있게 수정
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            # 두 번째 입력을 생성
            return self.model(x)


    net = FTSMPNet(img_channel=img_channel, width=width, first_middle_blk_num=first_middle_blk_num,
                  first_enc_blk_nums=first_enc_blks, first_dec_blk_nums=first_dec_blks,
                  second_middle_blk_num=second_middle_blk_num, second_enc_blk_nums=second_enc_blks,
                  second_dec_blk_nums=second_dec_blks, dwconv_kernel_size=7,patch_size=patch_size)

    wrapped_net = ModelWrapper(net)  # 모델을 래퍼로 감싸기
    inp_shape = (3, 720, 1280)

    from torchinfo import summary
    summary(wrapped_net, input_size=(1, 3, 256, 256), depth=10)

    macs, params = get_model_complexity_info(wrapped_net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
