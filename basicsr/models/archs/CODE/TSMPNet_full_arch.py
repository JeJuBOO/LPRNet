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
from basicsr.models.archs.arch_util import LayerNorm2d, patch_partition, patch_reverse, check_image_size, calculate_patch_psnr, calculate_patch_qualities
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
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, 
                               padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=k, 
                               padding=k // 2, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1,
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
        

        if not self.nafnetmode:
            cf_channel = 2 * c # Combined Feature Channel

            self.global_path = nn.Sequential(
                UpperNAFBlock(c, dwconv_kernel_size)
            )
            self.norm2 = LayerNorm2d(cf_channel)
            self.conv1 = nn.Conv2d(in_channels=cf_channel, out_channels=c, kernel_size=1,
                                   padding=0, stride=1, groups=1, bias=True)
        
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c

        self.norm3 = LayerNorm2d(c)

        self.conv2 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1,
                               padding=0, stride=1, groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, 
                               padding=0, stride=1, groups=1, bias=True)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        if self.nafnetmode:
            local_features = self.local_path(x)
            feature = local_features
        else:
            local_features = self.local_path(x)
            global_features = self.global_path(x)

            combined_feature = torch.cat([local_features, global_features], dim=1)
            combined_feature = self.norm2(combined_feature)
            feature = self.conv1(combined_feature)

        feature = self.dropout1(feature)
        x = inp + feature * self.beta

        x1 = self.norm3(x)
        x1 = self.conv2(x1)
        x1 = self.sg(x1)
        x1 = self.conv3(x1)

        x1 = self.dropout2(x1)

        output = x + x1 * self.gamma

        return output


# -------------------------   Second Stage Block -----------------------------
class SAM(nn.Module):
    def __init__(self, chan, out_chan):
        super().__init__()
        img_channel = 3

        self.sam_conv1 = nn.Conv2d(chan, out_chan, 1, bias=True)
        self.sam_conv3 = nn.Conv2d(img_channel, out_chan, 1, bias=True)

        self.ending = nn.Conv2d(in_channels=chan, out_channels=img_channel, kernel_size=1, 
                                padding=0, stride=1, groups=1, bias=True)

    def min_max_scaling(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def forward(self, x, x_img, target=None, qualities_weight=False, psnr_weight=False):
        B,_,_,_ = x.shape

        feature_patch = check_image_size(x, [32,32])
        feature_patch = patch_partition(feature_patch, [32,32])
        
        x1 = self.sam_conv1(feature_patch)

        first_out_img = self.ending(x) + x_img
        
        out_patch = check_image_size(first_out_img, [32,32])
        
        out_patch = patch_partition(out_patch, [32,32])

        x2 = torch.sigmoid(self.sam_conv3(out_patch)) # (Batch*patchnum,C,pH,pW)

        _,pC,pH,pW = out_patch.shape

        if qualities_weight:
            target_patch = check_image_size(target, [32,32])
            target_patch = patch_partition(target_patch, [32,32])
            
            qualities_weight = calculate_patch_qualities(out_patch).to(x2.device)
            qualities_weight = torch.sigmoid(qualities_weight)
            qualities_weight = qualities_weight.unsqueeze(-1).unsqueeze(-1)
            attention_weight = (1-qualities_weight) * x2
        
        elif psnr_weight:  
            target_patch = check_image_size(target, [32,32])
            target_patch = patch_partition(target_patch, [32,32])

            psnr_weight = calculate_patch_psnr(out_patch.view(B,-1,pC,pH,pW),target_patch.view(B,-1,pC,pH,pW)).to(x2.device) # (Batch*patchnum,1)
            psnr_weight = self.min_max_scaling(psnr_weight)
            psnr_weight = psnr_weight.unsqueeze(-1).unsqueeze(-1)# (Batch*patchnum,1,1,1)
            attention_weight = (1-psnr_weight) * x2

        else:
            attention_weight = x2
        
        first_feature = x1*attention_weight
        return first_feature, first_out_img
    
class PatchBlock(nn.Module):
    def __init__(self, c, drop_out_rate=0.):
        super().__init__()

        self.send_spatial_attention = False

        self.channel = c
        cf_channel = 2 * c # Combine Feature Channel

        self.pat1_conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                                    groups=1, bias=True)
        self.pat1_dconv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=3 // 2, stride=1, 
                                    groups=c, bias=True)
        self.pat1_conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                                    groups=1, bias=True)

        self.pat2_conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, 
                                    groups=1, bias=True)
        self.pat2_dconv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=3 // 2, stride=1, 
                                    groups=c, bias=True) 
        self.pat2_conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                                    groups=1, bias=True)
        
        self.conv1 = nn.Conv2d(in_channels=cf_channel, out_channels=c, kernel_size=1, padding=0, stride=1, 
                               groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        
        self.norm = LayerNorm2d(c)
        self.norm1 = LayerNorm2d(cf_channel)
        self.norm2 = LayerNorm2d(c)

        self.gelu = nn.GELU()

        self.channel_interaction_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 2, kernel_size=1),
            nn.BatchNorm2d(c // 2),
            nn.GELU(),
            nn.Conv2d(c // 2, c, kernel_size=1),
        )

        self.channel_interaction_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 2, kernel_size=1),
            nn.BatchNorm2d(c // 2),
            nn.GELU(),
            nn.Conv2d(c // 2, c, kernel_size=1),
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

        if isinstance(first_stage_feat, tuple):
            first_stage_feat = first_stage_feat[0]

        B_P, C, pH, pW = x.shape
               
        if self.training:
            inp_batch = 4  # train mode
        else:
            inp_batch = 1 

        patch_num = (B_P // inp_batch) 

        inp = x

        x = self.norm(x)
                
        x_pat1 = self.pat1_conv1(x)
        x_pat1 = self.pat1_dconv2(x_pat1)
        x_pat1 = self.gelu(x_pat1)

        interation_output = self.channel_interaction_1(first_stage_feat).unsqueeze(1) # (B, 1, C, 1, 1)
            
        x_pat1 = x_pat1.reshape(B_P // patch_num, patch_num, C, pH, pW)
        x_pat1 = x_pat1 * interation_output 
        x_pat1 = x_pat1.view(-1, C, pH, pW)

        x_pat1 = self.pat1_conv3(x_pat1)

        x_pat2 = self.pat2_conv1(x)
        x_pat2 = self.pat2_dconv2(x_pat2)
        x_pat2 = self.gelu(x_pat2)
        x_pat2 = x_pat2 * self.channel_interaction_2(x_pat2)
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
    
# -------------------------      Network Part     ------------------------------

class TSMPNet(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], enc_blk_nums=[], dec_blk_nums=[],
                dwconv_kernel_size=7):
        super().__init__()
        self.psnr_attention = False
        self.qualities_attention = True

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.intro2 = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)
        
        self.ending2 = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)
        
        self.feat_conv = nn.Conv2d(in_channels=width*2, out_channels=width, kernel_size=1, padding=0, stride=1, bias=True)

        self.feat_norm = LayerNorm2d(width*2)

        self.first_stage_encoders = nn.ModuleList()
        self.first_stage_decoders = nn.ModuleList()
        self.first_stage_middle_blks = nn.ModuleList()

        self.second_stage_encoders = nn.ModuleList()
        self.second_stage_decoders = nn.ModuleList()
        self.second_stage_middle_blks = nn.ModuleList()
        
        self.enc_channel_interaction = nn.ModuleList()

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

        self.sam = SAM(chan,width)
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
                      nn.Conv2d(chan// 2, chan // 2, kernel_size=1),
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
                  nn.Conv2d(chan// 2, chan // 2, kernel_size=1),
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

        self.padder_size = 2 ** len(self.first_stage_encoders)


    # -------------------    Foward stage  ---------------------
    def forward(self, inp, target):
        in_B, in_C, in_H, in_W = inp.shape
        inp = check_image_size(inp,[self.padder_size,self.padder_size])

        # ------------ First stage ------------
        x = inp
        x = self.intro(x)
        first_enc_features = []  # 첫번째 enc stage의 특징맵 저장
        first_dec_features = []  # 첫번째 dec stage의 특징맵 저장

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
        
        first_feature, first_out_img = self.sam(x, inp, target=target, qualities_weight=self.qualities_attention, psnr_weight=self.psnr_attention)

        # ------------ Second stage ------------
        inp = check_image_size(inp, [32, 32])  # (B, 2*C, H_pad, W_pad) 이후 H_pad=H
        inp = patch_partition(inp, [32, 32])  # (B*P, 2*C, pH, pW)
        x = self.intro2(inp)
        x = torch.concat((x, first_feature), 1)

        x = self.feat_norm(x)
        x = self.feat_conv(x)

        encs2 = []
        
        
        for s_encoder, down, first_skip_feature in zip(self.second_stage_encoders, self.second_stage_downs, first_dec_features[::-1]):
            for block in s_encoder:  
                x = block(x, first_skip_feature)  
            encs2.append(x)
            x = down(x)

        B, C, H, W = x.shape

        for block in self.second_stage_middle_blks: 
            x = block(x, mid_feature) 

        for decoder, up, enc_skip, dec_feature in zip(self.second_stage_decoders, self.second_stage_ups, encs2[::-1], first_dec_features):
            x = up(x)
            x = x + enc_skip
            for block in decoder:
                x = block(x, dec_feature)

        x = self.ending2(x)

        x = patch_reverse(x, in_H, in_W)
        second_out_img = x + first_out_img

        return second_out_img# , first_out_img


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

    print(macs, params)
