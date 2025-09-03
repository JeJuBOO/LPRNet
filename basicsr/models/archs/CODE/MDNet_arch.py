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


# from basicsr.metrics.psnr_ssim import calculate_psnr




class MDNet(nn.Module):

    def __init__(self, img_channel=3, width=32, first_middle_blk_num=1, first_enc_blk_nums=[], first_dec_blk_nums=[],
                 dwconv_kernel_size=7):
        super().__init__()

        self.freeze_intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)

        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3,
                                padding=1, stride=1, groups=1, bias=True)

        self.freeze_first_stage_encoders = nn.ModuleList()
        self.first_stage_decoders = nn.ModuleList()
        self.first_stage_middle_blks = nn.ModuleList()

        self.enc_channel_interaction = nn.ModuleList()
        self.dec_channel_interaction = nn.ModuleList()

        # First stage
        self.freeze_first_stage_downs = nn.ModuleList()
        self.first_stage_ups = nn.ModuleList()

        chan = width

        # -------------------    First stage  ---------------------
        for i in range(len(first_enc_blk_nums)):
            self.freeze_first_stage_encoders.append(
                nn.Sequential(
                    *[MDBlock(chan, dwconv_kernel_size) for _ in range(first_enc_blk_nums[i])]
                )
            )
            self.freeze_first_stage_downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.identity = nn.Identity()

        self.first_stage_middle_blks = \
            nn.Sequential(
                *[MDBlock(chan, dwconv_kernel_size) for _ in range(first_middle_blk_num)]
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
                    *[MDBlock(chan, dwconv_kernel_size) for _ in range(first_dec_blk_nums[i])]
                )
            )

        self.padder_size = 2 ** len(self.freeze_first_stage_encoders)

    # -------------------    Foward stage  ---------------------
    def forward(self, inp):
        in_B, in_C, in_H, in_W = inp.shape
        inp = check_image_size(inp, [self.padder_size, self.padder_size])

        x = inp
        x = self.freeze_intro(x.clone())
        first_enc_features = []  # 첫번째 enc stage의 특징맵 저장
        first_dec_features = []  # 첫번째 dec stage의 특징맵 저장

        for encoder, down in zip(self.freeze_first_stage_encoders, self.freeze_first_stage_downs):
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

        first_out_img = self.ending(x) + inp

        return first_out_img


class MDNet_Local(Local_Base, MDNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        MDNet.__init__(self, *args, **kwargs)

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

    first_enc_blks = [1, 1, 1, 1]
    first_middle_blk_num = 1
    first_dec_blks = [1, 1, 1, 1]



    # 모델을 래퍼로 감싸서 두 입력을 처리할 수 있게 수정
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            # 두 번째 입력을 생성
            return self.model(x)


    net = MDNet(img_channel=img_channel, width=width, first_middle_blk_num=first_middle_blk_num,
                  first_enc_blk_nums=first_enc_blks, first_dec_blk_nums=first_dec_blks,
                  dwconv_kernel_size=7)

    wrapped_net = ModelWrapper(net)  # 모델을 래퍼로 감싸기
    inp_shape = (3, 720, 1280)
    from torchinfo import summary
    summary(wrapped_net, input_size=(1, 3, 224, 224), depth=10)

    macs, params = get_model_complexity_info(wrapped_net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
