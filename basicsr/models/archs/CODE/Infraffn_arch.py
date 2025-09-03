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
    
def window_partition2(x, window_size):
    """
    Args:
        x: (B, C, H, W)  pytorch的卷积默认tensor格式为(B, C, H, W)
        window_size (tuple[int]): window size(M)
    Returns:
        windows: (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    # view: -> [B, C, H//Wh, Wh, W//Ww, Ww]
    x = x.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    # permute: -> [B, H//Wh, W//Ww, Wh, Ww, C]
    # view: -> [B*num_windows, Wh, Ww, C]
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size[0] * window_size[1], C)
    return windows
def window_reverse2(windows, window_size, B, H: int, W: int):
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

    # B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # view: [B*num_windows, N, C] -> [B, H//window_size, W//window_size, window_size, window_size, C]
    x = windows.view(B, H_padded // window_size[0], W_padded // window_size[1], window_size[0], window_size[1], -1)
    # permute: [B, H//Wh, W//Ww, Wh, Ww, C] -> [B, C, H//Wh, Wh, W//Ww, Ww]
    # view: [B, C, H//Wh, Wh, W//Ww, Ww] -> [B, C, H, W]
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H_padded, W_padded)
    x = x[:, :, :H, :W]
    return x
def window_reverse(windows, window_size: int, H: int, W: int):
    """
    num_windows = H//Wh * W//Ww
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # view: [B*num_windows, Wh, Ww, C] -> [B, H//Wh, W//Ww, Wh, Ww, C]
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    # permute: [B, H//Wh, W//Ww, Wh, Ww, C] -> [B, H//Wh, Wh, W//Ww, Ww, C]
    # view: [B, H//Wh, Wh, W//Ww, Ww, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
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

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)
    
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
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
      
class InfraFFN_Block(nn.Module):
    def __init__(self, dim, window_size, dwconv_kernel_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., drop_path=0., mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        attn_dim = dim

        self.window_size = window_size
        self.dwconv_kernel_size = dwconv_kernel_size
        self.num_heads = num_heads
        head_dim = attn_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        relative_coords = self._get_rel_pos()
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.proj_attn = nn.Linear(dim, dim)
        self.proj_attn_norm = nn.LayerNorm(dim)

        self.proj_cnn_1 = nn.Linear(dim, dim)
        self.proj_cnn_norm_1 = nn.LayerNorm(dim)

        self.proj_cnn_2 = nn.Linear(dim, dim)
        self.proj_cnn_norm_2 = nn.LayerNorm(dim)

        # conv branch
        self.dwconv3x3 = nn.Sequential(
            nn.Conv2d(
                dim, dim,
                kernel_size=self.dwconv_kernel_size,
                padding=self.dwconv_kernel_size // 2,
                groups=dim
            ),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # conv branch
        self.dwconv5x5 = nn.Sequential(
            nn.Conv2d(
                dim, dim,
                kernel_size=5,
                padding=5 // 2,
                groups=dim
            ),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=1),

        )


        self.projection_1 = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.projection_2 = nn.Conv2d(dim, dim // 2, kernel_size=1)

        self.conv_norm_1 = nn.BatchNorm2d(dim // 2)
        self.conv_norm_2 = nn.BatchNorm2d(dim // 2)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1)
        )

        self.attn_norm = nn.LayerNorm(dim)
        self.conv_norm = nn.LayerNorm(dim)
        # final projection
        self.projection_cat = nn.Linear(dim * 2, dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

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
        
        inp, original_h, original_w = check_patch_size(inp, self.window_size)
        B, C, H, W = inp.shape
        
        x = window_partition2(inp, self.window_size)
        # (num_windows*B, window_size*window_size, C) 

        shortcut = x
        x_atten = self.proj_attn_norm(self.proj_attn(x))

        x_cnn_1 = self.proj_cnn_norm_1(self.proj_cnn_1(x))
        x_cnn_1 = window_reverse2(x_cnn_1, self.window_size, B, original_h, original_w) # (B, C, H, W)

        x_cnn_2 = self.proj_cnn_norm_2(self.proj_cnn_2(x))
        x_cnn_2 = window_reverse2(x_cnn_2, self.window_size, B, original_h, original_w) # (B, C, H, W)


        # conv branch
        # (B, C, H, W)
        x_cnn_1 = self.dwconv3x3(x_cnn_1)

        x_cnn_1 = self.projection_1(x_cnn_1)

        x_cnn_2 = self.dwconv5x5(x_cnn_2)
        x_cnn_2 = self.projection_2(x_cnn_2)

        x_cnn_1 = self.conv_norm_1(x_cnn_1)
        x_cnn_2 = self.conv_norm_2(x_cnn_2)

        x_cnn = torch.cat([x_cnn_1, x_cnn_2], dim=1)
        channel_interaction = self.channel_interaction(x_cnn)

        B_, N, C = x_atten.shape

        qkv = self.qkv(x_atten).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # q (batch*num_patch, num_head, patch_size^2, channel)
        # num head :multi head attention 개수
        
        # channel interaction
        x_cnn2v = torch.sigmoid(channel_interaction).reshape(-1, 1, self.num_heads, 1, C // self.num_heads)
        v = v.reshape(x_cnn2v.shape[0], -1, self.num_heads, N, C // self.num_heads)
        v = v * x_cnn2v
        v = v.reshape(-1, self.num_heads, N, C // self.num_heads)

        q = q * self.scale  # Q/sqrt(dk)
        attn = (q @ k.transpose(-2, -1))  # Q*K^{T} / sqrt(dk)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [num_heads, N, N]
        attn = attn + relative_position_bias.unsqueeze(0)

        
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x_atten = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # spatial interaction
        x_spatial = window_reverse2(x_atten, self.window_size, B, original_h, original_w)

        spatial_interaction = self.spatial_interaction(x_spatial)

        x_cnn = torch.sigmoid(spatial_interaction) * x_cnn

        x_cnn = window_partition2(x_cnn, self.window_size)
        x_cnn = self.conv_norm(x_cnn)

        # concat
        x_atten = self.attn_norm(x_atten)

        x = torch.cat([x_cnn, x_atten], dim=2)
        x = self.projection_cat(x)

        # proj: -> [num_windows*B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        # view(): -> [num_windows*B, window_size, window_size, C]
        # x = x.view(-1, self.window_size[0], self.window_size[1], C)
        # window_reverse(): -> [B, Hp, Wp, C]
        # x = window_reverse(x, self.window_size, H, W)

        x = shortcut + self.drop_path(x)
        # mlp: -> [B, H*W, C]
        # +: -> [B, H*W, C]
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        x = window_reverse2(x, self.window_size, B, original_h, original_w)

        return x

class Infraffn_Net(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], patch_size=[16, 16], dwconv_kernel_size=7,
                 num_heads=[2, 4, 8, 16]):
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
                    *[InfraFFN_Block(chan,
                                    patch_size,
                                    dwconv_kernel_size,
                                    num_heads[i]) for _ in range(enc_blk_nums[i])]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[InfraFFN_Block(chan,
                                    patch_size,
                                    dwconv_kernel_size,
                                    num_heads[-1]) for _ in range(middle_blk_num)]
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
    
class Infraffn_NetLocal(Local_Base, Infraffn_Net):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        Infraffn_Net.__init__(self, *args, **kwargs)

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
    
    net = Infraffn_Net(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
    