# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np


# class ResNetEncoder(nn.Module):
#     def __init__(self, in_channels=7):
#         super(ResNetEncoder, self).__init__()
#         def conv_block(in_c, out_c):
#             return nn.Sequential(
#                 nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_c),
#                 nn.ReLU(inplace=True)
#             )

#         self.block1 = conv_block(in_channels, 64)
#         self.block2 = conv_block(64, 64)
#         self.block3 = conv_block(64, 128)
#         self.block4 = conv_block(128, 128)

#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         return x  # (B, 128, 250, 64)

# # --- 2. Positional Encoding ---
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=500):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1)].to(x.device)
#         return x

# # --- 3. 개선된 Conformer Block ---
# class ConformerBlock(nn.Module):
#     def __init__(self, d_model, n_heads=4, ff_multiplier=4):
#         super(ConformerBlock, self).__init__()
#         self.layer_norm1 = nn.LayerNorm(d_model)
#         self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

#         self.layer_norm2 = nn.LayerNorm(d_model)
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_model * ff_multiplier),
#             nn.GELU(),
#             nn.Linear(d_model * ff_multiplier, d_model)
#         )

#     def forward(self, x):
#         res = x
#         x = self.layer_norm1(x)
#         x, _ = self.self_attn(x, x, x)
#         x = res + x

#         res = x
#         x = self.layer_norm2(x)
#         x = self.ffn(x)
#         return res + x

# # --- 4. 전체 SELD 모델 ---
# class ResNetConformerSELDModel(nn.Module):
#     def __init__(self, 
#                  num_sed_classes=13, 
#                  doa_output_dim=39, 
#                  fc_proj_dim=256, 
#                  conformer_layers=8):
#         super(ResNetConformerSELDModel, self).__init__()
#         self.resnet = ResNetEncoder(in_channels=2)
#         self.fc_proj = nn.Linear(128 * 64, fc_proj_dim)
#         self.pos_encoding = PositionalEncoding(fc_proj_dim)
#         self.conformers = nn.ModuleList([ConformerBlock(fc_proj_dim) for _ in range(conformer_layers)])
#         self.time_pool = nn.AdaptiveAvgPool1d(50)

#         self.shared_fc = nn.Sequential(
#             nn.LayerNorm(fc_proj_dim),
#             nn.Linear(fc_proj_dim, fc_proj_dim),
#             nn.SELU()
#         )

#         self.sed_branch = nn.Sequential(
#             nn.Linear(fc_proj_dim, num_sed_classes),
#             nn.Sigmoid()
#         )

#         self.doa_branch = nn.Sequential(
#             nn.Linear(fc_proj_dim, fc_proj_dim),
#             nn.ReLU(),
#             nn.Linear(fc_proj_dim, doa_output_dim),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         # x: (B, 7, 250, 64)
#         x = self.resnet(x)  # (B, 128, 250, 64)
#         B, C, T, F = x.shape
#         x = x.permute(0, 2, 1, 3).contiguous().view(B, T, -1)  # (B, 250, 128*64)
#         x = self.fc_proj(x)  # (B, 250, 256)
#         x = self.pos_encoding(x)

#         for layer in self.conformers:
#             x = layer(x)

#         x = x.transpose(1, 2)  # (B, 256, 250)
#         x = self.time_pool(x).transpose(1, 2)  # (B, 50, 256)
#         x = self.shared_fc(x)  # (B, 50, 256)

#         sed_out = self.sed_branch(x)  # (B, 50, 13)
#         doa_out = self.doa_branch(x)  # (B, 50, 39)
#         return sed_out, doa_out


#////////////////////////////////////////////////// sed-doa model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import Conformer
#from model_fdy import DYCNN  # DYCRNN 안에 있는 DYCNN을 import

import torch
import torch.nn as nn

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample

#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += identity
#         out = self.relu(out)
#         return out


# def make_layer(block, in_channels, out_channels, num_blocks, stride=1):
#     downsample = None
#     if stride != 1 or in_channels != out_channels:
#         downsample = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#             nn.BatchNorm2d(out_channels)
#         )
#     layers = [block(in_channels, out_channels, stride, downsample)]
#     for _ in range(1, num_blocks):
#         layers.append(block(out_channels, out_channels))
#     return nn.Sequential(*layers)


# class ResNet18Encoder(nn.Module):
#     def __init__(self, in_channels=2):
#         super(ResNet18Encoder, self).__init__()
#         # No downsampling here
#         self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)

#         # Downsample once here: 251x64 -> ~125x32
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         # All subsequent layers have stride=1 to keep shape
#         self.layer1 = make_layer(BasicBlock, 64, 64, num_blocks=2, stride=1)
#         self.layer2 = make_layer(BasicBlock, 64, 128, num_blocks=2, stride=1)
#         self.layer3 = make_layer(BasicBlock, 128, 256, num_blocks=2, stride=1)
#         self.layer4 = make_layer(BasicBlock, 256, 512, num_blocks=2, stride=1)

#         # Final downsampling to exactly (63, 16)
#         self.pool = nn.AdaptiveAvgPool2d((63, 16))

#     def forward(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))  # (B, 64, 251, 64)
#         x = self.maxpool(x)                     # (B, 64, ~125, 32)
#         x = self.layer1(x)                      # (B, 64, ~125, 32)
#         x = self.layer2(x)                      # (B, 128, ~125, 32)
#         x = self.layer3(x)                      # (B, 256, ~125, 32)
#         x = self.layer4(x)                      # (B, 512, ~125, 32)
#         x = self.pool(x)                        # (B, 512, 63, 16)
#         return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # downsample if needed (channel mismatch or stride > 1)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out
# class ResNetEncoder(nn.Module):
#     def __init__(self, in_channels=2):
#         super(ResNetEncoder, self).__init__()

#         def conv_block(in_c, out_c):
#             return nn.Sequential(
#                 nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_c),
#                 nn.ReLU(inplace=True)
#             )

#         self.block1 = conv_block(in_channels, 64)
#         self.block2 = conv_block(64, 64)
#         self.block3 = conv_block(64, 128)
#         self.block4 = conv_block(128, 128)

#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         return x  # (B, 128, 250, 64)
class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=2):
        super(ResNetEncoder, self).__init__()
        self.layer1 = BasicBlock(in_channels, 64)
        self.layer2 = BasicBlock(64, 64)
        self.layer3 = BasicBlock(64, 128,stride=2)
        self.layer4 = BasicBlock(128, 128)

    def forward(self, x):
        x = self.layer1(x)  # (B, 64, T, F)
        x = self.layer2(x)  # (B, 64, T, F)
        x = self.layer3(x)  # (B, 128, T, F)
        x = self.layer4(x)  # (B, 128, T, F)
        return x

class ResNetConformerSELDModel(nn.Module):
    def __init__(self, 
                 num_sed_classes=13, 
                 doa_output_dim=39, 
                 fc_proj_dim=256,
                 conformer_layers=8):
        super(ResNetConformerSELDModel, self).__init__()

        self.resnet = ResNetEncoder(in_channels=2)
        # self.res18 =ResNet18Encoder()
        self.fc_proj = nn.Linear(128 * 32, fc_proj_dim)

        self.conformer = Conformer(
            input_dim=fc_proj_dim,
            num_heads=8,
            ffn_dim=fc_proj_dim * 4,
            num_layers=conformer_layers,
            depthwise_conv_kernel_size=31,
            dropout=0.1,
        )

        self.time_pool = nn.AdaptiveAvgPool1d(50)

        self.shared_fc = nn.Sequential(
            nn.LayerNorm(fc_proj_dim),
            nn.Linear(fc_proj_dim, fc_proj_dim),
            nn.SELU()
        )

        self.sed_branch = nn.Sequential(
            nn.Linear(fc_proj_dim, num_sed_classes),
            nn.Sigmoid()
        )

        self.doa_branch = nn.Sequential(
            nn.Linear(fc_proj_dim, fc_proj_dim),
            nn.ReLU(),
            nn.Linear(fc_proj_dim, doa_output_dim),
            nn.Tanh()
        )
#         self.dycnn = DYCNN(
#         n_input_ch=2,
#         n_filt=[64, 64, 128, 128],       # 최종 output channel = 128
#         DY_layers=[1, 1, 1, 1],          # 전부 DynamicConv 사용
#         pooling=[(1, 1), (1, 1), (1, 1), (1, 1)],  # pooling 없음
#         kernel=[3, 3, 3, 3],
#         pad=[1, 1, 1, 1],                # padding으로 feature map 크기 유지
#         stride=[1, 1, 1, 1],             # stride=1로 T/F 크기 유지
#         dilated_DY=[0, 0, 0, 0],
#         n_basis_kernels=4,
#         temperature=31,
#         pool_dim='freq',
#         pool_type='avg',
#         conv1d_kernel=[3, 1],
#         dy_chan_proportion=(1, 1),
# )

    def forward(self, x):
        x = self.resnet(x)  # (B, 128, 250, 64)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, -1)  # (B, 250, 128*64)
        x = self.fc_proj(x)  # (B, 250, 256)

        lengths = torch.full((x.size(0),), x.size(1), dtype=torch.int32).to(x.device)
        x, _ = self.conformer(x, lengths)  # (B, 250, 256)

        x = x.transpose(1, 2)  # (B, 256, 250)
        x = self.time_pool(x).transpose(1, 2)  # (B, 50, 256)
        x = self.shared_fc(x)  # (B, 50, 256)

        sed_out = self.sed_branch(x)  # (B, 50, 13)
        doa_out = self.doa_branch(x)  # (B, 50, 39)
        return sed_out, doa_out


#/////////////////////////////////////////////////////////////////////
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchaudio.models import Conformer
# from model_fdy import DYCNN  # DYCRNN 안에 있는 DYCNN을 import


# class DYCNNConformerSELDModel(nn.Module):
#     def __init__(self, 
#                  num_sed_classes=13, 
#                  doa_output_dim=39, 
#                  fc_proj_dim=256,
#                  conformer_layers=8):
#         super(DYCNNConformerSELDModel, self).__init__()

#         self.dycnn = DYCNN(
#             n_input_ch=2
#         )

#         # CNN 출력이 (B, 64, 250, 8)이 되므로, 64*8 = 512
#         self.fc_proj = nn.Linear(128 * 32, fc_proj_dim)

#         self.conformer = Conformer(
#             input_dim=fc_proj_dim,
#             num_heads=8,
#             ffn_dim=fc_proj_dim * 4,
#             num_layers=conformer_layers,
#             depthwise_conv_kernel_size=31,
#             dropout=0.1,
#         )

#         self.time_pool = nn.AdaptiveAvgPool1d(50)

#         self.shared_fc = nn.Sequential(
#             nn.LayerNorm(fc_proj_dim),
#             nn.Linear(fc_proj_dim, fc_proj_dim),
#             nn.SELU()
#         )

#         self.sed_branch = nn.Sequential(
#             nn.Linear(fc_proj_dim, num_sed_classes),
#             nn.Sigmoid()
#         )

#         self.doa_branch = nn.Sequential(
#             nn.Linear(fc_proj_dim, fc_proj_dim),
#             nn.ReLU(),
#             nn.Linear(fc_proj_dim, doa_output_dim),
#             nn.Tanh()
#         )

#     def forward(self, x):  # x: (B, 2, T, F)
#         x = self.dycnn(x)               # x: (B, 64, T, 8)
#         B, C, T, F = x.shape
#         x = x.permute(0, 2, 1, 3).contiguous().view(B, T, -1)  # (B, T, 64*8=512)

#         x = self.fc_proj(x)             # (B, T, 256)
#         lengths = torch.full((B,), x.size(1), dtype=torch.int32).to(x.device)
#         x, _ = self.conformer(x, lengths)   # (B, T, 256)

#         x = x.transpose(1, 2)                # (B, 256, T)
#         x = self.time_pool(x).transpose(1, 2)  # (B, 50, 256)

#         x = self.shared_fc(x)                # (B, 50, 256)

#         sed_out = self.sed_branch(x)         # (B, 50, num_sed_classes)
#         doa_out = self.doa_branch(x)         # (B, 50, doa_output_dim)

#         return sed_out, doa_out


# ////////////////////////////////////////////////// accdoa model

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchaudio.models import Conformer

# class ResNetEncoder(nn.Module):
#     def __init__(self, in_channels=2):
#         super().__init__()
#         def conv_block(in_c, out_c):
#             return nn.Sequential(
#                 nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_c),
#                 nn.ReLU(inplace=True)
#             )
#         self.block1 = conv_block(in_channels, 64)
#         self.block2 = conv_block(64, 64)
#         self.block3 = conv_block(64, 128)
#         self.block4 = conv_block(128, 128)

#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         return x  # (B, 128, 250, 64)

# class ResNetConformerACCDOAModel(nn.Module):
#     def __init__(self, num_classes=13, fc_proj_dim=256, conformer_layers=8):
#         super().__init__()
#         self.num_classes = num_classes
#         self.resnet = ResNetEncoder(in_channels=2)
#         self.fc_proj = nn.Linear(128 * 64, fc_proj_dim)

#         self.conformer = Conformer(
#             input_dim=fc_proj_dim,
#             num_heads=8,
#             ffn_dim=fc_proj_dim * 4,
#             num_layers=conformer_layers,
#             depthwise_conv_kernel_size=31,
#             dropout=0.1,
#         )

#         self.time_pool = nn.AdaptiveAvgPool1d(50)

#         self.shared_fc = nn.Sequential(
#             nn.LayerNorm(fc_proj_dim),
#             nn.Linear(fc_proj_dim, fc_proj_dim),
#             nn.SELU()
#         )

#         self.output_fc = nn.Linear(fc_proj_dim, num_classes * 3)  # x, y, dist per class

#     def forward(self, x):
#         B = x.size(0)
#         x = self.resnet(x)  # (B, 128, 250, 64)
#         B, C, T, Freq = x.shape
#         x = x.permute(0, 2, 1, 3).contiguous().view(B, T, -1)  # (B, 250, 128*64)
#         x = self.fc_proj(x)  # (B, 250, 256)

#         lengths = torch.full((B,), x.size(1), dtype=torch.int32).to(x.device)
#         x, _ = self.conformer(x, lengths)  # (B, 250, 256)

#         x = x.transpose(1, 2)  # (B, 256, 250)
#         x = self.time_pool(x).transpose(1, 2)  # (B, 50, 256)
#         x = self.shared_fc(x)  # (B, 50, 256)

#         accdoa_out = self.output_fc(x)  # (B, 50, 39)
#         accdoa_out = accdoa_out.view(B, 50, 3, self.num_classes)  # (B, 50, 3, 13)

#         doa_xy = torch.tanh(accdoa_out[:, :, 0:2, :])     # (B, 50, 2, 13)
#         dist = F.relu(accdoa_out[:, :, 2:3, :])           # (B, 50, 1, 13)

#         accdoa_out = torch.cat([doa_xy, dist], dim=2)     # (B, 50, 3, 13)
#         accdoa_out = accdoa_out.view(B, 50, -1)           # (B, 50, 39)
#         return accdoa_out


# class ResNetConformerBinaryModel(nn.Module):
#     def __init__(self, fc_proj_dim=256, conformer_layers=8):
#         super().__init__()
#         self.resnet = ResNetEncoder(in_channels=2)
#         self.fc_proj = nn.Linear(128 * 64, fc_proj_dim)

#         self.conformer = Conformer(
#             input_dim=fc_proj_dim,
#             num_heads=8,
#             ffn_dim=fc_proj_dim * 4,
#             num_layers=conformer_layers,
#             depthwise_conv_kernel_size=31,
#             dropout=0.1,
#         )

#         self.time_pool = nn.AdaptiveAvgPool1d(50)

#         self.shared_fc = nn.Sequential(
#             nn.LayerNorm(fc_proj_dim),
#             nn.Linear(fc_proj_dim, fc_proj_dim),
#             nn.SELU()
#         )

#         self.output_fc = nn.Linear(fc_proj_dim, 1)  # binary output

#     def forward(self, x):
#         B = x.size(0)
#         x = self.resnet(x)  # (B, 128, 250, 64)
#         B, C, T, Freq = x.shape
#         x = x.permute(0, 2, 1, 3).contiguous().view(B, T, -1)  # (B, 250, 128*64)
#         x = self.fc_proj(x)  # (B, 250, 256)

#         lengths = torch.full((B,), x.size(1), dtype=torch.int32).to(x.device)
#         x, _ = self.conformer(x, lengths)  # (B, 250, 256)

#         x = x.transpose(1, 2)  # (B, 256, 250)
#         x = self.time_pool(x).transpose(1, 2)  # (B, 50, 256)
#         x = self.shared_fc(x)  # (B, 50, 256)

#         x = self.output_fc(x)          # (B, 50, 1)
#         x = torch.sigmoid(x).squeeze(-1)  # (B, 50)
#         return x

#////////////////////////////////////////////////

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchaudio.models import Conformer
# from torchinfo import summary

# class CNNChannelEmbedding(nn.Module):
#     def __init__(self, in_channels=2, out_channels=64):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.act = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.act(x)
#         return x


# class AggressiveDownsampleCSTConformer(nn.Module):
#     def __init__(
#         self,
#         num_sed_classes=13,
#         doa_output_dim=39,
#         hidden_dim=256,
#         ch_num_layers=1,
#         fr_num_layers=1,
#         tm_num_layers=1,
#         out_channels=64,
#         time_pool_len=20
#     ):
#         super().__init__()

#         self.hidden_dim = hidden_dim
#         self.ch_num_layers = ch_num_layers
#         self.fr_num_layers = fr_num_layers
#         self.tm_num_layers = tm_num_layers

#         self.cnn_channel_embed = CNNChannelEmbedding(in_channels=2, out_channels=out_channels)

#         # lazy init placeholders
#         self.conformer_chan = None
#         self.conformer_freq = None
#         self.conformer_time = None

#         self.reduce_channel_1 = nn.Conv2d(out_channels, out_channels // 2, kernel_size=1)
#         self.reduce_channel_2 = nn.Conv2d(out_channels // 2, out_channels // 4, kernel_size=1)

#         self.time_pool_len = time_pool_len
#         self.fc_after_conformer = None

#         self.shared_fc = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SELU()
#         )

#         self.sed_branch = nn.Sequential(
#             nn.Linear(hidden_dim, num_sed_classes),
#             nn.Sigmoid()
#         )

#         self.doa_branch = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, doa_output_dim),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         B, _, T, FF = x.shape

#         x_cnn = self.cnn_channel_embed(x)
#         B, C, T, FF = x_cnn.shape

#         T2, FF2 = T // 4, FF // 4
#         x_cnn = F.adaptive_avg_pool2d(x_cnn, (T2, FF2))

#         # lazy init 채널 Conformer
#         if self.conformer_chan is None:
#             self.conformer_chan = Conformer(
#                 input_dim=T2 * FF2,
#                 num_heads=8,
#                 ffn_dim=self.hidden_dim * 4,
#                 num_layers=self.ch_num_layers,
#                 depthwise_conv_kernel_size=31,
#                 dropout=0.1
#             ).to(x.device)

#         x_chan = x_cnn.view(B, C, T2 * FF2)
#         length_chan = torch.full((B,), C, dtype=torch.int32, device=x.device)
#         x_chan_out, _ = self.conformer_chan(x_chan, length_chan)
#         x_chan_out = x_chan_out.view(B, C, T2, FF2)
#         x_chan_out = self.reduce_channel_1(x_chan_out)

#         B, C2, T2b, FF2b = x_chan_out.shape
#         T3, FF3 = T2b // 2, FF2b // 2
#         x_chan_out = F.adaptive_avg_pool2d(x_chan_out, (T3, FF3))

#         # lazy init 주파수 Conformer
#         if self.conformer_freq is None:
#             self.conformer_freq = Conformer(
#                 input_dim=C2 * T3,
#                 num_heads=8,
#                 ffn_dim=self.hidden_dim * 4,
#                 num_layers=self.fr_num_layers,
#                 depthwise_conv_kernel_size=31,
#                 dropout=0.1
#             ).to(x.device)

#         x_freq = x_chan_out.permute(0, 3, 1, 2).reshape(B, FF3, C2 * T3)
#         len_freq = torch.full((B,), FF3, dtype=torch.int32, device=x.device)
#         x_freq_out, _ = self.conformer_freq(x_freq, len_freq)
#         x_freq_out = x_freq_out.view(B, FF3, C2, T3).permute(0, 2, 3, 1)
#         x_freq_out = self.reduce_channel_2(x_freq_out)

#         B, C3, T4, FF4 = x_freq_out.shape
#         T5 = T4 // 2
#         x_freq_out = F.adaptive_avg_pool2d(x_freq_out, (T5, FF4))

#         # lazy init 시간 Conformer
#         if self.conformer_time is None:
#             self.conformer_time = Conformer(
#                 input_dim=C3 * FF4,
#                 num_heads=8,
#                 ffn_dim=self.hidden_dim * 4,
#                 num_layers=self.tm_num_layers,
#                 depthwise_conv_kernel_size=31,
#                 dropout=0.1
#             ).to(x.device)

#         x_time = x_freq_out.permute(0, 2, 1, 3).reshape(B, T5, C3 * FF4)
#         len_time = torch.full((B,), T5, dtype=torch.int32, device=x.device)
#         x_time_out, _ = self.conformer_time(x_time, len_time)

#         x_time_out = x_time_out.transpose(1, 2)
#         x_pool = F.adaptive_avg_pool1d(x_time_out, 50).transpose(1, 2)

#         final_dim = C3 * FF4
#         if self.fc_after_conformer is None:
#             self.fc_after_conformer = nn.Linear(final_dim, self.shared_fc[0].normalized_shape[0]).to(x.device)

#         x_fc = F.relu(self.fc_after_conformer(x_pool))
#         x_shared = self.shared_fc(x_fc)

#         sed_out = self.sed_branch(x_shared)
#         doa_out = self.doa_branch(x_shared)

#         return sed_out, doa_out


# # 사용 예시
# if __name__ == "__main__":
#     B, T, FF = 2, 240, 64
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     x = torch.randn(B, 2, T, FF).to(device)

#     model = AggressiveDownsampleCSTConformer(
#         hidden_dim=256,
#         ch_num_layers=1,
#         fr_num_layers=1,
#         tm_num_layers=1,
#         out_channels=64,
#         time_pool_len=30,
#     ).to(device)

#     summary(model, input_size=(B, 2, T, FF), device=device.type)

#     sed, doa = model(x)
#     print("sed:", sed.shape)
#     print("doa:", doa.shape)
#     for name, param in model.named_parameters(): print(name, param.shape, param.numel())
#     total_params = 0
#     for name, param in model.named_parameters():
#         pcount = param.numel()
#         total_params += pcount
#         print(f"{name}: {param.shape} -> {pcount}")

#     print("Total params:", total_params)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchaudio.models import Conformer
# from einops import rearrange

# class IRFFN(nn.Module):
#     def __init__(self, in_channels, expand_ratio=4):
#         super().__init__()
#         hidden_dim = in_channels * expand_ratio
#         self.expand = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
#         self.depthwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
#         self.project = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
#         self.activation = nn.SiLU()
#         self.norm = nn.BatchNorm2d(in_channels)

#     def forward(self, x):
#         residual = x
#         x = self.activation(self.expand(x))
#         x = self.activation(self.depthwise(x))
#         x = self.project(x)
#         return self.norm(x + residual)

# class PatchUnfoldConformer(nn.Module):
#     def __init__(self, cnn_channels, patch_size=(20, 8), hidden_dim=256):
#         super().__init__()
#         self.patch_size = patch_size
#         self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
#         self.fold = None  # lazy init
#         self.patch_area = patch_size[0] * patch_size[1]
#         self.conformer = Conformer(
#             input_dim=cnn_channels,
#             num_heads=8,
#             ffn_dim=hidden_dim * 4,
#             num_layers=1,
#             depthwise_conv_kernel_size=31,
#             dropout=0.1,
#             conv_module=IRFFN(cnn_channels)  # Inject IRFFN to reduce FFN overhead
#         )

#     def forward(self, x):
#         B, C, T, F = x.shape
#         x_unfold = self.unfold(x)  # (B, C*patch_area, N_patches)
#         patch_area = self.patch_area
#         num_patches = x_unfold.shape[-1]
#         x_unfold = x_unfold.contiguous().view(B * num_patches, C, patch_area)  # (B*N, C, patch_dim)

#         lengths = torch.full((B * num_patches,), patch_area, dtype=torch.int32, device=x.device)
#         x_out, _ = self.conformer(x_unfold.transpose(1, 2), lengths)  # (B*N, patch_dim, C)

#         x_out = x_out.transpose(1, 2).contiguous().view(B, -1, num_patches)  # (B, C*patch_dim, N_patches)

#         if self.fold is None:
#             patch_T, patch_F = self.patch_size
#             self.fold = nn.Fold(output_size=(T, F), kernel_size=self.patch_size, stride=self.patch_size).to(x.device)

#         x_folded = self.fold(x_out)  # (B, C, T, F)
#         x_flat = rearrange(x_folded, 'b c t f -> b t (f c)')  # (B, T, F*C)
#         return x_flat

# class ULECSTConformer(nn.Module):
#     def __init__(self, num_sed_classes=13, doa_output_dim=39, hidden_dim=256, cnn_channels=64, time_pool_len=50):
#         super().__init__()
#         self.time_pool_len = time_pool_len

#         self.cnn = nn.Sequential(
#             nn.Conv2d(2, cnn_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(cnn_channels),
#             nn.ReLU()
#         )

#         self.channel_attn = PatchUnfoldConformer(cnn_channels, patch_size=(20, 8), hidden_dim=hidden_dim)

#         self.spectral_attn = Conformer(input_dim=cnn_channels, num_heads=8, ffn_dim=hidden_dim * 4,
#                                        num_layers=1, depthwise_conv_kernel_size=31, dropout=0.1,
#                                        conv_module=IRFFN(cnn_channels))

#         self.temporal_attn = Conformer(input_dim=cnn_channels, num_heads=8, ffn_dim=hidden_dim * 4,
#                                        num_layers=1, depthwise_conv_kernel_size=31, dropout=0.1,
#                                        conv_module=IRFFN(cnn_channels))

#         self.fc_after_conformer = nn.Linear(cnn_channels, hidden_dim)
#         self.shared_fc = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SELU()
#         )

#         self.sed_branch = nn.Sequential(
#             nn.Linear(hidden_dim, num_sed_classes),
#             nn.Sigmoid()
#         )

#         self.doa_branch = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, doa_output_dim),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         B, _, T, F = x.shape
#         x = self.cnn(x)  # (B, C, T, F)

#         x = self.channel_attn(x)  # (B, T, F*C)

#         x_freq = rearrange(x, 'b t (f c) -> (b t) f c', f=F)
#         len_freq = torch.full((x_freq.shape[0],), F, dtype=torch.int32, device=x.device)
#         x_freq, _ = self.spectral_attn(x_freq, len_freq)
#         x = rearrange(x_freq, '(b t) f c -> b t (f c)', b=B, t=T)

#         x_temp = rearrange(x, 'b t (f c) -> (b f) t c', f=F)
#         len_temp = torch.full((x_temp.shape[0],), T, dtype=torch.int32, device=x.device)
#         x_temp, _ = self.temporal_attn(x_temp, len_temp)
#         x = rearrange(x_temp, '(b f) t c -> b t (f c)', b=B, f=F)

#         x = x.transpose(1, 2)  # (B, C, T)
#         x = F.adaptive_avg_pool1d(x, self.time_pool_len).transpose(1, 2)  # (B, time_pool_len, C)

#         x = F.relu(self.fc_after_conformer(x))
#         x = self.shared_fc(x)

#         sed = self.sed_branch(x)
#         doa = self.doa_branch(x)
#         return sed, doa

# if __name__ == "__main__":
#     model = ULECSTConformer()
#     dummy_input = torch.randn(2, 2, 240, 64)
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print("Total trainable parameters:", total_params)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F_torch
# from einops import rearrange
# from torchaudio.models.conformer import Conformer as TorchaudioConformer
# from torch.nn import Fold 
# from torchinfo import summary



# class ResNetEncoder(nn.Module):
#     def __init__(self, in_channels=2):
#         super().__init__()
#         def conv_block(in_c, out_c):
#             return nn.Sequential(
#                 nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_c),
#                 nn.ReLU(inplace=True)
#             )
#         self.block1 = conv_block(in_channels, 64)
#         self.block2 = conv_block(64, 64)

#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         return x  # (B, 128, T, F)



# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, patch_size=(10, 4), dropout=0.1, freq_bins=64):
#         super().__init__()
#         self.in_channels = in_channels
#         self.patch_size = patch_size
#         self.patch_area = patch_size[0] * patch_size[1]
#         self.freq_bins = freq_bins

#         # Patch 단위로 Unfold/Fold
#         self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
#         self.fold = None  # runtime 시에 device 맞춰서 생성

#         # 채널 방향 MHSA
#         self.mhsa = nn.MultiheadAttention(embed_dim=self.patch_area, num_heads=8, batch_first=True)

#         self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
#         self.norm = nn.LayerNorm(in_channels * freq_bins)

#     def forward(self, x):  # x: (B, C, T, F)
#         B, C, T, F = x.shape
#         assert F == self.freq_bins, f"Expected freq_bins={self.freq_bins}, but got F={F}"

#         # 1) Patch Unfold: (B, C * patch_area, N)
#         x_unf = self.unfold(x)
#         N = x_unf.shape[-1]

#         # 2) (B * N, C, patch_area)
#         x_unf = x_unf.view(B * N, C, self.patch_area)

#         # 3) MHSA across channel axis
#         attn_out, _ = self.mhsa(x_unf, x_unf, x_unf)  # (B * N, C, patch_area)

#         # 4) (B * N, C, patch_area) -> (B, C * patch_area, N)
#         attn_out = attn_out.reshape(B, -1, N)

#         # 5) Patch Fold: (B, C, T, F)
#         if self.fold is None:
#             self.fold = nn.Fold(output_size=(T, F), kernel_size=self.patch_size, stride=self.patch_size).to(x.device)
#         x_fold = self.fold(attn_out)

#         # 6) Residual + Dropout + Norm (Pre-Norm 구조)
#         x_flat = rearrange(x_fold, 'b c t f -> b t (f c)')
#         x_residual = rearrange(x, 'b c t f -> b t (f c)')

#         x_flat = self.dropout(x_flat)
#         x_flat = self.norm(x_flat + x_residual)

#         x_out = rearrange(x_flat, 'b t (f c) -> b c t f', f=F, c=C)
#         return x_out




# # class FrequencyAttentionCompressor(nn.Module):
# #     def __init__(self, in_channels, time_len=251, compressed_dim=256):
# #         super().__init__()
# #         self.time_len = time_len
# #         self.compressed_dim = compressed_dim
# #         self.input_dim = time_len * in_channels

# #         # (T×C) → d_model로 줄이기 위한 프로젝션
# #         self.proj_in = nn.Linear(self.input_dim, compressed_dim)

# #         # Conformer: F에 대해 attention
# #         self.conformer = TorchaudioConformer(
# #             input_dim=compressed_dim,
# #             num_heads=8,
# #             ffn_dim=1024,
# #             num_layers=2,
# #             depthwise_conv_kernel_size=31,
# #             dropout=0.1
# #         )

# #         # d_model → T×C로 복원
# #         self.proj_out = nn.Linear(compressed_dim, self.input_dim)

# #     def forward(self, x):  # x: (B, C, T, F)
# #         B, C, T, F = x.shape
# #         assert T == self.time_len, f"Expected time length {self.time_len}, got {T}"

# #         # Rearrange: (B, C, T, F) → (B, F, T×C)
# #         x = rearrange(x, 'b c t f -> b f (t c)')  # (B, F, T*C)

# #         # Projection to compressed_dim
# #         x_proj = self.proj_in(x)  # (B, F, d_model)
# #         x_residual = x_proj.clone()  # Save residual before Conformer

# #         # Conformer attention across frequency dimension
# #         lengths = torch.full((B,), F, dtype=torch.int32, device=x.device)
# #         x_out, _ = self.conformer(x_proj, lengths)  # (B, F, d_model)

# #         # Add residual
# #         x_out = x_out + x_residual  # (B, F, d_model)

# #         # Projection back to T×C
# #         x_out = self.proj_out(x_out)  # (B, F, T×C)

# #         # Rearrange: (B, F, T×C) → (B, C, T, F)
# #         x_out = rearrange(x_out, 'b f (t c) -> b c t f', t=T, c=C)

# #         return x_out

# class FrequencyAttentionCompressor(nn.Module):
#     def __init__(self, in_channels, time_len=251, freq_bins=64, dropout_rate=0.1):
#         super().__init__()
#         self.in_channels = in_channels
#         self.time_len = time_len
#         self.freq_bins = freq_bins
#         self.dropout_rate = dropout_rate

#         # Attention dim = in_channels = C
#         self.embed_dim = in_channels

#         # MHSA: Frequency attention
#         self.sp_mhsa = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=8, batch_first=True)

#         if dropout_rate:
#             self.drop_out = nn.Dropout(dropout_rate)

#         # LayerNorm over F×C dimension
#         self.sp_layer_norm = nn.LayerNorm(freq_bins * in_channels)

#     def forward(self, x):  # x: (B, C, T, F)
#         B, C, T, F = x.shape
#         assert F == self.freq_bins and C == self.embed_dim, f"Expected freq={self.freq_bins}, channels={self.embed_dim}"

#         # Save residual: (B, C, T, F) → (B, T, F*C)
#         xc = rearrange(x, 'b c t f -> b t (f c)').contiguous()

#         # Rearrange for MHSA: (B, C, T, F) → (B*T, F, C)
#         xs = rearrange(x, 'b c t f -> (b t) f c').contiguous()

#         # Apply MHSA
#         xs, _ = self.sp_mhsa(xs, xs, xs)  # (B*T, F, C)

#         # Restore shape: (B*T, F, C) → (B, T, F*C)
#         xs = rearrange(xs, '(b t) f c -> b t (f c)', t=T).contiguous()

#         # Residual + Dropout
#         xs = xs + xc
#         if self.dropout_rate:
#             xs = self.drop_out(xs)

#         # LayerNorm
#         xs = self.sp_layer_norm(xs)  # (B, T, F*C)

#         # Optional: reshape back to (B, C, T, F)
#         x_out = rearrange(xs, 'b t (f c) -> b c t f', f=F, c=C).contiguous()

#         return x_out


# class TemporalAttentionCompressor(nn.Module):
#     def __init__(self, in_channels, freq_bins=64, compressed_dim=256):
#         super().__init__()
#         self.freq_bins = freq_bins
#         self.input_dim = in_channels * freq_bins
#         self.compressed_dim = compressed_dim

#         # (C×F) → d_model로 축소
#         self.proj_in = nn.Linear(self.input_dim, compressed_dim)

#         # Conformer: T 축으로 attention
#         self.conformer = TorchaudioConformer(
#             input_dim=compressed_dim,
#             num_heads=8,
#             ffn_dim=1024,
#             num_layers=2,
#             depthwise_conv_kernel_size=31,
#             dropout=0.1
#         )

#         # d_model → C×F 복원
#         self.proj_out = nn.Linear(compressed_dim, self.input_dim)

#     def forward(self, x):  # x: (B, C, T, F)
#         B, C, T, F = x.shape
#         assert F == self.freq_bins, f"Expected F={self.freq_bins}, got {F}"

#         # Rearrange: (B, C, T, F) → (B, T, C×F)
#         x = rearrange(x, 'b c t f -> b t (c f)')  # (B, T, C*F)

#         # Projection to compressed_dim
#         x_proj = self.proj_in(x)  # (B, T, d_model)
#         x_residual = x_proj.clone()

#         # Attention across temporal axis
#         lengths = torch.full((B,), T, dtype=torch.int32, device=x.device)
#         x_out, _ = self.conformer(x_proj, lengths)  # (B, T, d_model)

#         # Residual connection
#         x_out = x_out + x_residual

#         # Project back
#         x_out = self.proj_out(x_out)  # (B, T, C×F)

#         # Rearrange back to original
#         x_out = rearrange(x_out, 'b t (c f) -> b c t f', c=C, f=F)

#         return x_out

# class CSTAttentionBlock(nn.Module):
#     def __init__(self, in_channels, time_len=251, freq_bins=64, compressed_dim=256):
#         super().__init__()

#         self.channel_attn = ChannelAttention(
#             in_channels=in_channels,        # 예: 128
#             patch_size=(10, 4),             # TF 압축용 패치 크기
#             dropout=0.1          # linear projection 사용 여부
#         )

#         # self.freq_attn = FrequencyAttentionCompressor(
#         #     in_channels=in_channels,
#         #     time_len=time_len,
#         #     compressed_dim=compressed_dim
#         # )
#         self.freq_attn = FrequencyAttentionCompressor(
#             in_channels=in_channels,
#             time_len=time_len,
#             freq_bins=freq_bins  # <-- 주파수 축 길이 명시적으로 넘겨야 함
#         )
#         self.temp_attn = TemporalAttentionCompressor(
#             in_channels=in_channels,
#             freq_bins=freq_bins,
#             compressed_dim=compressed_dim
#         )

#     def forward(self, x):  # (B, C, T, F)
#         x = self.channel_attn(x)
#         x = self.freq_attn(x)
#         x = self.temp_attn(x)
#         return x


# class CSTConformerModel(nn.Module):
#     def __init__(self, in_channels=2, base_channels=64, time_len=251, freq_bins=64,
#                  compressed_dim=256,
#                  num_sed_classes=13, doa_output_dim=39, time_pool_len=50):
#         super().__init__()

#         self.time_len = time_len
#         self.freq_bins = freq_bins

#         self.encoder = ResNetEncoder(in_channels)

#         self.lpu = nn.Sequential(
#             nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, groups=base_channels),
#             nn.SiLU()
#         )

#         self.cst_block1 = CSTAttentionBlock(
#             in_channels=base_channels,
#             time_len=time_len,
#             freq_bins=freq_bins,
#             compressed_dim=compressed_dim
#         )
#         self.cst_block2 = CSTAttentionBlock(
#             in_channels=base_channels,
#             time_len=time_len,
#             freq_bins=freq_bins,
#             compressed_dim=compressed_dim
#         )

#         self.time_pool_len = time_pool_len
#         self.fc_after_cst = nn.Sequential(
#             nn.Linear(base_channels * freq_bins, compressed_dim),
#             nn.LayerNorm(compressed_dim),
#             nn.SELU()
#         )

#         self.sed_branch = nn.Sequential(
#             nn.Linear(compressed_dim, num_sed_classes),
#             nn.Sigmoid()
#         )

#         self.doa_branch = nn.Sequential(
#             nn.Linear(compressed_dim, compressed_dim),
#             nn.ReLU(),
#             nn.Linear(compressed_dim, doa_output_dim),
#             nn.Tanh()
#         )

#     def forward(self, x):  # x: (B, 2, T, F)
#         B, _, T, F = x.shape
#         assert T == self.time_len and F == self.freq_bins, f"Input shape must be (B, 2, {self.time_len}, {self.freq_bins})"

#         x = self.encoder(x)  # (B, C, T, F)

#         residual = x
#         x = self.lpu(x) + residual  # LPU with residual

#         x = self.cst_block1(x)
#         x = self.cst_block2(x)

#         x = x.transpose(1, 2).reshape(B, T, -1)  # (B, T, C×F)
#         x = F_torch.adaptive_avg_pool1d(x.transpose(1, 2), self.time_pool_len).transpose(1, 2)
#         x = self.fc_after_cst(x)

#         sed = self.sed_branch(x)
#         doa = self.doa_branch(x)

#         return sed, doa













# #////////////////////////////////////////////////////////////////////////////

# class IRFFN1D(nn.Module):
#     def __init__(self, input_dim, expand_ratio=4):
#         super().__init__()
#         hidden_dim = input_dim * expand_ratio
#         self.expand = nn.Linear(input_dim, hidden_dim)
#         self.depthwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
#         self.project = nn.Linear(hidden_dim, input_dim)
#         self.activation = nn.SiLU()
#         self.norm = nn.LayerNorm(input_dim)

#     def forward(self, x):  # (B, T, D)
#         residual = x
#         x = self.activation(self.expand(x))  # (B, T, hidden)
#         x = x.transpose(1, 2)  # (B, hidden, T)
#         x = self.depthwise(x)
#         x = x.transpose(1, 2)  # (B, T, hidden)
#         x = self.project(x)
#         return self.norm(x + residual)

# # LPU 추가
# class LPU(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.depthwise_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
#         self.activation = nn.SiLU()

#     def forward(self, x):
#         return x + self.activation(self.depthwise_conv(x))  # (B, C, T, F)





# # IRFFN을 포함한 Conformer
# class ConformerWithIRFFN(TorchaudioConformer):
#     def __init__(self, input_dim, num_heads, ffn_dim, num_layers, depthwise_conv_kernel_size, dropout):
#         super().__init__(
#             input_dim=input_dim,
#             num_heads=num_heads,
#             ffn_dim=ffn_dim,
#             num_layers=num_layers,
#             depthwise_conv_kernel_size=depthwise_conv_kernel_size,
#             dropout=dropout,
#         )
#         for layer in self.conformer_layers:
#             layer.ffn1 = IRFFN1D(input_dim)
#             layer.ffn2 = IRFFN1D(input_dim)

# # 패치 기반 채널 어텐션
# class PatchUnfoldConformer(nn.Module):
#     def __init__(self, cnn_channels, patch_size=(20, 8), hidden_dim=256):
#         super().__init__()
#         self.patch_size = patch_size
#         self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
#         self.fold = None  # lazy init
#         self.patch_area = patch_size[0] * patch_size[1]
#         self.conformer = ConformerWithIRFFN(
#             input_dim=self.patch_area,
#             num_heads=8,
#             ffn_dim=hidden_dim * 4,
#             num_layers=2,
#             depthwise_conv_kernel_size=31,
#             dropout=0.1
#         )

#     def forward(self, x):
#         B, C, T, F = x.shape
#         x_unfold = self.unfold(x)  # (B, C*patch_area, N_patches)
#         patch_area = self.patch_area
#         num_patches = x_unfold.shape[-1]
#         x_unfold = x_unfold.contiguous().view(B * num_patches, C, patch_area)  # (B*N, C, patch_dim)

#         lengths = torch.full((B * num_patches,), C, dtype=torch.int32, device=x.device)
#         x_out, _ = self.conformer(x_unfold, lengths)  # (B*N, C, patch_dim)

#         x_out = x_out.reshape(B, -1, num_patches)  # (B, C*patch_dim, N_patches)

#         if self.fold is None:
#             patch_T, patch_F = self.patch_size
#             self.fold = nn.Fold(output_size=(T, F), kernel_size=self.patch_size, stride=self.patch_size).to(x.device)

#         x_folded = self.fold(x_out)  # (B, C, T, F)
#         x_flat = rearrange(x_folded, 'b c t f -> b t (f c)')  # (B, T, F*C)
#         return x_flat
    


# # 전체 모델
# class ULECSTConformer(nn.Module):
#     def __init__(self, num_sed_classes=13, doa_output_dim=39, hidden_dim=256, cnn_channels=64, time_pool_len=50):
#         super().__init__()
#         self.time_pool_len = time_pool_len

#         self.cnn = nn.Sequential(
#             nn.Conv2d(2, cnn_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(cnn_channels),
#             nn.ReLU()
#         )
#         self.lpu = LPU(cnn_channels)

#         self.channel_attn = PatchUnfoldConformer(40, patch_size=(10, 4), hidden_dim=hidden_dim)

#         self.spectral_attn = ConformerWithIRFFN(input_dim=cnn_channels, num_heads=8, ffn_dim=hidden_dim * 4,
#                                                 num_layers=2, depthwise_conv_kernel_size=31, dropout=0.1)

#         self.temporal_attn = ConformerWithIRFFN(input_dim=cnn_channels, num_heads=8, ffn_dim=hidden_dim * 4,
#                                                 num_layers=2, depthwise_conv_kernel_size=31, dropout=0.1)

#         self.fc_after_conformer = nn.Linear(4096, hidden_dim)
#         self.shared_fc = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SELU()
#         )

#         self.sed_branch = nn.Sequential(
#             nn.Linear(hidden_dim, num_sed_classes),
#             nn.Sigmoid()
#         )

#         self.doa_branch = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, doa_output_dim),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         B, _, T, F = x.shape
#         x = self.cnn(x)  # (B, C, T, F)
#         x = self.lpu(x)  # (B, C, T, F)

#         x = self.channel_attn(x)  # (B, T, F*C)

#         x_freq = rearrange(x, 'b t (f c) -> (b t) f c', f=F)
#         len_freq = torch.full((x_freq.shape[0],), F, dtype=torch.int32, device=x.device)
#         x_freq, _ = self.spectral_attn(x_freq, len_freq)
#         x = rearrange(x_freq, '(b t) f c -> b t (f c)', b=B, t=T)

#         x_temp = rearrange(x, 'b t (f c) -> (b f) t c', f=F)
#         len_temp = torch.full((x_temp.shape[0],), T, dtype=torch.int32, device=x.device)
#         x_temp, _ = self.temporal_attn(x_temp, len_temp)
#         x = rearrange(x_temp, '(b f) t c -> b t (f c)', b=B, f=F)

#         x = x.transpose(1, 2)  # (B, C, T)
#         x = F_torch.adaptive_avg_pool1d(x, self.time_pool_len).transpose(1, 2)  # (B, time_pool_len, C)

#         x = F_torch.relu(self.fc_after_conformer(x))
#         x = self.shared_fc(x)

#         sed = self.sed_branch(x)
#         doa = self.doa_branch(x)
#         return sed, doa
# import pandas as pd
# if __name__ == "__main__":
    
#     # 더미 입력
#     dummy_input = torch.randn(1, 2, 251, 64)

#     # 모델 선언
#     model = CSTConformerModel()

#     # 전체 파라미터 수 출력
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"✅ Total trainable parameters: {total_params:,}")

#     # 모듈별 파라미터 세기
#     param_details = []
#     for name, module in model.named_modules():
#         if any(p.requires_grad for p in module.parameters()):
#             module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
#             param_details.append((name, module_params))

#     # DataFrame 정리
#     param_df = pd.DataFrame(param_details, columns=["Module", "Trainable Parameters"])
#     param_df = param_df[param_df["Trainable Parameters"] > 0].sort_values(by="Trainable Parameters", ascending=False)

#     # 출력
#     print("\n📊 Parameter Breakdown:")
#     print(param_df.to_string(index=False))