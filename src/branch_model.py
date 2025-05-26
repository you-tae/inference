import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
from parameters import params

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=dilation, dilation=dilation,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class ConvFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Conv1d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv1d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class HybridTCNBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dilations):
        super().__init__()
        self.ms_convs = nn.ModuleList([
            DepthwiseSeparableConv1d(dim, dim, kernel_size=3, dilation=d) for d in dilations
        ])
        self.norm = nn.BatchNorm1d(dim)
        self.ffn = ConvFFN(dim, hidden_dim)
        self.alpha = nn.Parameter(torch.ones(len(dilations)))

    def forward(self, x):
        conv_outputs = [conv(x) for conv in self.ms_convs]
        weights = Func.softmax(self.alpha, dim=0)
        out = sum(w * o for w, o in zip(weights, conv_outputs))
        return self.ffn(self.norm(out)) + x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, stride_size, padding=0):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=stride_size, padding=padding)

    def forward(self, x):
        return self.proj(x)

class ClasswiseBranch(nn.Module):
    def __init__(self, in_dim, hidden_dim=27, out_dim=6):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.branch(x)

class main_model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        input_shape = (params['input_channels'], 251, params['nb_mels'])
        cnn_filters = params['nb_conv_filters']
        dropout_rate = params['dropout']
        pool_sizes = params['f_pool_size']
        patch_size = params['patch_size']
        stride_size = params['stride_size']
        embed_dim = params['embed_dim']
        patch_padding = params.get('patch_padding', patch_size // 2)
        pooling_length = params.get('pooling_length', 50)
        # fc_hidden = params.get('fc_hidden', 128)
        fc_hidden = 128  # Hardcoded for now, can be parameterized later
        # doa_branch_hidden = params.get('doa_branch_hidden', 27)
        doa_branch_hidden = 6 * 8 # Hardcoded for now, can be parameterized later
        # dist_branch_hidden = params.get('dist_branch_hidden', 27)        
        dist_branch_hidden = 64 # Hardcoded for now, can be parameterized later
        
        dilation_sets = params['tcn_dilation_sets']

        # self.nb_classes = params.get('nb_classes', 13)
        self.nb_classes = 13 # Hardcoded for now, can be parameterized later
        # self.tracks_per_class = params.get('tracks_per_class', 3)
        self.tracks_per_class = 3 # Hardcoded for now, can be parameterized later

        # CNN frontend
        self.cnn_layers = nn.Sequential()
        in_channels = input_shape[0]
        for i, pool in enumerate(pool_sizes):
            self.cnn_layers.add_module(f'conv2d_{i}', nn.Conv2d(in_channels, cnn_filters, kernel_size=3, padding=1))
            self.cnn_layers.add_module(f'bn_{i}', nn.BatchNorm2d(cnn_filters))
            self.cnn_layers.add_module(f'relu_{i}', nn.ReLU())
            self.cnn_layers.add_module(f'pool_{i}', nn.MaxPool2d(kernel_size=(1, pool)))
            self.cnn_layers.add_module(f'dropout_{i}', nn.Dropout2d(dropout_rate))
            in_channels = cnn_filters

        freq_reduced = int(input_shape[2] // np.prod(pool_sizes))
        cnn_output_dim = cnn_filters * freq_reduced

        self.patch_embed = PatchEmbedding(cnn_output_dim, embed_dim, patch_size, stride_size, patch_padding)
        self.patch_norm = nn.BatchNorm1d(embed_dim)

        self.tcn_blocks = nn.Sequential(*[
            HybridTCNBlock(embed_dim, embed_dim * 4, dilations=dset) for dset in dilation_sets
        ])

        self.pooling_length = pooling_length
        self.pre_branch = nn.Linear(embed_dim, fc_hidden)

        # Create class-specific DOA branches
        self.class_branches = nn.ModuleList([
            ClasswiseBranch(fc_hidden, doa_branch_hidden, self.tracks_per_class * 2) 
            for _ in range(self.nb_classes)
        ])

        # Distance head: shared output for all classes × tracks
        self.distance_head = nn.Sequential(
            nn.Linear(fc_hidden, dist_branch_hidden),
            nn.ReLU(),
            nn.Linear(dist_branch_hidden, self.nb_classes * self.tracks_per_class),
            nn.ReLU()
        )

        
        
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1, 3)
        B, T, C, F = x.shape
        x = x.reshape(B, T, C * F).permute(0, 2, 1)
        x = self.patch_embed(x)
        x = self.patch_norm(x)
        x = self.tcn_blocks(x)
        x = Func.adaptive_avg_pool1d(x, self.pooling_length)
        x = x.permute(0, 2, 1)  # [B, T, D]
        x = self.pre_branch(x)  # [B, T, fc_hidden]

        doa_outputs = [branch(x) for branch in self.class_branches]  # list of [B, T, 2×track]
        doa_out = torch.stack(doa_outputs, dim=2)  # [B, T, C, 2×track]
        doa = doa_out.view(B, 50, self.nb_classes, self.tracks_per_class, 2)                     # [B, T, C, track, 2]
        dist_out = self.distance_head(x)
        dist_out = dist_out.view(B, 50, self.nb_classes, self.tracks_per_class, 1)
        combined = torch.cat([doa, dist_out], dim=-1)  # [B, T, C, track, 3] (x, y, d)
        combined = combined.permute(0, 1, 3, 4, 2)
        out = combined.reshape(B, self.pooling_length, -1)  # ✅ view → reshape
        return out
