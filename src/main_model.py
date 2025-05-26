import torch
import torch.nn as nn
import torch.nn.functional as F
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
        weights = F.softmax(self.alpha, dim=0)
        out = sum(w * o for w, o in zip(weights, conv_outputs))
        return self.ffn(self.norm(out)) + x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, stride_size, padding=0):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=stride_size, padding=padding)

    def forward(self, x):
        return self.proj(x)

class main_model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        input_shape = (self.params['input_channels'], 251, self.params['nb_mels'])  # [B, C=2, T=251, F]
        cnn_filters = self.params['nb_conv_filters']
        dropout_rate = self.params['dropout']
        pool_sizes = self.params['f_pool_size']
        patch_size = self.params['patch_size']
        stride_size = self.params['stride_size']
        embed_dim = self.params['embed_dim']
        patch_padding = self.params.get('patch_padding', patch_size // 2)
        pooling_length = self.params.get('pooling_length', 50)
        fc_dims = self.params.get('fc_dims', [256, 128])
        dilation_sets = self.params['tcn_dilation_sets']

        self.nb_classes = self.params['nb_classes']
        self.max_polyphony = self.params['max_polyphony']

        # CNN Frontend
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

        self.patch_embed = PatchEmbedding(
            in_channels=cnn_output_dim,
            embed_dim=embed_dim,
            patch_size=patch_size,
            stride_size=stride_size,
            padding=patch_padding
        )
        self.patch_norm = nn.BatchNorm1d(embed_dim)

        self.tcn_blocks = nn.Sequential(*[
            HybridTCNBlock(embed_dim, hidden_dim=embed_dim * 4, dilations=dset) for dset in dilation_sets
        ])

        final_output_dim = self.max_polyphony * 3 * self.nb_classes
        fc_layers = []
        in_dim = embed_dim
        for d in fc_dims:
            fc_layers.extend([nn.Linear(in_dim, d), nn.ReLU()])
            in_dim = d
        fc_layers.append(nn.Linear(in_dim, final_output_dim))
        self.output_linear = nn.Sequential(*fc_layers)

        self.dao_act = nn.Tanh()
        self.dist_act = nn.ReLU()
        
        self.pooling_length = pooling_length

    def forward(self, x, video_features=None):
        x = self.cnn_layers(x)                      # [B, C, T, F']
        x = x.permute(0, 2, 1, 3)                   # [B, T, C, F']
        B, T, C, freq = x.shape
        x = x.reshape(B, T, C * freq).permute(0, 2, 1)  # [B, C*F', T]
        x = self.patch_embed(x)
        x = self.patch_norm(x)
        x = self.tcn_blocks(x)
        x = F.adaptive_avg_pool1d(x, self.pooling_length)
        x = x.permute(0, 2, 1)
        x = self.output_linear(x)

        B, T, _ = x.shape
        x = x.reshape(B, T, 3, 3, 13)
        doa = x[:, :, :, 0:2, :]
        dist = x[:, :, :, 2:3, :]
        
        doa = self.dao_act(doa)
        dist = self.dist_act(dist)
        
        x = torch.cat([doa, dist], dim = 3)
        x = x.view(B, T, -1)
        return x

