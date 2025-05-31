import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from parameters import params


# ------------------ Positional Encoding ------------------
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ------------------ Hybrid TCN Components ------------------
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, padding=dilation,
                                   dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ConvFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Conv1d(dim, hidden_dim, 1)
        self.fc2 = nn.Conv1d(hidden_dim, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class HybridTCNBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dilations):
        super().__init__()
        self.ms_convs = nn.ModuleList([
            DepthwiseSeparableConv1d(dim, dim, dilation=d) for d in dilations
        ])
        self.norm = nn.BatchNorm1d(dim)
        self.ffn = ConvFFN(dim, hidden_dim)
        self.alpha = nn.Parameter(torch.ones(len(dilations)))

    def forward(self, x):
        outputs = [m(x) for m in self.ms_convs]
        weights = F.softmax(self.alpha, dim=0)
        combined = sum(w * o for w, o in zip(weights, outputs))
        return self.ffn(self.norm(combined)) + x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, stride_size, padding=0):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=stride_size, padding=padding)

    def forward(self, x):
        return self.proj(x)


# ------------------ Main Model ------------------
class main_model(nn.Module):
    def __init__(self, params):
        super().__init__()
        input_shape = (params['input_channels'], 94, params['nb_mels'])
        cnn_filters = params['nb_conv_filters']
        dropout_rate = params['dropout']
        pool_sizes = params['f_pool_size']
        patch_size = params['patch_size']
        stride_size = params['stride_size']
        embed_dim = params['embed_dim']
        dilation_sets = params['tcn_dilation_sets']
        fc_dims = params.get('fc_dims', [256, 128])
        patch_padding = patch_size // 2
        pooling_length = params.get('pooling_length', 50)

        self.cnn_layers = nn.Sequential()
        in_channels = input_shape[0]
        for i, pool in enumerate(pool_sizes):
            self.cnn_layers.add_module(f'conv_{i}', nn.Conv2d(in_channels, cnn_filters, 3, padding=1))
            self.cnn_layers.add_module(f'bn_{i}', nn.BatchNorm2d(cnn_filters))
            self.cnn_layers.add_module(f'relu_{i}', nn.ReLU())
            self.cnn_layers.add_module(f'pool_{i}', nn.MaxPool2d(kernel_size=(1, pool)))
            self.cnn_layers.add_module(f'drop_{i}', nn.Dropout2d(dropout_rate))
            in_channels = cnn_filters

        freq_reduced = int(input_shape[2] // np.prod(pool_sizes))
        cnn_output_dim = cnn_filters * freq_reduced

        self.patch_embed = PatchEmbedding(
            cnn_output_dim, embed_dim, patch_size, stride_size, patch_padding
        )
        self.patch_norm = nn.BatchNorm1d(embed_dim)

        self.tcn_blocks = nn.Sequential(*[
            HybridTCNBlock(embed_dim, embed_dim * 4, dset) for dset in dilation_sets
        ])

        self.pos_enc = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, 8, 512, 0.1, batch_first=True)
        self.temporal_tf = nn.TransformerEncoder(encoder_layer, num_layers=2)

        fc_layers = []
        in_dim = embed_dim
        for d in fc_dims:
            fc_layers += [nn.Linear(in_dim, d), nn.ReLU()]
            in_dim = d
        fc_layers.append(nn.Linear(in_dim, 2))  # ← 바뀐 부분: [is_left, is_right]
        self.output_linear = nn.Sequential(*fc_layers)



    def forward(self, x):
        x = self.cnn_layers(x)                   # [B, C, T, F]
        x = x.permute(0, 2, 1, 3)                # [B, T, C, F]
        B, T, C, freq = x.shape
        x = x.reshape(B, T, C * freq).permute(0, 2, 1)  # [B, CF, T]

        x = self.patch_embed(x)                  # [B, embed, T′]
        x = self.patch_norm(x)
        x = x.permute(0, 2, 1)                   # [B, T′, embed]
        x = self.pos_enc(x)
        x = self.temporal_tf(x)
        x = x.permute(0, 2, 1)                   # [B, embed, T′]
        x = self.tcn_blocks(x)

        x = x.mean(dim=-1)                       # ← 시간축 평균: [B, embed]
        x = self.output_linear(x)                # [B, 2]
        return torch.sigmoid(x)                  # [B, 2] → [is_left_prob, is_right_prob]


# ------------------ 실행 예시 ------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = main_model(params).to(device)
    model.eval()

    B, C, T, freq_bins = 2, 2, 94, 64
    dummy_input = torch.randn(B, C, T, freq_bins).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print("Output shape:", output.shape)  # torch.Size([2, 2])
    print("첫 번째 샘플 결과 (is_left, is_right):", output[0])