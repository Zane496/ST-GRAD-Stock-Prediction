import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveAdjacency(nn.Module):
    """Adaptive Adjacency Matrix Generator."""
    def __init__(self, num_nodes, embed_dim=10):
        super().__init__()
        self.E1 = nn.Parameter(torch.empty(num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.empty(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.E1)
        nn.init.xavier_uniform_(self.E2)
        self.eps = 1e-6

    def forward(self):
        adj = torch.relu(torch.mm(self.E1, self.E2.t())) + self.eps
        return F.softmax(adj, dim=1)

class UDConv(nn.Module):
    """U-D Convolution Module."""
    def __init__(self, in_channels, latent_channels):
        super().__init__()
        self.conv_up = nn.Conv2d(in_channels, latent_channels, kernel_size=1)
        self.conv_down = nn.Conv2d(latent_channels, 1, kernel_size=1)

    def forward(self, x):
        x_up = F.relu(self.conv_up(x))
        return F.relu(self.conv_down(x_up))

class CrossAttentionView(nn.Module):
    """Cross Attention View Module."""
    def __init__(self, num_cav, node, step=0, latent_channels=32):
        super().__init__()
        self.step = step
        self.node = node
        self.latent_channels = latent_channels
        self.num_cav = num_cav
        self.p = int(self.step / 2)

        in_ch = 3 * self.p if self.num_cav == 1 else 3
        self.ud_conv = UDConv(in_ch, latent_channels)

    def readout(self, x):
        if self.num_cav in [0, 2]:
            max_val, _ = torch.max(x, dim=1, keepdim=True)
            min_val, _ = torch.min(x, dim=1, keepdim=True)
            mean_val = torch.mean(x, dim=1, keepdim=True)
            return torch.cat([max_val, min_val, mean_val], dim=1)

        if self.num_cav == 1:
            if self.step <= 2:
                max_val, _ = torch.max(x, dim=1, keepdim=True)
                min_val, _ = torch.min(x, dim=1, keepdim=True)
                mean_val = torch.mean(x, dim=1, keepdim=True)
                return torch.cat([max_val, min_val, mean_val], dim=1)
            else:
                x_split = x.split(int(self.step / self.p), dim=1)
                x_rt = []
                for x_i in x_split:
                    max_v, _ = torch.max(x_i, dim=1, keepdim=True)
                    min_v, _ = torch.min(x_i, dim=1, keepdim=True)
                    mean_v = torch.mean(x_i, dim=1, keepdim=True)
                    x_rt.append(torch.cat([max_v, min_v, mean_v], dim=1))
                return torch.cat(x_rt, dim=1)
        return None

    def forward(self, x):
        x_read = self.readout(x)
        attention = self.ud_conv(x_read)
        device = x.device

        if self.num_cav == 0:
            x_conv = F.conv2d(x.permute(0, 1, 3, 2), torch.ones(1, self.latent_channels, 1, 1).to(device)).permute(0, 1, 3, 2)
            return torch.tanh(x_conv) * torch.sigmoid(attention)
        if self.num_cav == 1:
            x_conv = F.conv2d(x, torch.ones(1, self.step, 1, 1).to(device))
            return torch.tanh(x_conv) * torch.sigmoid(attention)
        if self.num_cav == 2:
            x_conv = F.conv2d(x, torch.ones(1, self.node, 1, 1).to(device))
            return torch.tanh(x_conv) * torch.sigmoid(attention)
        return None

class TemporalAttention(nn.Module):
    """Temporal Attention Mechanism."""
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        batch_size, C, T, N = x.shape
        x_reshaped = x.permute(0, 3, 2, 1).reshape(-1, T, C)
        attn_output, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        x_out = self.norm(x_reshaped + attn_output)
        return x_out.reshape(batch_size, N, T, C).permute(0, 3, 2, 1)

class DiffusionConvolution(nn.Module):
    """Diffusion Convolution Layer."""
    def __init__(self, num_nodes, feature_dim, max_diffusion_step=2, embed_dim=10):
        super().__init__()
        self.num_nodes = num_nodes
        self.max_diffusion_step = max_diffusion_step
        self.adaptive_adj = AdaptiveAdjacency(num_nodes, embed_dim)

        self.gcn_weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(feature_dim, feature_dim))
            for _ in range(max_diffusion_step + 1)
        ])
        for weight in self.gcn_weights:
            nn.init.xavier_uniform_(weight)

    def _build_matrix_powers(self, A):
        powers = [torch.eye(self.num_nodes, device=A.device)]
        for _ in range(1, self.max_diffusion_step + 1):
            powers.append(torch.matmul(powers[-1], A))
        return powers

    def forward(self, x_cross, A):
        batch_size, C, T, N = x_cross.shape
        A_powers = self._build_matrix_powers(A)
        A_T_powers = self._build_matrix_powers(A.t())
        A_adp = self.adaptive_adj()

        output = torch.zeros_like(x_cross)
        x_reshaped = x_cross.permute(0, 2, 3, 1).reshape(-1, N, C)

        for k in range(self.max_diffusion_step):
            diffused = torch.einsum('ij,bjc->bic', A_powers[k], x_reshaped)
            weighted = diffused @ self.gcn_weights[k]
            output += weighted.reshape(batch_size, T, N, C).permute(0, 3, 1, 2)

            diffused = torch.einsum('ij,bjc->bic', A_T_powers[k], x_reshaped)
            weighted = diffused @ self.gcn_weights[k]
            output += weighted.reshape(batch_size, T, N, C).permute(0, 3, 1, 2)

            if k == 0:
                diffused = torch.einsum('ij,bjc->bic', A_adp, x_reshaped)
                weighted = diffused @ self.gcn_weights[k]
                output += weighted.reshape(batch_size, T, N, C).permute(0, 3, 1, 2)

        return output

class STBlock(nn.Module):
    """Spatio-Temporal Block."""
    def __init__(self, num_nodes, feature_dim, seq_len, diffusion_steps=2, embed_dim=10):
        super().__init__()
        self.st_view = CrossAttentionView(num_cav=0, step=seq_len, node=num_nodes, latent_channels=feature_dim)
        self.fs_view = CrossAttentionView(num_cav=1, step=seq_len, node=num_nodes, latent_channels=feature_dim)
        self.ft_view = CrossAttentionView(num_cav=2, step=seq_len, node=num_nodes, latent_channels=feature_dim)

        self.alpha_st = nn.Parameter(torch.ones(1))
        self.alpha_fs = nn.Parameter(torch.ones(1))
        self.alpha_ft = nn.Parameter(torch.ones(1))

        self.diffusion_conv = DiffusionConvolution(num_nodes, feature_dim, diffusion_steps, embed_dim)
        self.residual_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.norm = nn.LayerNorm([feature_dim, seq_len, num_nodes])
        self.temporal_attention = TemporalAttention(feature_dim)

    def forward(self, x, A):
        x_temp = self.temporal_attention(x)
        x = x + x_temp

        x_orig = x
        x_st = self.st_view(x)
        x_fs = self.fs_view(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_ft = self.ft_view(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        weights = torch.softmax(torch.stack([self.alpha_st, self.alpha_fs, self.alpha_ft]), dim=0)
        x_cross = weights[0] * x_st + weights[1] * x_fs + weights[2] * x_ft

        x_gcn = self.diffusion_conv(x_cross, A)
        x_res = x_orig + self.residual_conv(x_gcn)
        return F.relu(self.norm(x_res))

class ST_GRAD(nn.Module):
    """Main ST-GRAD Architecture."""
    def __init__(self, in_channels, out_step, num_nodes, seq_len,
                 hidden_channels=32, num_layers=2, diffusion_steps=2, embed_dim=10):
        super().__init__()
        self.num_nodes = num_nodes
        self.norm = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout(0.3)
        self.init_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        self.st_blocks = nn.ModuleList([
            STBlock(feature_dim=hidden_channels, num_nodes=num_nodes, diffusion_steps=diffusion_steps, embed_dim=embed_dim, seq_len=seq_len)
            for _ in range(num_layers)
        ])

        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(hidden_channels, out_step, kernel_size=1)
        )

    def forward(self, x, A):
        x = self.dropout(self.norm(x))
        x = self.init_conv(x)

        layer_outputs = []
        for block in self.st_blocks:
            x = block(x, A)
            layer_outputs.append(x)

        x_out = x + sum(layer_outputs)
        output = self.output_conv(x_out)
        return output[:, :, -1, :].squeeze(1)