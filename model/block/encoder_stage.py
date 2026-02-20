import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResBlock3D(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.conv1 = Conv3DBlock(channels, channels, groups=groups)
        self.conv2 = nn.Conv3d(channels, channels, 3, 1, 1)
        self.norm = nn.GroupNorm(min(groups, channels), channels)
        self.act = nn.SiLU()
        # Squeeze-and-Excitation lightweight channel attention
        self.se_ratio = max(1, channels // 8)
        self.se_fc1 = nn.Linear(channels, channels // 8)
        self.se_fc2 = nn.Linear(channels // 8, channels)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm(self.conv2(x))
        # SE
        w = self.global_pool(x).view(x.size(0), -1)
        w = torch.relu(self.se_fc1(w))
        w = torch.sigmoid(self.se_fc2(w)).view(x.size(0), x.size(1), 1, 1, 1)
        x = x * w
        return self.act(x + residual)

class ModalityEncoder(nn.Module):
    def __init__(self, latent_dim: int = 256, base_channels: int = 16):
        super().__init__()
        # Downsampling path
        self.enc1 = nn.Sequential(
            Conv3DBlock(1, base_channels),
            ResBlock3D(base_channels)
        )
        self.down1 = nn.Conv3d(base_channels, base_channels * 2, 3, 2, 1)
        self.enc2 = nn.Sequential(
            Conv3DBlock(base_channels * 2, base_channels * 2),
            ResBlock3D(base_channels * 2)
        )
        self.down2 = nn.Conv3d(base_channels * 2, base_channels * 4, 3, 2, 1)
        self.enc3 = nn.Sequential(
            Conv3DBlock(base_channels * 4, base_channels * 4),
            ResBlock3D(base_channels * 4)
        )
        self.down3 = nn.Conv3d(base_channels * 4, latent_dim, 3, 2, 1)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock3D(latent_dim),
            ResBlock3D(latent_dim)
        )
    
    def forward(self, x):
        x = self.enc1(x)  # x: [B, 1, H, W, D]
        x = self.down1(x)
        x = self.enc2(x)
        x = self.down2(x)
        x = self.enc3(x)
        x = self.down3(x)
        x = self.bottleneck(x) # Output: [B, latent_dim, h, w, d]
        return x

class SharedProjection(nn.Module):
    """
    Projection head để đưa các modality về không gian chung
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            Conv3DBlock(latent_dim, latent_dim),
            Conv3DBlock(latent_dim, latent_dim),
            nn.Conv3d(latent_dim, latent_dim, 1)
        )
    
    def forward(self, x):
        return self.proj(x)
