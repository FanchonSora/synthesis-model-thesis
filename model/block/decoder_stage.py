import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from model.block.encoder_stage import Conv3DBlock, ResBlock3D

class ModalityDecoder(nn.Module):
    """
    Decoder cho từng modality
    Input: [B, C, h, w, d] - latent
    Output: [B, 1, H, W, D] - reconstructed image
    """
    def __init__(self, latent_dim: int = 256, base_channels: int = 16):
        super().__init__()
        # Upsampling path
        self.up1 = nn.ConvTranspose3d(latent_dim, base_channels * 4, 4, 2, 1)
        self.dec1 = nn.Sequential(
            Conv3DBlock(base_channels * 4, base_channels * 4),
            ResBlock3D(base_channels * 4),
            ResBlock3D(base_channels * 4)
        )
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 4, 2, 1)
        self.dec2 = nn.Sequential(
            Conv3DBlock(base_channels * 2, base_channels * 2),
            ResBlock3D(base_channels * 2)
        )
        self.up3 = nn.ConvTranspose3d(base_channels * 2, base_channels, 4, 2, 1)
        self.dec3 = nn.Sequential(
            Conv3DBlock(base_channels, base_channels),
            ResBlock3D(base_channels)
        )
        # Final output
        self.out = nn.Conv3d(base_channels, 1, 3, 1, 1)
    
    def forward(self, z):
        # z: [B, latent_dim, h, w, d]
        x = self.up1(z)
        x = self.dec1(x)
        x = self.up2(x)
        x = self.dec2(x)
        x = self.up3(x)
        x = self.dec3(x)
        x = self.out(x)
        # Output: [B, 1, H, W, D]
        return x

