import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torch.utils.checkpoint import checkpoint
from model.block.encoder_stage import Conv3DBlock, ResBlock3D

class AttentionBlock3D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W, D = x.shape
        residual = x
        x = self.norm(x)
        # Channel-wise scaling for memory efficiency
        qkv = self.qkv(x).view(B, 3 * self.num_heads, C // self.num_heads, H, W, D)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Compute attention with reduced spatial resolution for memory
        spatial_size = H * W * D
        if spatial_size > 8192:
            # Use strided attention for large spatial sizes
            stride = 2
            q_attn = q[:, :, :, ::stride, ::stride, ::stride].reshape(B, -1, (H + stride - 1) // stride * (W + stride - 1) // stride * (D + stride - 1) // stride)
            k_attn = k[:, :, :, ::stride, ::stride, ::stride].reshape(B, -1, (H + stride - 1) // stride * (W + stride - 1) // stride * (D + stride - 1) // stride)
            v_attn = v[:, :, :, ::stride, ::stride, ::stride].reshape(B, -1, (H + stride - 1) // stride * (W + stride - 1) // stride * (D + stride - 1) // stride)
        else:
            q_attn = q.reshape(B, -1, spatial_size)
            k_attn = k.reshape(B, -1, spatial_size)
            v_attn = v.reshape(B, -1, spatial_size)
        
        attn = (q_attn @ k_attn.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v_attn).reshape(B, -1, C // self.num_heads).view(B, C, H, W, D)
        out = self.proj(out)
        return out + residual


class CrossAttention3D(nn.Module):
    def __init__(self, dim: int, heads: int = None, num_heads: int = None):
        super().__init__()
        # accept either 'heads' or 'num_heads' keyword for flexibility
        h = heads if heads is not None else (num_heads if num_heads is not None else 4)
        self.dim = dim
        self.heads = h
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=h, batch_first=True)

    def forward(self, x, cond):
        # x: [B, C, H, W, D], cond: [B, C, H, W, D]
        B, C, H, W, D = x.shape
        seq_x = x.view(B, C, -1).permute(0, 2, 1)    # [B, Sx, C]
        seq_c = cond.view(B, C, -1).permute(0, 2, 1) # [B, Sc, C]
        out, _ = self.mha(seq_x, seq_c, seq_c)
        out = out.permute(0, 2, 1).view(B, C, H, W, D)
        return out
    
class UNet3D(nn.Module):
    def __init__(self, in_ch: int, base_ch: int, out_ch: int, use_gradient_checkpoint: bool = True):
        super().__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint
        self.stem = nn.Sequential(
            Conv3DBlock(in_ch, base_ch),
            ResBlock3D(base_ch)
        )
        self.down1 = self._down(base_ch, base_ch * 2)
        self.down2 = self._down(base_ch * 2, base_ch * 4)
        self.down3 = self._down(base_ch * 4, base_ch * 8)
        # mid block now supports cross-attention (cond will be passed separately)
        self.mid_res1 = ResBlock3D(base_ch * 8)
        self.mid_cross = CrossAttention3D(base_ch * 8, num_heads=4)
        self.mid_attn = AttentionBlock3D(base_ch * 8)
        self.mid_res2 = ResBlock3D(base_ch * 8)
        self.up3 = self._up(base_ch * 8 + base_ch * 8, base_ch * 4)
        self.up2 = self._up(base_ch * 4 + base_ch * 4, base_ch * 2)
        self.up1 = self._up(base_ch * 2 + base_ch * 2, base_ch)
        # add attention at high-res decoder
        self.attn_up1 = AttentionBlock3D(base_ch)
        # project conditioning to match mid/up channel dims for cross-attention
        # use UNet input channels (in_ch) as conditioning channel size
        self.cond_proj_mid = nn.Conv3d(in_channels=in_ch, out_channels=base_ch * 8, kernel_size=1)
        self.cond_proj_up2 = nn.Conv3d(in_channels=in_ch, out_channels=base_ch * 2, kernel_size=1)
        # multi-scale cross-attention at decoder mid-high resolution
        self.up2_cross = CrossAttention3D(base_ch * 2, num_heads=4)
        # additional multi-scale cross-attention as requested
        self.up3_cross = CrossAttention3D(base_ch * 4, num_heads=4)
        self.down2_cross = CrossAttention3D(base_ch * 4, num_heads=4)
        # conditioning projections for these scales
        self.cond_proj_down2 = nn.Conv3d(in_channels=in_ch, out_channels=base_ch * 4, kernel_size=1)
        self.cond_proj_up3 = nn.Conv3d(in_channels=in_ch, out_channels=base_ch * 4, kernel_size=1)
        self.out = nn.Conv3d(base_ch, out_ch, 3, 1, 1)

    def _down(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, 2, 1),
            ResBlock3D(out_ch),
            ResBlock3D(out_ch)
        )

    def _up(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, 4, 2, 1),
            ResBlock3D(out_ch),
            ResBlock3D(out_ch)
        )

    def forward(self, x, cond: Optional[torch.Tensor] = None):
        x = self.stem(x)
        if self.use_gradient_checkpoint and self.training:
            s1 = checkpoint(self.down1, x, use_reentrant=False)
            s2 = checkpoint(self.down2, s1, use_reentrant=False)
            # apply down2-level cross-attention conditioning
            if cond is not None:
                cond_down2 = cond
                if cond_down2.shape[2:] != s2.shape[2:]:
                    cond_down2 = F.interpolate(cond_down2, size=s2.shape[2:], mode='trilinear', align_corners=False)
                cond_down2 = self.cond_proj_down2(cond_down2)
                s2 = checkpoint(lambda a, cm=cond_down2: a + self.down2_cross(a, cm), s2, use_reentrant=False)
            s3 = checkpoint(self.down3, s2, use_reentrant=False)
            x = s3
            x = checkpoint(self.mid_res1, x, use_reentrant=False)
            if cond is not None:
                # prepare projected conditioning to match mid channels
                cond_mid = cond
                if cond_mid.shape[2:] != x.shape[2:]:
                    cond_mid = F.interpolate(cond_mid, size=x.shape[2:], mode='trilinear', align_corners=False)
                cond_mid = self.cond_proj_mid(cond_mid)
                x = checkpoint(lambda a, cm=cond_mid: a + self.mid_cross(a, cm), x, use_reentrant=False)
            x = checkpoint(self.mid_attn, x, use_reentrant=False)
            x = checkpoint(self.mid_res2, x, use_reentrant=False)
        else:
            s1 = self.down1(x)
            s2 = self.down2(s1)
            # apply down2-level cross-attention conditioning
            if cond is not None:
                cond_down2 = cond
                if cond_down2.shape[2:] != s2.shape[2:]:
                    cond_down2 = F.interpolate(cond_down2, size=s2.shape[2:], mode='trilinear', align_corners=False)
                cond_down2 = self.cond_proj_down2(cond_down2)
                s2 = s2 + self.down2_cross(s2, cond_down2)
            s3 = self.down3(s2)
            x = self.mid_res1(s3)
            if cond is not None:
                # resize cond to mid spatial if necessary
                if cond.shape[2:] != x.shape[2:]:
                    cond_mid = F.interpolate(cond, size=x.shape[2:], mode='trilinear', align_corners=False)
                else:
                    cond_mid = cond
                # project conditioning channels to match mid
                cond_mid = self.cond_proj_mid(cond_mid)
                x = x + self.mid_cross(x, cond_mid)
            x = self.mid_attn(x)
            x = self.mid_res2(x)
        # Ensure spatial dims match before concatenation (robust to odd sizes)
        def match_and_cat(a, b):
            # a is the tensor that will be passed through up block (smaller spatial),
            # b is the skip connection; we resize b to match a's spatial dims if needed
            if a.shape[2:] != b.shape[2:]:
                b = F.interpolate(b, size=a.shape[2:], mode='trilinear', align_corners=False)
            return torch.cat([a, b], dim=1)

        x = self.up3(match_and_cat(x, s3))
        # apply up3-level cross-attention conditioning
        if cond is not None:
            if cond.shape[2:] != x.shape[2:]:
                cond_up3 = F.interpolate(cond, size=x.shape[2:], mode='trilinear', align_corners=False)
            else:
                cond_up3 = cond
            cond_up3 = self.cond_proj_up3(cond_up3)
            x = x + self.up3_cross(x, cond_up3)
        x = self.up2(match_and_cat(x, s2))
        # apply multi-scale cross-attention at this decoder level if cond provided
        if cond is not None:
            # match cond spatial to current x
            if cond.shape[2:] != x.shape[2:]:
                cond_up2 = F.interpolate(cond, size=x.shape[2:], mode='trilinear', align_corners=False)
            else:
                cond_up2 = cond
            # project conditioning channels to match decoder channels
            cond_up2 = self.cond_proj_up2(cond_up2)
            x = x + self.up2_cross(x, cond_up2)
        x = self.up1(match_and_cat(x, s1))
        # lightweight attention at highest decoder resolution
        x = x + self.attn_up1(x)
        # return both output and a mid-level feature for deep supervision
        mid_feat = x  # high-resolution decoder feature (before final conv)
        return self.out(x), mid_feat


class DiffusionUNet(nn.Module):
    """
    Diffusion UNet nhận:
    - z_t: latent bị nhiễu [B, C, h, w, d]
    - z_cond: latent điều kiện [B, C, h, w, d]
    - t: timestep [B]
    - target_modality_id: ID của modality target [B]
    - modality_mask: mask M [B, 4]
    - domain_id: domain của sample [B]
    Output: ε_hat (noise prediction) [B, C, h, w, d]
    Supports sinusoidal time embedding and classifier-free guidance (unconditional training flag).
    """
    def __init__(self, latent_dim: int = 256, time_dim: int = 256, num_modalities: int = 4, num_domains: int = 3, unet: Optional[nn.Module] = None):
        super().__init__()
        # ---- Embeddings ----
        # use sinusoidal time embeddings -> then MLP
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.modality_embed = nn.Embedding(num_modalities, latent_dim)
        self.domain_embed = nn.Embedding(num_domains, latent_dim)
        self.mask_proj = nn.Linear(num_modalities, latent_dim)
        # ---- Projections ----
        self.input_proj = Conv3DBlock(latent_dim * 2, latent_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(time_dim + latent_dim * 3, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        # ---- UNet backbone ----
        self.unet = unet if unet is not None else UNet3D(
            in_ch=latent_dim,
            base_ch=64,
            out_ch=latent_dim
        )
        # store total timesteps for normalized time embedding
        self.num_timesteps = None

    def forward(self, z_t, z_cond, t, target_modality_id, modality_mask, domain_id, unconditional: bool = False):
        B = z_t.shape[0]
        # ---- Condition embeddings ----
        # normalize timestep for embedding to keep scale compatible with latents
        if self.num_timesteps is not None:
            t_for_emb = (t.float() / float(self.num_timesteps))
        else:
            t_for_emb = t.float()
        t_emb = get_sinusoidal_embedding(t_for_emb, self.time_dim).to(z_t.device)
        t_emb = self.time_mlp(t_emb)
        mod_emb = self.modality_embed(target_modality_id)
        dom_emb = self.domain_embed(domain_id)
        mask_emb = self.mask_proj(modality_mask.float())
        if unconditional:
            mod_emb = torch.zeros_like(mod_emb)
            dom_emb = torch.zeros_like(dom_emb)
            mask_emb = torch.zeros_like(mask_emb)
        cond = torch.cat([t_emb, mod_emb, dom_emb, mask_emb], dim=-1)
        cond = self.cond_proj(cond)
        # ---- Input ----
        # ---- Input ----
        x = torch.cat([z_t, z_cond], dim=1)
        x = self.input_proj(x)
        # Classifier-free guidance: zero conditioning spatial when unconditional
        if unconditional:
            z_cond_input = torch.zeros_like(z_cond)
        else:
            z_cond_input = z_cond
        # ---- UNet (use cross-attention only; do not add cond_spatial)
        # UNet now can return (epsilon_hat, mid_feat)
        unet_out = self.unet(x, cond=z_cond_input)
        if isinstance(unet_out, tuple) or isinstance(unet_out, list):
            epsilon_hat, mid_feat = unet_out
        else:
            epsilon_hat = unet_out
            mid_feat = None
        # ensure output spatial shape matches input z_t (robustness)
        if epsilon_hat.shape[2:] != z_t.shape[2:]:
            epsilon_hat = F.interpolate(epsilon_hat, size=z_t.shape[2:], mode='trilinear', align_corners=False)
        if mid_feat is not None:
            # allow caller to request mid feature by passing return_mid=True
            return epsilon_hat, mid_feat
        return epsilon_hat


def get_sinusoidal_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    """Sinusoidal positional embedding for timesteps.
    timesteps: (B,) long tensor
    returns: (B, dim) float tensor
    """
    assert len(timesteps.shape) == 1
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, dtype=torch.float32)) * torch.arange(0, half, dtype=torch.float32) / half
    )
    freqs = freqs.to(timesteps.device)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=emb.device)], dim=1)
    return emb