import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torch.utils.checkpoint import checkpoint
from model.block.encoder_stage import Conv3DBlock, ResBlock3D

class AttentionBlock3D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels  = channels
        self.num_heads = num_heads
        self.norm      = nn.GroupNorm(min(8, channels), channels)
        self.qkv       = nn.Conv3d(channels, channels * 3, 1)
        self.proj      = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = x.shape
        residual = x
        x = self.norm(x)
        qkv     = self.qkv(x)                     # [B, 3C, H, W, D]
        spatial  = H * W * D
        head_dim = C // self.num_heads
        qkv = qkv.reshape(B, 3, self.num_heads, head_dim, spatial)
        q, k, v = qkv.unbind(dim=1)              # each [B, num_heads, head_dim, spatial]
        if spatial > 1024:
            k_3d = k.reshape(B * self.num_heads, head_dim, H, W, D)
            v_3d = v.reshape(B * self.num_heads, head_dim, H, W, D)
            k_3d = F.avg_pool3d(k_3d, kernel_size=2, stride=2)
            v_3d = F.avg_pool3d(v_3d, kernel_size=2, stride=2)
            k = k_3d.reshape(B, self.num_heads, head_dim, -1)
            v = v_3d.reshape(B, self.num_heads, head_dim, -1)
        scale = head_dim ** -0.5
        attn  = torch.einsum('bhdn,bhdm->bhnm', q * scale, k)  # [B, heads, sq, sk]
        attn  = F.softmax(attn, dim=-1)
        out   = torch.einsum('bhnm,bhdm->bhdn', attn, v)        # [B, heads, head_dim, sq]
        out   = out.reshape(B, C, H, W, D)
        return self.proj(out) + residual

class CrossAttention3D(nn.Module):
    def __init__(self, dim: int, heads: int = None, num_heads: int = None):
        super().__init__()
        h = heads if heads is not None else (num_heads if num_heads is not None else 4)
        self.dim   = dim
        self.heads = h
        self.mha   = nn.MultiheadAttention(embed_dim=dim, num_heads=h, batch_first=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = x.shape
        seq_x = x.view(B, C, -1).permute(0, 2, 1)     # [B, Sx, C]
        seq_c = cond.view(B, C, -1).permute(0, 2, 1)  # [B, Sc, C]
        out, _ = self.mha(seq_x, seq_c, seq_c)
        return out.permute(0, 2, 1).view(B, C, H, W, D)

class UNet3D(nn.Module):
    def __init__(self, in_ch: int, base_ch: int, out_ch: int, use_gradient_checkpoint: bool = True):
        super().__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint
        self.stem  = nn.Sequential(Conv3DBlock(in_ch, base_ch), ResBlock3D(base_ch))
        self.down1 = self._down(base_ch,     base_ch * 2)
        self.down2 = self._down(base_ch * 2, base_ch * 4)
        self.down3 = self._down(base_ch * 4, base_ch * 8)
        # Mid block
        self.mid_res1  = ResBlock3D(base_ch * 8)
        self.mid_cross = CrossAttention3D(base_ch * 8, num_heads=4)
        self.mid_attn  = AttentionBlock3D(base_ch * 8)   # spatial 2×2×2=8 → OK
        self.mid_res2  = ResBlock3D(base_ch * 8)
        # Decoder
        self.up3 = self._up(base_ch * 8 + base_ch * 8, base_ch * 4)
        self.up2 = self._up(base_ch * 4 + base_ch * 4, base_ch * 2)
        self.up1 = self._up(base_ch * 2 + base_ch * 2, base_ch)
        # Attention at highest decoder resolution (16×16×16=4096)
        self.attn_up1 = AttentionBlock3D(base_ch)
        # Conditioning projections per scale
        self.cond_proj_mid   = nn.Conv3d(in_ch, base_ch * 8, 1)
        self.cond_proj_up2   = nn.Conv3d(in_ch, base_ch * 2, 1)
        self.cond_proj_up3   = nn.Conv3d(in_ch, base_ch * 4, 1)
        self.cond_proj_down2 = nn.Conv3d(in_ch, base_ch * 4, 1)
        # Multi-scale cross-attention
        self.up2_cross   = CrossAttention3D(base_ch * 2, num_heads=4)
        self.up3_cross   = CrossAttention3D(base_ch * 4, num_heads=4)
        self.down2_cross = CrossAttention3D(base_ch * 4, num_heads=4)
        self.out = nn.Conv3d(base_ch, out_ch, 3, 1, 1)

    def _down(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, 2, 1),
            ResBlock3D(out_ch),
            ResBlock3D(out_ch),
        )

    def _up(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, 4, 2, 1),
            ResBlock3D(out_ch),
            ResBlock3D(out_ch),
        )

    @staticmethod
    def _match_and_cat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Resize skip connection b to match a's spatial dims, then cat."""
        if a.shape[2:] != b.shape[2:]:
            b = F.interpolate(b, size=a.shape[2:], mode='trilinear', align_corners=False)
        return torch.cat([a, b], dim=1)

    def _cond_resize(self, cond: torch.Tensor, target_shape) -> torch.Tensor:
        if cond.shape[2:] != target_shape:
            return F.interpolate(cond, size=target_shape, mode='trilinear', align_corners=False)
        return cond

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        # ---- Encoder ----
        if self.use_gradient_checkpoint and self.training:
            s1 = checkpoint(self.down1, x, use_reentrant=False)
            s2 = checkpoint(self.down2, s1, use_reentrant=False)
            if cond is not None:
                cd2 = self.cond_proj_down2(self._cond_resize(cond, s2.shape[2:]))
                # FIX: usar named function helper en lugar de lambda con closure
                s2 = checkpoint(self._apply_cross_down2, s2, cd2, use_reentrant=False)
            s3 = checkpoint(self.down3, s2, use_reentrant=False)
            x  = s3
            x = checkpoint(self.mid_res1, x, use_reentrant=False)
            if cond is not None:
                cm = self.cond_proj_mid(self._cond_resize(cond, x.shape[2:]))
                x  = checkpoint(self._apply_cross_mid, x, cm, use_reentrant=False)
            x = checkpoint(self.mid_attn, x, use_reentrant=False)
            x = checkpoint(self.mid_res2, x, use_reentrant=False)
        else:
            s1 = self.down1(x)
            s2 = self.down2(s1)
            if cond is not None:
                cd2 = self.cond_proj_down2(self._cond_resize(cond, s2.shape[2:]))
                s2  = s2 + self.down2_cross(s2, cd2)
            s3 = self.down3(s2)
            x  = self.mid_res1(s3)
            if cond is not None:
                cm = self.cond_proj_mid(self._cond_resize(cond, x.shape[2:]))
                x  = x + self.mid_cross(x, cm)
            x = self.mid_attn(x)
            x = self.mid_res2(x)
        # ---- Decoder ----
        x = self.up3(self._match_and_cat(x, s3))
        if cond is not None:
            cu3 = self.cond_proj_up3(self._cond_resize(cond, x.shape[2:]))
            x   = x + self.up3_cross(x, cu3)
        x = self.up2(self._match_and_cat(x, s2))
        if cond is not None:
            cu2 = self.cond_proj_up2(self._cond_resize(cond, x.shape[2:]))
            x   = x + self.up2_cross(x, cu2)
        x = self.up1(self._match_and_cat(x, s1))
        x = x + self.attn_up1(x)   # spatial 16³=4096 → k,v downsampled (threshold 1024) ✓
        mid_feat = x                # [B, base_ch, 16, 16, 16] for deep supervision
        return self.out(x), mid_feat
    # ---- Helper methods for gradient checkpoint (avoid lambda closures) ----
    def _apply_cross_down2(self, x: torch.Tensor, cm: torch.Tensor) -> torch.Tensor:
        return x + self.down2_cross(x, cm)
    def _apply_cross_mid(self, x: torch.Tensor, cm: torch.Tensor) -> torch.Tensor:
        return x + self.mid_cross(x, cm)

class DiffusionUNet(nn.Module):
    def __init__(self, latent_dim: int = 256, time_dim: int = 256,
                 num_modalities: int = 4, num_domains: int = 3,
                 unet: Optional[nn.Module] = None):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.modality_embed = nn.Embedding(num_modalities, latent_dim)
        self.domain_embed   = nn.Embedding(num_domains, latent_dim)
        self.mask_proj      = nn.Linear(num_modalities, latent_dim)
        self.input_proj = Conv3DBlock(latent_dim * 2, latent_dim)
        self.cond_proj  = nn.Sequential(
            nn.Linear(time_dim + latent_dim * 3, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.unet = unet if unet is not None else UNet3D(
            in_ch=latent_dim, base_ch=64, out_ch=latent_dim)
        self.num_timesteps = None

    def forward(self, z_t, z_cond, t, target_modality_id, modality_mask,
                domain_id, unconditional: bool = False):
        B = z_t.shape[0]
        t_for_emb = (t.float() / float(self.num_timesteps)
                     if self.num_timesteps is not None else t.float())
        t_emb    = get_sinusoidal_embedding(t_for_emb, self.time_dim).to(z_t.device)
        t_emb    = self.time_mlp(t_emb)
        mod_emb  = self.modality_embed(target_modality_id)
        dom_emb  = self.domain_embed(domain_id)
        mask_emb = self.mask_proj(modality_mask.float())
        if unconditional:
            mod_emb  = torch.zeros_like(mod_emb)
            dom_emb  = torch.zeros_like(dom_emb)
            mask_emb = torch.zeros_like(mask_emb)
        cond = torch.cat([t_emb, mod_emb, dom_emb, mask_emb], dim=-1)
        cond = self.cond_proj(cond)                    # [B, latent_dim]
        x = self.input_proj(torch.cat([z_t, z_cond], dim=1))
        x = x + cond.view(B, -1, 1, 1, 1)
        z_cond_input = torch.zeros_like(z_cond) if unconditional else z_cond
        unet_out = self.unet(x, cond=z_cond_input)
        if isinstance(unet_out, (tuple, list)):
            epsilon_hat, mid_feat = unet_out
        else:
            epsilon_hat, mid_feat = unet_out, None
        if epsilon_hat.shape[2:] != z_t.shape[2:]:
            epsilon_hat = F.interpolate(epsilon_hat, size=z_t.shape[2:],
                                        mode='trilinear', align_corners=False)
        return (epsilon_hat, mid_feat) if mid_feat is not None else epsilon_hat


def get_sinusoidal_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    assert len(timesteps.shape) == 1
    half  = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, dtype=torch.float32))
        * torch.arange(0, half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=emb.device)], dim=1)
    return emb