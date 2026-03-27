import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from model.block.decoder_stage import ModalityDecoder
from model.block.diffusion_stage import DiffusionUNet, UNet3D
from model.block.encoder_stage import ModalityEncoder, SharedProjection
import math


MODALITIES = ["t1", "t1ce", "t2", "flair"]


class DiffusionSynthesisModel(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        num_timesteps: int = 1000,
        num_modalities: int = 4,
        num_domains: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps
        self.num_modalities = num_modalities

        self.encoder_T1 = ModalityEncoder(latent_dim)
        self.encoder_T1ce = ModalityEncoder(latent_dim)
        self.encoder_T2 = ModalityEncoder(latent_dim)
        self.encoder_Flair = ModalityEncoder(latent_dim)
        self.projection = SharedProjection(latent_dim)

        self.modality_embed = nn.Embedding(4, latent_dim)

        self.unet_backbone = UNet3D(in_ch=latent_dim, base_ch=96, out_ch=latent_dim)
        self.diffusion_unet = DiffusionUNet(
            latent_dim=latent_dim,
            num_modalities=num_modalities,
            num_domains=num_domains,
            unet=self.unet_backbone,
        )
        self.diffusion_unet.num_timesteps = self.num_timesteps

        self.deep_proj = nn.Conv3d(96, latent_dim, kernel_size=1)

        self.decoder_T1 = ModalityDecoder(latent_dim)
        self.decoder_T1ce = ModalityDecoder(latent_dim)
        self.decoder_T2 = ModalityDecoder(latent_dim)
        self.decoder_Flair = ModalityDecoder(latent_dim)

        def cosine_beta_schedule(timesteps, s=0.008):
            x = torch.linspace(0, timesteps, timesteps + 1) / timesteps
            alphas_cumprod = torch.cos(((x + s) / (1 + s)) * (math.pi / 2)) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return betas.clamp(max=0.999)

        betas = cosine_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer(
            "alphas_cumprod_prev",
            torch.cat([betas.new_ones(1), alphas_cumprod[:-1]]),
        )
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

        self.modality_mha = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=min(4, max(1, latent_dim // 16)),
            batch_first=True,
        )
        self.fuse_proj = nn.Conv3d(latent_dim, latent_dim, kernel_size=1)

        self.use_v_pred = True

    def get_encoder(self, name: str) -> nn.Module:
        return {
            "t1": self.encoder_T1,
            "t1ce": self.encoder_T1ce,
            "t2": self.encoder_T2,
            "flair": self.encoder_Flair,
        }[name]

    def get_decoder(self, mod_id: int) -> nn.Module:
        return [
            self.decoder_T1,
            self.decoder_T1ce,
            self.decoder_T2,
            self.decoder_Flair,
        ][mod_id]

    def encode_modalities(self, x_dict: Dict[str, torch.Tensor], mask: torch.Tensor) -> List[torch.Tensor]:
        modality_names = ["t1", "t1ce", "t2", "flair"]
        B = list(x_dict.values())[0].shape[0]
        z_list = []
        ref_shape = None

        for i, mod_name in enumerate(modality_names):
            x = x_dict[mod_name]
            encoder = self.get_encoder(mod_name)
            avail = (mask[:, i] == 1)

            if avail.any():
                z_sel = self.projection(encoder(x[avail]))
                emb = self.modality_embed(
                    torch.full((z_sel.shape[0],), i, device=z_sel.device, dtype=torch.long)
                ).view(z_sel.shape[0], -1, 1, 1, 1)
                z_sel = z_sel + emb

                z_full = torch.zeros(B, *z_sel.shape[1:], device=z_sel.device, dtype=z_sel.dtype)
                z_full[avail] = z_sel
                ref_shape = z_sel.shape[1:]
            else:
                if ref_shape is None:
                    with torch.no_grad():
                        z_tmp = self.projection(encoder(x[:1]))
                    ref_shape = z_tmp.shape[1:]
                z_full = torch.zeros(B, *ref_shape, device=x.device, dtype=x.dtype)

            z_list.append(z_full)

        return z_list

    def encode_target(
        self,
        x_target: torch.Tensor,
        target_modality_id: torch.Tensor,
        target_modality_name: Optional[str] = None,
    ) -> torch.Tensor:
        results = []
        for i in range(x_target.shape[0]):
            mod_id = int(target_modality_id[i].item())
            mod_name = MODALITIES[mod_id]
            z = self.projection(self.get_encoder(mod_name)(x_target[i:i+1]))
            results.append(z)
        return torch.cat(results, dim=0)

    def fuse_latents(self, z_list: List[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        B = z_list[0].shape[0]
        masked = [
            z * mask[:, i].view(B, 1, 1, 1, 1).to(z.device)
            for i, z in enumerate(z_list)
        ]
        z_stack = torch.stack(masked, dim=1)  # [B, M, C, H, W, D]
        B, M, C, H, W, D = z_stack.shape

        z_perm = (
            z_stack.permute(0, 3, 4, 5, 1, 2)
            .contiguous()
            .view(B * H * W * D, M, C)
        )
        attn_out, _ = self.modality_mha(z_perm, z_perm, z_perm)
        fused = (
            attn_out.mean(dim=1)
            .view(B, H, W, D, C)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )
        return self.fuse_proj(fused)

    def forward_diffusion(self, z_target: torch.Tensor, t: torch.Tensor):
        epsilon = torch.randn_like(z_target)
        a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        z_t = a * z_target + b * epsilon
        return z_t, epsilon

    def decode(self, z_0: torch.Tensor, target_modality_id: torch.Tensor) -> torch.Tensor:
        results = []
        for i in range(z_0.shape[0]):
            m = int(target_modality_id[i].item())
            results.append(self.get_decoder(m)(z_0[i:i+1]))
        return torch.cat(results, dim=0)

    def denoise_step(self, z_t: torch.Tensor, epsilon_hat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        beta_t = self.betas[t].view(-1, 1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        alpha_cumprod_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1, 1)

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - alpha_cumprod_t)
        posterior_mean = coef1 * (z_t - coef2 * epsilon_hat)
        posterior_var = (
            beta_t * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)
        ).clamp(min=1e-20)

        noise = torch.randn_like(z_t)
        nonzero = (t > 0).view(-1, 1, 1, 1, 1).float()
        return posterior_mean + nonzero * torch.sqrt(posterior_var) * noise

    def forward(
        self,
        x_dict,
        x_target_gt,
        target_modality_name,
        target_modality_id,
        modality_mask,
        domain_id,
        t=None,
        unconditional=False,
        num_infer_steps=None,
    ):
        B = list(x_dict.values())[0].shape[0]
        device = list(x_dict.values())[0].device

        z_available = self.encode_modalities(x_dict, modality_mask)
        z_cond = self.fuse_latents(z_available, modality_mask)

        if x_target_gt is not None:
            if t is None:
                t = torch.randint(0, self.num_timesteps, (B,), device=device)

            z_target = self.encode_target(
                x_target=x_target_gt,
                target_modality_id=target_modality_id,
                target_modality_name=target_modality_name,
            )

            z_t, epsilon = self.forward_diffusion(z_target, t)

            unet_out = self.diffusion_unet(
                z_t,
                z_cond,
                t,
                target_modality_id,
                modality_mask,
                domain_id,
                unconditional,
            )

            pred, mid_feat = unet_out if isinstance(unet_out, (tuple, list)) else (unet_out, None)

            a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
            b = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)

            if self.use_v_pred:
                v_hat = pred
                v_target = a * epsilon - b * z_target
                z_0_hat = (a * z_t - b * v_hat).clamp(-10, 10)
            else:
                v_hat = pred
                v_target = epsilon
                z_0_hat = ((z_t - b * v_hat) / (a + 1e-8)).clamp(-10, 10)

            x_hat = self.decode(z_0_hat, target_modality_id)

            out = {
                "v_hat": v_hat,
                "v_target": v_target,
                "z_0_hat": z_0_hat,
                "z_target": z_target,
                "x_hat": x_hat,
                "x_target": x_target_gt,
                "z_cond": z_cond,
                "t": t,
            }

            if mid_feat is not None:
                z_mid = self.deep_proj(mid_feat)
                if z_mid.shape[2:] != z_target.shape[2:]:
                    z_mid = F.interpolate(
                        z_mid,
                        size=z_target.shape[2:],
                        mode="trilinear",
                        align_corners=False,
                    )
                out["z_mid_pred"] = z_mid

            return out

        z_t = torch.randn_like(z_cond)
        num_steps = int(num_infer_steps) if num_infer_steps is not None else self.num_timesteps

        for step in reversed(range(num_steps)):
            t_step = torch.full((B,), step, device=device, dtype=torch.long)
            unet_out = self.diffusion_unet(
                z_t,
                z_cond,
                t_step,
                target_modality_id,
                modality_mask,
                domain_id,
                unconditional=False,
            )
            pred = unet_out[0] if isinstance(unet_out, (tuple, list)) else unet_out

            if self.use_v_pred:
                a = self.sqrt_alphas_cumprod[t_step].view(-1, 1, 1, 1, 1)
                b = self.sqrt_one_minus_alphas_cumprod[t_step].view(-1, 1, 1, 1, 1)
                epsilon_hat = b * z_t + a * pred
            else:
                epsilon_hat = pred

            z_t = self.denoise_step(z_t, epsilon_hat, t_step)

        return {
            "x_hat": self.decode(z_t, target_modality_id),
            "z_cond": z_cond,
        }

def create_model(
    latent_dim=128,
    num_timesteps=1000,
    num_modalities=4,
    num_domains=3,
) -> DiffusionSynthesisModel:
    return DiffusionSynthesisModel(
        latent_dim=latent_dim,
        num_timesteps=num_timesteps,
        num_modalities=num_modalities,
        num_domains=num_domains,
    )