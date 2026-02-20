import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from model.block.decoder_stage import ModalityDecoder
from model.block.diffusion_stage import DiffusionUNet, UNet3D
from model.block.encoder_stage import ModalityEncoder, SharedProjection

class DiffusionSynthesisModel(nn.Module):
    def __init__(self,  latent_dim: int = 128, num_timesteps: int = 1000, num_modalities: int = 4, num_domains: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps
        self.num_modalities = num_modalities
        # Encoders cho từng modality
        self.encoder_T1 = ModalityEncoder(latent_dim)
        self.encoder_T1ce = ModalityEncoder(latent_dim)
        self.encoder_T2 = ModalityEncoder(latent_dim)
        self.encoder_Flair = ModalityEncoder(latent_dim)
        # Shared projection head
        self.projection = SharedProjection(latent_dim)
        # Embedding cho modality available
        self.modality_embed = nn.Embedding(num_embeddings=4, embedding_dim=latent_dim)
        # Diffusion UNet
        # stronger UNet backbone: increased base channels
        self.unet_backbone = UNet3D(
            in_ch=latent_dim,
            base_ch=96,
            out_ch=latent_dim
        )
        self.diffusion_unet = DiffusionUNet(
            latent_dim=latent_dim,
            num_modalities=num_modalities,
            num_domains=num_domains,
            unet=self.unet_backbone
        )
        # Deep-supervision projection: map UNet mid features to latent_dim for MSE supervision
        base_ch = 96
        # mid-level feature from UNet has `base_ch` channels (high-res decoder feature)
        self.deep_proj = nn.Conv3d(base_ch, latent_dim, kernel_size=1)
        # provide total timestep count to UNet so it can normalize timestep embeddings
        try:
            self.diffusion_unet.num_timesteps = self.num_timesteps
        except Exception:
            pass
        # Decoders cho từng modality
        self.decoder_T1 = ModalityDecoder(latent_dim)
        self.decoder_T1ce = ModalityDecoder(latent_dim)
        self.decoder_T2 = ModalityDecoder(latent_dim)
        self.decoder_Flair = ModalityDecoder(latent_dim)
        # Diffusion schedule (cosine as in Nichol & Dhariwal)
        import math
        def cosine_beta_schedule(timesteps, s: float = 0.008):
            steps = timesteps
            x = torch.linspace(0, steps, steps + 1, device=torch.device('cpu')) / steps
            alphas_cumprod = torch.cos(((x + s) / (1 + s)) * (math.pi / 2)) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return betas.clamp(max=0.999)

        betas = cosine_beta_schedule(num_timesteps).to(next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device('cpu'))
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        # previous cumprod (alpha_{t-1}), used for posterior variance
        self.register_buffer('alphas_cumprod_prev', torch.cat([self.alphas.new_ones(1), self.alphas_cumprod[:-1]]))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        # Attention fusion: treat modalities as tokens and run self-attention across modalities per-spatial-location
        # modality-level MHA (embed dim = latent_dim)
        self.modality_mha = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=min(4, latent_dim // 16), batch_first=True)
        self.fuse_proj = nn.Conv3d(latent_dim, latent_dim, kernel_size=1)
        # velocity (v) prediction option (enable for more stable training)
        self.use_v_pred = True
    def get_encoder(self, modality_name: str):
        """Get encoder by modality name"""
        encoders = {
            't1': self.encoder_T1,
            't1ce': self.encoder_T1ce,
            't2': self.encoder_T2,
            'flair': self.encoder_Flair
        }
        return encoders[modality_name]
    
    def get_decoder(self, modality_id: int):
        """Get decoder by modality ID (0-3)"""
        decoders = [self.decoder_T1, self.decoder_T1ce, self.decoder_T2, self.decoder_Flair]
        return decoders[modality_id]
    
    def encode_modalities(self, x_dict: Dict[str, torch.Tensor], mask: torch.Tensor):
        """
        Encode available modalities (không encode target)
        Args:
            x_dict: Dictionary {'T1': tensor, 'T1ce': tensor, ...}
            mask: [B, 4] - 1 = available, 0 = missing/target
        Returns:
            z_available: List of latents [B, C, h, w, d]
        """
        modality_names = ['t1', 't1ce', 't2', 'flair']
        # Encode only available samples per modality to avoid producing "zero-image" latents
        B = list(x_dict.values())[0].shape[0]
        z_list = []
        example_z = None
        for i, mod_name in enumerate(modality_names):
            x = x_dict[mod_name]
            encoder = self.get_encoder(mod_name)
            # mask for samples where this modality is available
            avail = (mask[:, i] == 1)
            if avail.any():
                x_sel = x[avail]
                f_sel = encoder(x_sel)
                z_sel = self.projection(f_sel)
                # modality embedding only applied to available entries
                emb = self.modality_embed(
                    torch.full((z_sel.shape[0],), i, device=z_sel.device, dtype=torch.long)
                ).view(z_sel.shape[0], -1, 1, 1, 1)
                z_sel = z_sel + emb
                # allocate full-batch tensor and scatter
                z_full = z_sel.new_zeros((B, z_sel.shape[1], z_sel.shape[2], z_sel.shape[3], z_sel.shape[4]))
                z_full[avail] = z_sel
                example_z = z_sel if example_z is None else example_z
            else:
                # no available samples for this modality in batch -> create zero tensor with spatial dims inferred
                # infer dims from previously encoded modality if possible
                if example_z is not None:
                    C, H, W, D = example_z.shape[1:]
                    z_full = torch.zeros((B, C, H, W, D), device=next(self.parameters()).device)
                else:
                    # fallback: encode full batch (may be zeros) to get shapes then zero
                    f_tmp = encoder(x)
                    z_tmp = self.projection(f_tmp)
                    z_full = torch.zeros_like(z_tmp)
            z_list.append(z_full)
        return z_list
    
    def encode_target(self, x_target: torch.Tensor, target_modality_name: str):
        """
        Encode target modality (CHỈ để train)
        Args:
            x_target: [B, 1, H, W, D]
            target_modality_name: 'T1' | 'T1ce' | 'T2' | 'Flair'
        Returns:
            z_target: [B, C, h, w, d]
        """
        encoder = self.get_encoder(target_modality_name)
        f_target = encoder(x_target)
        z_target = self.projection(f_target)
        return z_target
    
    def fuse_latents(self, z_list: List[torch.Tensor], mask: torch.Tensor):
        """
        Fusion các latents available → z_cond
        """
        # Apply modality mask per-slot to avoid spurious signals from missing modalities
        # Stack modalities: [B, M, C, H, W, D]
        B = z_list[0].shape[0]
        device = z_list[0].device
        masked = []
        for i, z in enumerate(z_list):
            m = mask[:, i].view(B, 1, 1, 1, 1).to(z.device)
            masked.append(z * m)
        z_stack = torch.stack(masked, dim=1)  # [B, M, C, H, W, D]
        B, M, C, H, W, D = z_stack.shape
        # reshape to run attention across modalities at each spatial position
        # -> (B * H * W * D, M, C)
        z_permute = z_stack.permute(0, 3, 4, 5, 1, 2).contiguous().view(B * H * W * D, M, C)
        # run multi-head attention across modalities
        attn_out, _ = self.modality_mha(z_permute, z_permute, z_permute)
        # aggregate modalities (mean across tokens) to form fused feature per spatial location
        fused = attn_out.mean(dim=1)  # [B * H * W * D, C]
        fused = fused.view(B, H, W, D, C).permute(0, 4, 1, 2, 3).contiguous()  # [B, C, H, W, D]
        # final 1x1 projection
        z_cond = self.fuse_proj(fused)
        return z_cond
    
    def forward_diffusion(self, z_target: torch.Tensor, t: torch.Tensor):
        """
        Add noise vào z_target
        Args:
            z_target: [B, C, h, w, d] - clean latent
            t: [B] - timestep
        Returns:
            z_t: [B, C, h, w, d] - noisy latent
            epsilon: [B, C, h, w, d] - noise đã add
        """
        epsilon = torch.randn_like(z_target)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        z_t = sqrt_alpha * z_target + sqrt_one_minus_alpha * epsilon
        return z_t, epsilon
    
    def predict_noise(self, z_t, z_cond, t, target_modality_id, modality_mask, domain_id, unconditional: bool = False, z_target: torch.Tensor = None):
        """
        Predict noise từ UNet
        Supports classifier-free guidance via `unconditional` flag (when True, conditioning embeddings are zeroed inside the UNet).
        Returns:
            epsilon_hat: [B, C, h, w, d]
        """
        pred = self.diffusion_unet(z_t, z_cond, t, target_modality_id, modality_mask, domain_id, unconditional)
        # DiffusionUNet may return (pred, mid_feat); accept either form
        mid_feat = None
        if isinstance(pred, (tuple, list)):
            pred, mid_feat = pred
        # If using v-pred parameterization, model's raw output is v_hat; convert to epsilon_hat
        if self.use_v_pred:
            v_hat = pred
            # buffers
            sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
            # Two usage paths:
            # - Training / when z_target is available: epsilon_hat = (v + sqrt_one_minus_alpha * z_target) / sqrt_alpha
            # - Inference / when z_target is None: derive epsilon_hat from v_hat and current z_t using algebraic relations
            if z_target is None:
                # From: z_t = a * x0 + b * eps
                #       v = a * eps - b * x0
                # Solve -> eps = b * z_t + a * v
                epsilon_hat = sqrt_one_minus_alpha * z_t + sqrt_alpha * v_hat
                return epsilon_hat
            else:
                epsilon_hat = (v_hat + sqrt_one_minus_alpha * z_target) / (sqrt_alpha + 1e-8)
                return epsilon_hat
        return pred
    
    def denoise_step(self, z_t, epsilon_hat, t):
        """
        Denoise một bước để lấy z_0 (estimate)
        Args:
            z_t: [B, C, h, w, d]
            epsilon_hat: [B, C, h, w, d]
            t: [B]
        
        Returns:
            z_0_hat: [B, C, h, w, d]
        """
        # Index buffers by timestep tensor t
        beta_t = self.betas[t].view(-1, 1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        alpha_cumprod_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1, 1)

        # DDPM posterior mean: 1/sqrt(alpha_t) * (z_t - beta_t / sqrt(1 - alpha_cumprod_t) * eps)
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - alpha_cumprod_t)
        posterior_mean = coef1 * (z_t - coef2 * epsilon_hat)

        # posterior variance
        posterior_var = beta_t * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)
        posterior_var = posterior_var.clamp(min=1e-20)
        posterior_std = torch.sqrt(posterior_var)

        # For t == 0 we should not add noise
        noise = torch.randn_like(z_t)
        nonzero_mask = (t > 0).view(-1, 1, 1, 1, 1).float()

        z_prev = posterior_mean + nonzero_mask * posterior_std * noise
        return z_prev
    
    def decode(self, z_0: torch.Tensor, target_modality_id: torch.Tensor):
        """
        Decode latent z_0 về image space theo target modality
        """
        B = z_0.size(0)
        # probe output shape (no grad, AMP-safe)
        with torch.no_grad():
            sample_out = self.get_decoder(0)(z_0[:1])
        outputs = sample_out.new_zeros(
            (B,) + sample_out.shape[1:]
        )
        for m in range(self.num_modalities):
            idx = (target_modality_id == m)
            if idx.any():
                decoder = self.get_decoder(m)
                outputs[idx] = decoder(z_0[idx])
        return outputs

    def forward(self, x_dict: Dict[str, torch.Tensor], x_target_gt: torch.Tensor, target_modality_name: str, target_modality_id: torch.Tensor,
                modality_mask: torch.Tensor, domain_id: torch.Tensor, t: Optional[torch.Tensor] = None, unconditional: bool = False,
                num_infer_steps: Optional[int] = None):
        B = list(x_dict.values())[0].shape[0]
        device = list(x_dict.values())[0].device
        if t is None:
            t = torch.randint(0, self.num_timesteps, (B,), device=device)
        # 1. Encode available modalities
        z_available = self.encode_modalities(x_dict, modality_mask)
        # 2. Fuse latents → z_cond
        z_cond = self.fuse_latents(z_available, modality_mask)
        # Ensure z_cond spatial shape matches target latent's shape (robustness)
        if x_target_gt is not None:
            # x_target_gt will be encoded to z_target shortly; ensure spatial dims match exactly
            tmp_enc = self.get_encoder(target_modality_name)(x_target_gt)
            tmp_z = self.projection(tmp_enc)
            if z_cond.shape[2:] != tmp_z.shape[2:]:
                raise RuntimeError(f"Spatial shape mismatch between z_cond {z_cond.shape[2:]} and encoder output {tmp_z.shape[2:]}. Check encoder/projection consistency.")
        #  TRAIN MODE
        if x_target_gt is not None:
            z_target = self.encode_target(x_target_gt, target_modality_name)
            z_t, epsilon = self.forward_diffusion(z_target, t)
            # Call UNet directly to obtain mid-level features for deep supervision
            unet_out = self.diffusion_unet(z_t, z_cond, t, target_modality_id, modality_mask, domain_id, unconditional)
            mid_feat = None
            if isinstance(unet_out, (tuple, list)):
                pred, mid_feat = unet_out
            else:
                pred = unet_out

            # convert prediction depending on v-pred or eps-pred
            if self.use_v_pred:
                v_hat = pred
                sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
                sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
                epsilon_hat = (v_hat + sqrt_one_minus_alpha * z_target) / (sqrt_alpha + 1e-8)
            else:
                epsilon_hat = pred

            # If mid-level feature present, project and resize for deep supervision
            if mid_feat is not None:
                # mid_feat: [B, C_mid, h, w, d] -> project to latent_dim
                z_mid_proj = self.deep_proj(mid_feat)
                # resize to z_target spatial for loss comparison
                z_mid_resized = F.interpolate(z_mid_proj, size=z_target.shape[2:], mode='trilinear', align_corners=False)
            else:
                z_mid_resized = None
            # Estimate z0 (for optional reconstruction) without performing a reverse sampling step
            sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
            z_0_hat = (z_t - sqrt_one_minus_alpha_cumprod_t * epsilon_hat) / (sqrt_alpha_cumprod_t + 1e-8)
            x_hat = self.decode(z_0_hat, target_modality_id)
            out = {
                "epsilon_hat": epsilon_hat,
                "epsilon": epsilon,
                "z_0_hat": z_0_hat,
                "z_target": z_target,
                "x_hat": x_hat,
                "x_target": x_target_gt,
                "z_cond": z_cond,
            }
            if z_mid_resized is not None:
                out["z_mid_pred"] = z_mid_resized
            return out
        # INFERENCE / VISUALIZE MODE
        else:
            # start from pure noise
            z_t = torch.randn_like(z_cond)
            num_steps = int(num_infer_steps) if (num_infer_steps is not None) else self.num_timesteps
            # run shortened reverse chain for visualization/speed if requested
            for step in reversed(range(num_steps)):
                t_step = torch.full((B,), step, device=device, dtype=torch.long)
                epsilon_hat = self.predict_noise(
                    z_t, z_cond, t_step, target_modality_id, modality_mask, domain_id
                )
                z_t = self.denoise_step(z_t, epsilon_hat, t_step)
            x_hat = self.decode(z_t, target_modality_id)
            return {
                "x_hat": x_hat,
                "z_cond": z_cond,
            }

def create_model(latent_dim: int = 128, num_timesteps: int = 1000, num_modalities: int = 4, num_domains: int = 3):
    model = DiffusionSynthesisModel(
        latent_dim=latent_dim,
        num_timesteps=num_timesteps,
        num_modalities=num_modalities,
        num_domains=num_domains
    )
    return model