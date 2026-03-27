import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionSynthesisLoss(nn.Module):
    def __init__(self,alphas_cumprod: torch.Tensor,min_snr_gamma: float = 5.0,lambda_diff: float = 1.0,lambda_recon: float = 0.25,lambda_ssim: float = 0.10,lambda_mid: float = 0.03,recon_snr_threshold: float = 1.0,use_charbonnier: bool = True,):
        super().__init__()
        self.register_buffer("alphas_cumprod", alphas_cumprod.clone().float())
        self.min_snr_gamma = min_snr_gamma
        self.lambda_diff = lambda_diff
        self.lambda_recon = lambda_recon
        self.lambda_ssim = lambda_ssim
        self.lambda_mid = lambda_mid
        self.recon_snr_threshold = recon_snr_threshold
        self.use_charbonnier = use_charbonnier

    @staticmethod
    def charbonnier_loss(x, y, eps: float = 1e-3):
        diff = x - y
        return torch.sqrt(diff * diff + eps * eps).mean()

    @staticmethod
    def compute_snr(alphas_cumprod_t: torch.Tensor):
        # snr = alpha_bar / (1 - alpha_bar)
        return alphas_cumprod_t / (1.0 - alphas_cumprod_t).clamp(min=1e-8)

    def min_snr_weight_vpred(self, t: torch.Tensor):
        alpha_bar_t = self.alphas_cumprod[t]                        # [B]
        snr = self.compute_snr(alpha_bar_t)                        # [B]
        # v-pred weighting
        w = torch.minimum(snr, torch.full_like(snr, self.min_snr_gamma)) / (snr + 1.0)
        return w, snr

    def forward(self, outputs: dict, ssim_fn=None):
        device = outputs["v_hat"].device
        v_hat = torch.nan_to_num(outputs["v_hat"], nan=0.0, posinf=1e4, neginf=-1e4)
        v_target = torch.nan_to_num(outputs["v_target"], nan=0.0, posinf=1e4, neginf=-1e4)
        t = outputs["t"].long()
        # Core diffusion loss
        w, snr = self.min_snr_weight_vpred(t)                      # [B]
        per_sample_diff = (v_hat - v_target).pow(2).flatten(1).mean(dim=1)   # [B]
        diffusion_loss = (w * per_sample_diff).mean()
        total_loss = self.lambda_diff * diffusion_loss
        loss_dict = {
            "diffusion_loss": diffusion_loss.detach(),
        }
        # Deep supervision (light)
        if outputs.get("z_mid_pred") is not None and outputs.get("z_target") is not None:
            z_mid = torch.nan_to_num(outputs["z_mid_pred"], nan=0.0, posinf=1e4, neginf=-1e4)
            z_target = torch.nan_to_num(outputs["z_target"], nan=0.0, posinf=1e4, neginf=-1e4)
            mid_loss = F.l1_loss(z_mid, z_target)
            total_loss = total_loss + self.lambda_mid * mid_loss
            loss_dict["mid_loss"] = mid_loss.detach()

        # Image recon only at low/mid noise
        if outputs.get("x_hat") is not None and outputs.get("x_target") is not None:
            x_hat = outputs["x_hat"]
            x_gt = outputs["x_target"]
            low_noise_mask = (snr >= self.recon_snr_threshold).float()   # [B]
            if low_noise_mask.any():
                if self.use_charbonnier:
                    per_sample_recon = torch.sqrt(
                        (x_hat - x_gt).pow(2) + 1e-6
                    ).flatten(1).mean(dim=1)
                else:
                    per_sample_recon = (x_hat - x_gt).abs().flatten(1).mean(dim=1)
                recon_loss = (low_noise_mask * per_sample_recon).sum() / low_noise_mask.sum().clamp(min=1.0)
                total_loss = total_loss + self.lambda_recon * recon_loss
                loss_dict["recon_loss"] = recon_loss.detach()
                if ssim_fn is not None:
                    x_hat_ssim = self._normalize_per_sample(x_hat)
                    x_gt_ssim = self._normalize_per_sample(x_gt)
                    ssim_val = ssim_fn(x_hat_ssim, x_gt_ssim)
                    ssim_fn.reset()
                    ssim_loss = 1.0 - ssim_val
                    total_loss = total_loss + self.lambda_ssim * ssim_loss
                    loss_dict["ssim_loss"] = ssim_loss.detach()
        loss_dict["total_loss"] = total_loss.detach()
        return total_loss, loss_dict

    @staticmethod
    def _normalize_per_sample(x):
        B = x.shape[0]
        x_flat = x.view(B, -1)
        minv = x_flat.min(dim=1)[0].view(B, 1, 1, 1, 1)
        maxv = x_flat.max(dim=1)[0].view(B, 1, 1, 1, 1)
        return (x - minv) / (maxv - minv + 1e-8)