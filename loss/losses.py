import torch
import torch.nn as nn

class DiffusionSynthesisLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Primary objective: predict noise (epsilon)
        self.noise_loss_fn = nn.MSELoss()
        # deep supervision weight (latent-level)
        self.deep_sup_w = 0.1
        # mid-level deep supervision weight (projected UNet mid -> latent)
        self.deep_mid_w = 0.1
        # Optional image loss removed to avoid conflicting training signal
        # self.img_loss_fn = nn.L1Loss()

    def forward(self, outputs):
        loss_noise = self.noise_loss_fn(
            outputs["epsilon_hat"],
            outputs["epsilon"]
        )
        # Deep supervision: encourage fused conditioning to match encoded target latent
        deep_loss = torch.tensor(0.0, device=outputs["epsilon_hat"].device)
        if outputs.get("z_target") is not None and outputs.get("z_cond") is not None:
            try:
                deep_loss = nn.functional.mse_loss(outputs["z_cond"], outputs["z_target"]) * self.deep_sup_w
            except Exception:
                deep_loss = torch.tensor(0.0, device=outputs["epsilon_hat"].device)
        # mid-level deep supervision (UNet mid features projected -> latent space)
        mid_loss = torch.tensor(0.0, device=outputs["epsilon_hat"].device)
        if outputs.get("z_mid_pred") is not None and outputs.get("z_target") is not None:
            try:
                mid_loss = nn.functional.mse_loss(outputs["z_mid_pred"], outputs["z_target"]) * self.deep_mid_w
            except Exception:
                mid_loss = torch.tensor(0.0, device=outputs["epsilon_hat"].device)

        return loss_noise + deep_loss + mid_loss

