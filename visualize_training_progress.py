import os
import torch
import matplotlib.pyplot as plt
import random

MODALITIES = ["t1", "t1ce", "t2", "flair"]

@torch.no_grad()
def visualize_training_progress(model,dataloader,device,epoch,save_dir,num_samples=3,):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    batch = next(iter(dataloader))
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    images = batch["image"]          # [B,4,H,W,D]
    domain_id = batch["domain_id"]
    B, _, H, W, D = images.shape
    target_modality_id = epoch % 4
    target_modality_name = MODALITIES[target_modality_id]
    target_ids = torch.full(
        (B,), target_modality_id, device=device, dtype=torch.long
    )
    modality_mask = torch.ones(B, 4, device=device)
    modality_mask[:, target_modality_id] = 0.0
    # Input (missing target modality)
    x_input = images.clone()
    x_input[:, target_modality_id] = 0.0
    x_dict = {
        "t1":    x_input[:, 0:1],
        "t1ce":  x_input[:, 1:2],
        "t2":    x_input[:, 2:3],
        "flair": x_input[:, 3:4],
    }
    x_target_gt = images[:, target_modality_id:target_modality_id+1]
    # Run inference branch: do not pass explicit timestep; run reverse diffusion
    outputs = model(
        x_dict=x_dict,
        x_target_gt=None,
        target_modality_name=target_modality_name,
        target_modality_id=target_ids,
        modality_mask=modality_mask,
        domain_id=domain_id,
        num_infer_steps=model.num_timesteps,   # 1000
    )
    synth = outputs["x_hat"]
    mid = D // 2
    for i in range(min(num_samples, B)):
        gt = images[i, target_modality_id, :, :, mid].cpu()
        gen = synth[i, 0, :, :, mid].cpu()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(gt, cmap="gray")
        axes[0].set_title("GT")
        axes[1].imshow(torch.zeros_like(gt), cmap="gray")
        axes[1].set_title("Input (missing)")
        axes[2].imshow(gen, cmap="gray")
        axes[2].set_title("Generated")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                save_dir,
                f"epoch_{epoch:03d}_sample_{i}_{target_modality_name}.png"
            ),
            dpi=150
        )
        plt.close()
    model.train()
