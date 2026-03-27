import os
import torch
import matplotlib.pyplot as plt

MODALITIES = ["t1", "t1ce", "t2", "flair"]

@torch.no_grad()
def visualize_training_progress(model, dataloader, device, epoch, save_dir, num_samples=3):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    try:
        batch = next(iter(dataloader))
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)
        images = batch["image"]       # [B, 4, H, W, D]
        domain_id = batch["domain_id"]
        B, _, H, W, D = images.shape
        target_modality_id_int = epoch % 4
        target_ids = torch.full((B,), target_modality_id_int, device=device, dtype=torch.long)
        modality_mask = torch.ones(B, 4, device=device, dtype=images.dtype)
        modality_mask[:, target_modality_id_int] = 0.0
        x_input = images.clone()
        x_input[:, target_modality_id_int] = 0.0
        x_dict = {
            "t1": x_input[:, 0:1],
            "t1ce": x_input[:, 1:2],
            "t2": x_input[:, 2:3],
            "flair": x_input[:, 3:4],
        }
        x_target_gt = images[:, target_modality_id_int:target_modality_id_int + 1]
        # low-noise visualization
        t_vis = torch.full((B,), 10, device=device, dtype=torch.long)
        outputs = model(
            x_dict=x_dict,
            x_target_gt=x_target_gt,
            target_modality_name=None,
            target_modality_id=target_ids,
            modality_mask=modality_mask,
            domain_id=domain_id,
            t=t_vis,
            unconditional=False,
        )
        synth = outputs["x_hat"]   # [B,1,H,W,D]
        mid = D // 2
        def norm_for_display(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor - tensor.min()
            return tensor / (tensor.max() + 1e-8)
        input_mod_id = (target_modality_id_int + 1) % 4
        for i in range(min(num_samples, B)):
            gt = images[i, target_modality_id_int, :, :, mid].detach().cpu().float()
            gen = synth[i, 0, :, :, mid].detach().cpu().float()
            inp = images[i, input_mod_id, :, :, mid].detach().cpu().float()
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(norm_for_display(gt), cmap="gray")
            axes[0].set_title(f"GT ({MODALITIES[target_modality_id_int]})")
            axes[1].imshow(norm_for_display(inp), cmap="gray")
            axes[1].set_title(f"Input ({MODALITIES[input_mod_id]})")
            axes[2].imshow(norm_for_display(gen), cmap="gray")
            axes[2].set_title("Generated (t=10)")
            for ax in axes:
                ax.axis("off")
            plt.suptitle(
                f"Epoch {epoch:03d} — target: {MODALITIES[target_modality_id_int]}",
                fontsize=12,
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    save_dir,
                    f"epoch_{epoch:03d}_sample_{i}_{MODALITIES[target_modality_id_int]}.png",
                ),
                dpi=150,
            )
            plt.close()
    except Exception as e:
        print(f"[WARN] visualize_training_progress failed: {e}")
    finally:
        model.train()