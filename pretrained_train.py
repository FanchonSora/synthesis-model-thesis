import os
import time
import json
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
from utils import EarlyStopping, load_checkpoint, save_checkpoint, EMA
from loss.losses import DiffusionSynthesisLoss
from visualize_training import visualize_training_progress


MODALITIES = ["t1", "t1ce", "t2", "flair"]


def normalize_per_sample(x: torch.Tensor) -> torch.Tensor:
    B = x.shape[0]
    x_flat = x.view(B, -1)
    minv = x_flat.min(dim=1)[0].view(B, 1, 1, 1, 1)
    maxv = x_flat.max(dim=1)[0].view(B, 1, 1, 1, 1)
    return (x - minv) / (maxv - minv + 1e-8)


@torch.no_grad()
def compute_batch_metrics(
    x_hat: torch.Tensor,
    x_gt: torch.Tensor,
    ssim3d: StructuralSimilarityIndexMeasure,
):
    mse_val = F.mse_loss(x_hat, x_gt)
    mae_val = F.l1_loss(x_hat, x_gt)

    x_hat_norm = normalize_per_sample(x_hat)
    x_gt_norm = normalize_per_sample(x_gt)
    mse_norm = F.mse_loss(x_hat_norm, x_gt_norm)
    psnr_val = 10.0 * torch.log10(1.0 / (mse_norm + 1e-8))

    ssim_val = ssim3d(x_hat_norm, x_gt_norm)
    ssim3d.reset()

    return {
        "mse": mse_val,
        "mae": mae_val,
        "psnr": psnr_val,
        "ssim": ssim_val,
    }


def sample_target_modalities(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, 4, (batch_size,), device=device, dtype=torch.long)


def build_inputs_from_targets(images: torch.Tensor, target_modality_id: torch.Tensor):
    B = images.shape[0]
    device = images.device

    modality_mask = torch.ones(B, 4, device=device, dtype=images.dtype)
    modality_mask.scatter_(1, target_modality_id.unsqueeze(1), 0.0)

    x_input = images.clone()
    batch_idx = torch.arange(B, device=device)
    x_input[batch_idx, target_modality_id] = 0.0

    x_dict = {
        "t1": x_input[:, 0:1],
        "t1ce": x_input[:, 1:2],
        "t2": x_input[:, 2:3],
        "flair": x_input[:, 3:4],
    }
    x_target_gt = images[batch_idx, target_modality_id].unsqueeze(1)
    return x_dict, x_target_gt, modality_mask


@torch.no_grad()
def evaluate_generation_on_val(
    model,
    val_loader,
    device,
    num_infer_steps: int = 1000,
    max_batches: int | None = None,
):
    model.eval()
    ssim3d = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_psnr = 0.0
    total_ssim = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_count = 0

    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        images = batch["image"]
        domain_id = batch["domain_id"]
        B = images.shape[0]

        for mod_id in range(4):
            target_modality_id = torch.full(
                (B,), mod_id, device=device, dtype=torch.long
            )
            x_dict, x_target_gt, modality_mask = build_inputs_from_targets(
                images, target_modality_id
            )

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(
                    x_dict=x_dict,
                    x_target_gt=None,
                    target_modality_name=None,
                    target_modality_id=target_modality_id,
                    modality_mask=modality_mask,
                    domain_id=domain_id,
                    t=None,
                    unconditional=False,
                    num_infer_steps=num_infer_steps,
                )

            x_hat = outputs["x_hat"]

            metrics = compute_batch_metrics(
                x_hat=x_hat,
                x_gt=x_target_gt,
                ssim3d=ssim3d,
            )

            total_psnr += metrics["psnr"].item()
            total_ssim += metrics["ssim"].item()
            total_mae += metrics["mae"].item()
            total_mse += metrics["mse"].item()
            total_count += 1

    denom = max(total_count, 1)
    return {
        "psnr": total_psnr / denom,
        "ssim": total_ssim / denom,
        "mae": total_mae / denom,
        "mse": total_mse / denom,
    }


def train(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs: int = 340,
    lr: float = 1e-5,
    checkpoint_dir: str = "/home/nvsinh1/brats_segmentation/synthesis-model/model",
    eval_every: int = 5,
    val_infer_steps: int = 1000,
):
    model.to(device)

    ssim3d = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    criterion = DiffusionSynthesisLoss(
        alphas_cumprod=model.alphas_cumprod,
        min_snr_gamma=5.0,
        lambda_diff=1.0,
        lambda_recon=0.40,
        lambda_ssim=0.15,
        lambda_mid=0.03,
        recon_snr_threshold=0.5,
        use_charbonnier=True,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    ema = EMA(decay=0.9999)
    ema.register(model)
    start_epoch, _ = load_checkpoint(
        os.path.join(checkpoint_dir, "checkpoints"),
        model,
        optimizer,
        scheduler,
        ema=None,
    )
    scaler = GradScaler(device=device.type, enabled=(device.type == "cuda"))
    early_stopper = EarlyStopping(patience=30)

    best_val_psnr = -float("inf")

    print("🚀 Start pretrained fine-tuning")
    print(f"Total epochs: {num_epochs}")
    print(f"Start epoch : {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()

        epoch_loss = 0.0
        epoch_diff_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_ssim_loss = 0.0
        epoch_mid_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        epoch_mae = 0.0
        metric_count = 0

        t_start = time.time()

        if epoch % eval_every == 0:
            ema.store(model)
            ema.copy_to(model)
            try:
                visualize_training_progress(
                    model=model,
                    dataloader=train_loader,
                    device=device,
                    epoch=epoch,
                    save_dir=os.path.join(checkpoint_dir, "visualize_train"),
                    num_samples=3,
                )
            except Exception as e:
                print(f"[WARN] Visualize failed at epoch {epoch}: {e}")
            finally:
                ema.restore(model)
                model.train()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs - 1}",
            total=len(train_loader),
            ncols=140,
            leave=False,
        )

        for step, batch in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)

            images = batch["image"]        # [B,4,H,W,D]
            domain_id = batch["domain_id"] # [B]
            B = images.shape[0]

            target_modality_id = sample_target_modalities(B, device)
            x_dict, x_target_gt, modality_mask = build_inputs_from_targets(
                images, target_modality_id
            )

            t = torch.randint(0, model.num_timesteps, (B,), device=device)
            unconditional = torch.rand(1, device=device).item() < 0.1

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(
                    x_dict=x_dict,
                    x_target_gt=x_target_gt,
                    target_modality_name=None,
                    target_modality_id=target_modality_id,
                    modality_mask=modality_mask,
                    domain_id=domain_id,
                    t=t,
                    unconditional=unconditional,
                )
                total_loss, loss_items = criterion(outputs, ssim_fn=ssim3d)

            if not torch.isfinite(total_loss):
                print(f"[WARN] Non-finite loss at epoch {epoch}, step {step}. Skip batch.")
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            epoch_loss += total_loss.item()
            epoch_diff_loss += loss_items.get("diffusion_loss", torch.tensor(0.0, device=device)).item()
            epoch_recon_loss += loss_items.get("recon_loss", torch.tensor(0.0, device=device)).item()
            epoch_ssim_loss += loss_items.get("ssim_loss", torch.tensor(0.0, device=device)).item()
            epoch_mid_loss += loss_items.get("mid_loss", torch.tensor(0.0, device=device)).item()

            if outputs.get("x_hat") is not None:
                with torch.no_grad():
                    metrics = compute_batch_metrics(
                        x_hat=outputs["x_hat"].detach(),
                        x_gt=x_target_gt.detach(),
                        ssim3d=ssim3d,
                    )
                    epoch_psnr += metrics["psnr"].item()
                    epoch_ssim += metrics["ssim"].item()
                    epoch_mae += metrics["mae"].item()
                    metric_count += 1
                    psnr_show = metrics["psnr"].item()
                    ssim_show = metrics["ssim"].item()
            else:
                psnr_show = 0.0
                ssim_show = 0.0

            pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "diff": f"{loss_items.get('diffusion_loss', torch.tensor(0.0, device=device)).item():.4f}",
                    "recon": f"{loss_items.get('recon_loss', torch.tensor(0.0, device=device)).item():.4f}",
                    "psnr": f"{psnr_show:.2f}",
                    "ssim": f"{ssim_show:.3f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        scheduler.step()

        num_steps = max(len(train_loader), 1)
        avg_loss = epoch_loss / num_steps
        avg_diff_loss = epoch_diff_loss / num_steps
        avg_recon_loss = epoch_recon_loss / num_steps
        avg_ssim_loss = epoch_ssim_loss / num_steps
        avg_mid_loss = epoch_mid_loss / num_steps
        avg_psnr = epoch_psnr / max(metric_count, 1)
        avg_ssim = epoch_ssim / max(metric_count, 1)
        avg_mae = epoch_mae / max(metric_count, 1)

        print(
            f"[Epoch {epoch:03d}] "
            f"Loss: {avg_loss:.4f} | "
            f"Diff: {avg_diff_loss:.4f} | "
            f"Recon: {avg_recon_loss:.4f} | "
            f"SSIM_L: {avg_ssim_loss:.4f} | "
            f"Mid: {avg_mid_loss:.4f} | "
            f"PSNR: {avg_psnr:.2f} | "
            f"SSIM: {avg_ssim:.4f} | "
            f"MAE: {avg_mae:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {time.time() - t_start:.1f}s"
        )

        val_results = None
        is_best = False

        if ((epoch + 1) % eval_every == 0) and (val_loader is not None):
            ema.store(model)
            ema.copy_to(model)
            try:
                val_results = evaluate_generation_on_val(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    num_infer_steps=val_infer_steps,
                    max_batches=None,
                )
            finally:
                ema.restore(model)

            print(
                f"[Val {epoch:03d}] "
                f"PSNR: {val_results['psnr']:.4f} | "
                f"SSIM: {val_results['ssim']:.4f} | "
                f"MAE: {val_results['mae']:.4f} | "
                f"MSE: {val_results['mse']:.4f}"
            )

            is_best = val_results["psnr"] > best_val_psnr
            if is_best:
                best_val_psnr = val_results["psnr"]

        save_state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_loss": avg_loss,
            "best_val_psnr": best_val_psnr,
        }

        save_checkpoint(
            save_state,
            is_best=is_best,
            ckpt_dir=os.path.join(checkpoint_dir, "checkpoints"),
            max_keep=5,
        )

        if val_results is not None:
            os.makedirs(os.path.join(checkpoint_dir, "logs"), exist_ok=True)
            with open(
                os.path.join(checkpoint_dir, "logs", f"val_epoch_{epoch:03d}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(val_results, f, indent=2)

        early_stopper.step(avg_loss)
        if early_stopper.stop:
            print("⏹️ Early stopping triggered")
            break

    print("✅ Pretrained fine-tuning finished")