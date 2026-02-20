import os
import time
import torch
import math
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from helper import EarlyStopping, load_checkpoint, save_checkpoint, EMA
from loss.losses import DiffusionSynthesisLoss
import random
from tqdm import tqdm
from visualize_training_progress import visualize_training_progress
try:
    from skimage.metrics import structural_similarity as ssim_sk
    has_skimage = True
except Exception:
    has_skimage = False
try:
    from pytorch_msssim import ms_ssim
    has_ms_ssim = True
except Exception:
    has_ms_ssim = False

MODALITIES = ["t1", "t1ce", "t2", "flair"]
def train(model, train_loader, device, num_epochs=300, lr=1e-4, checkpoint_dir="/home/nvsinh1/brats_segmentation/synthesis-model/model"):
    model.to(device)
    # -------- Loss --------
    criterion = DiffusionSynthesisLoss()
    # -------- Optimizer & Scheduler --------
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )
    # -------- EMA --------
    ema = EMA(decay=0.9999)
    ema.register(model)
    # -------- Resume --------
    start_epoch = 0
    best_loss = float("inf")
    start_epoch, best_loss = load_checkpoint(
        os.path.join(checkpoint_dir, "checkpoints"),
        model,
        optimizer,
        scheduler,
        ema
    )
    # -------- AMP scaler --------
    scaler = GradScaler(enabled=(device.type == "cuda"))
    # -------- Early stopping --------
    early_stopper = EarlyStopping(patience=30)
    print("🚀 Start training")
    print(f"Total epochs: {num_epochs}")
    print(f"Start epoch: {start_epoch}")
    # -------- Training loop --------
    for epoch in range(start_epoch, num_epochs):
        # visualize using EMA weights for better quality
        ema.store(model)
        ema.copy_to(model)
        visualize_training_progress(
            model=model,
            dataloader=train_loader,
            device=device,
            epoch=epoch,
            save_dir=os.path.join(checkpoint_dir, "visualize_train"),
            num_samples=3
        )
        ema.restore(model)
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        epoch_mae = 0.0
        metric_count = 0
        t_start = time.time()
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs-1}",
            total=len(train_loader),
            ncols=100,
            leave=False
        )
        for step, batch in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)
            # Move batch to device
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            images = batch["image"]        # [B, 4, H, W, D]
            domain_id = batch["domain_id"] # [B]
            B = images.shape[0]
            # 1. Randomly choose target modality (missing one)
            target_modality_id_int = random.randint(0, 3)
            target_modality_name = MODALITIES[target_modality_id_int]
            target_modality_id = torch.full(
                (B,),
                target_modality_id_int,
                device=device,
                dtype=torch.long
            )
            # 2. Build modality mask 1 = available, 0 = missing (target)
            modality_mask = torch.ones(B, 4, device=device)
            modality_mask[:, target_modality_id_int] = 0.0
            # 3. Build INPUT (missing modality = zero)
            x_input = images.clone()
            x_input[:, target_modality_id_int] = 0.0
            x_dict = {
                "t1":    x_input[:, 0:1],
                "t1ce":  x_input[:, 1:2],
                "t2":    x_input[:, 2:3],
                "flair": x_input[:, 3:4],
            }
            # 4. Build GT TARGET 
            x_target_gt = images[:, target_modality_id_int:target_modality_id_int+1]
            # 5. Forward model (target GT truyền RIÊNG)
            # Sample timesteps with a bias towards smaller timesteps (more structure)
            # with probability 0.7 use biased sampling, otherwise uniform
            if random.random() < 0.7:
                alpha = 1.8
                u = torch.rand(B, device=device)
                t = ( (u ** alpha) * (model.num_timesteps - 1) ).long()
            else:
                t = torch.randint(0, model.num_timesteps, (B,), device=device)
            # classifier-free guidance training: occasionally drop conditioning
            uc_prob = 0.1
            unconditional = (random.random() < uc_prob)

            with autocast(enabled=(device.type == "cuda")):
                outputs = model(
                    x_dict=x_dict,
                    x_target_gt=x_target_gt,
                    target_modality_name=target_modality_name,
                    target_modality_id=target_modality_id,
                    modality_mask=modality_mask,
                    domain_id=domain_id,
                    t=t,
                    unconditional=unconditional,
                )
                # 6. Loss (primary: noise MSE)
                loss = criterion(outputs)
                # small reconstruction loss on low-t samples to encourage structure
                recon_weight = 0.2
                recon_loss = torch.tensor(0.0, device=device)
                # latent perceptual weight (MSE in latent space)
                latent_perc_w = 0.01
                latent_perc = torch.tensor(0.0, device=device)
                # optional MS-SSIM weight
                ms_w = 0.01
                # metrics only meaningful for low-noise timesteps
                metric_t_thresh = 50
                metric_mask = (t < metric_t_thresh)
                if outputs.get("x_hat") is not None and metric_mask.any():
                    x_hat_dev = outputs["x_hat"]
                    x_gt_dev = x_target_gt
                    x_hat_small = x_hat_dev[metric_mask]
                    x_gt_small = x_gt_dev[metric_mask]
                    if x_hat_small.numel() > 0:
                        recon_loss = F.l1_loss(x_hat_small, x_gt_small)
                        # MS-SSIM if available (on small samples)
                        if has_ms_ssim:
                            try:
                                ms = ms_ssim(x_hat_small, x_gt_small, data_range=1.0, size_average=True)
                                ms_loss = 1.0 - ms
                            except Exception:
                                ms_loss = torch.tensor(0.0, device=device)
                        else:
                            ms_loss = torch.tensor(0.0, device=device)
                else:
                    ms_loss = torch.tensor(0.0, device=device)

                # latent perceptual loss between z0_hat and z_target if available
                if outputs.get("z_0_hat") is not None and outputs.get("z_target") is not None:
                    latent_perc = F.mse_loss(outputs["z_0_hat"], outputs["z_target"]) * latent_perc_w

                total_loss = loss + recon_weight * recon_loss + ms_w * ms_loss + latent_perc

            # --- Metrics (image space) computed only for low-noise timesteps ---
            if outputs.get("x_hat") is not None and metric_mask.any():
                x_hat = outputs["x_hat"].detach().cpu()
                x_target = x_target_gt.detach().cpu()
                # compute PSNR, slice-SSIM (center slice), MAE per selected sample
                idxs = metric_mask.detach().cpu().nonzero(as_tuple=False).view(-1).tolist()
                for i in idxs:
                    xh = x_hat[i, 0].numpy()  # H,W,D
                    xt = x_target[i, 0].numpy()
                    mse = float(((xh - xt) ** 2).mean())
                    data_range = float(xt.max() - xt.min())
                    if data_range <= 1e-8:
                        data_range = 1.0
                    psnr = 20 * math.log10(data_range) - 10 * math.log10(mse + 1e-12)
                    mae = float(abs(xh - xt).mean())
                    ssim_val = None
                    if has_skimage:
                        try:
                            cs = xt.shape[-1] // 2
                            ssim_val = ssim_sk(xt[..., cs], xh[..., cs], data_range=(xt[..., cs].max() - xt[..., cs].min()) if (xt[..., cs].max() != xt[..., cs].min()) else 1.0)
                        except Exception:
                            ssim_val = None
                    epoch_psnr += psnr
                    epoch_mae += mae
                    if ssim_val is not None:
                        epoch_ssim += ssim_val
                    metric_count += 1

            # backward on total_loss (noise + small recon)
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            # update EMA after optimizer step
            ema.update(model)
            epoch_loss += total_loss.item()
            # 7. Logging (report full objective used for backward)
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })
        # LR scheduler step 
        scheduler.step()
        epoch_loss /= len(train_loader)
        epoch_time = time.time() - t_start
        avg_psnr = epoch_psnr / metric_count if metric_count > 0 else 0.0
        avg_ssim = epoch_ssim / metric_count if (metric_count > 0 and epoch_ssim > 0) else 0.0
        avg_mae = epoch_mae / metric_count if metric_count > 0 else 0.0
        print(
            f"[Epoch {epoch:03d}] "
            f"Loss: {epoch_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"PSNR: {avg_psnr:.2f} | "
            f"SSIM: {avg_ssim:.4f} | "
            f"MAE: {avg_mae:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_state": ema.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_loss": best_loss,
            },
            is_best=is_best,
            ckpt_dir=os.path.join(checkpoint_dir, "checkpoints"),
            max_keep=3
        )
        early_stopper.step(epoch_loss)
        if early_stopper.stop:
            print("⏹️ Early stopping triggered")
            break
    print("✅ Training finished")
