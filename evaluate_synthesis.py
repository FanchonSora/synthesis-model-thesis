import os
import csv
import json
from typing import Dict, List, Optional
import torch
from torch.amp import autocast
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
from model.architecture import create_model
from data.brats_dataset import test_loader
from synthesis_utils import build_inputs_from_fixed_target, MODALITIES
from utils import load_checkpoint, EMA

# Basic tensor helpers
def normalize_per_sample(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"Expected 5D tensor [B,1,H,W,D], got {tuple(x.shape)}")
    B = x.shape[0]
    x_flat = x.view(B, -1)
    minv = x_flat.min(dim=1)[0].view(B, 1, 1, 1, 1)
    maxv = x_flat.max(dim=1)[0].view(B, 1, 1, 1, 1)
    return (x - minv) / (maxv - minv + 1e-8)

def to_bcdhw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"Expected 5D tensor, got shape={tuple(x.shape)}")
    return x.permute(0, 1, 4, 2, 3).contiguous()

@torch.no_grad()
def compute_samplewise_metrics(
    x_hat: torch.Tensor,
    x_gt: torch.Tensor,
    ssim3d: StructuralSimilarityIndexMeasure,
) -> Dict[str, torch.Tensor]:
    if x_hat.shape != x_gt.shape:
        raise ValueError(f"Shape mismatch: x_hat={x_hat.shape}, x_gt={x_gt.shape}")
    mae = torch.mean(torch.abs(x_hat - x_gt), dim=(1, 2, 3, 4))
    mse = torch.mean((x_hat - x_gt) ** 2, dim=(1, 2, 3, 4))
    x_hat_norm = normalize_per_sample(x_hat)
    x_gt_norm = normalize_per_sample(x_gt)
    mse_norm = torch.mean((x_hat_norm - x_gt_norm) ** 2, dim=(1, 2, 3, 4))
    psnr = 10.0 * torch.log10(1.0 / (mse_norm + 1e-8))
    x_hat_3d = to_bcdhw(x_hat_norm)
    x_gt_3d = to_bcdhw(x_gt_norm)
    ssim_vals = []
    B = x_hat.shape[0]
    for i in range(B):
        val = ssim3d(x_hat_3d[i:i+1], x_gt_3d[i:i+1])
        ssim_vals.append(val.detach())
        ssim3d.reset()
    ssim = torch.stack(ssim_vals, dim=0)
    return {
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
    }

# Aggregation helpers
METRIC_KEYS = ["mae", "mse", "psnr", "ssim"]

def make_metric_sum_dict() -> Dict[str, float]:
    return {k: 0.0 for k in METRIC_KEYS}

def average_metric_dict(metric_sum: Dict[str, float], count: int) -> Dict[str, float]:
    denom = max(count, 1)
    return {k: metric_sum[k] / denom for k in METRIC_KEYS}

def write_case_csv(rows: List[Dict], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def print_summary(results: Dict) -> None:
    print("\n================ SYNTHESIS GENERATION EVALUATION ================")
    print(f"Checkpoint              : {results['checkpoint_name']}")
    print(f"Use EMA                 : {results['use_ema']}")
    print(f"Num infer steps         : {results['num_infer_steps']}")
    print(f"Num test cases          : {results['num_cases']}")
    print(f"Num modality evals      : {results['num_modality_evals']}")
    print("\n[Overall]")
    for k, v in results["overall"].items():
        print(f"  {k:>6s}: {v:.6f}")
    print("\n[Per Modality]")
    for mod_name, vals in results["per_modality"].items():
        print(f"  {mod_name.upper()}")
        for k, v in vals.items():
            print(f"    {k:>6s}: {v:.6f}")

# Model loading
def load_model_for_eval(
    checkpoint_dir: str,
    device: torch.device,
    latent_dim: int = 128,
    num_timesteps: int = 1000,
    num_modalities: int = 4,
    num_domains: int = 3,
    use_ema: bool = True,
):
    model = create_model(
        latent_dim=latent_dim,
        num_timesteps=num_timesteps,
        num_modalities=num_modalities,
        num_domains=num_domains,
    ).to(device)
    ema = EMA(decay=0.9999) if use_ema else None
    if ema is not None:
        ema.register(model)
    _, _ = load_checkpoint(
        ckpt_dir=checkpoint_dir,
        model=model,
        optimizer=None,
        scheduler=None,
        ema=ema,
    )
    if use_ema and ema is not None:
        ema.copy_to(model)
    model.eval()
    return model

# Core evaluation: single missing modality generation
@torch.no_grad()
def evaluate_missing_modality_generation(
    model,
    loader,
    device,
    num_infer_steps: int = 250,
    max_batches: Optional[int] = None,
    autocast_enabled: bool = True,
    save_case_predictions: bool = False,
    prediction_dir: Optional[str] = None,
):
    model.eval()
    ssim3d = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    overall_sum = make_metric_sum_dict()
    overall_count = 0
    per_modality_sum = {m: make_metric_sum_dict() for m in MODALITIES}
    per_modality_count = {m: 0 for m in MODALITIES}
    per_case_rows: List[Dict] = []
    num_cases = 0
    if save_case_predictions and prediction_dir is not None:
        os.makedirs(prediction_dir, exist_ok=True)
    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    pbar = tqdm(
        enumerate(loader),
        total=total_batches,
        desc=f"Generation Eval ({num_infer_steps} steps)",
        ncols=140,
    )
    for batch_idx, batch in pbar:
        if max_batches is not None and batch_idx >= max_batches:
            break
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)
        images = batch["image"]              # [B,4,H,W,D]
        domain_id = batch["domain_id"]       # [B]
        case_ids = batch.get("case_id", None)
        B = images.shape[0]
        num_cases += B
        for mod_id, mod_name in enumerate(MODALITIES):
            target_modality_id = torch.full((B,), mod_id, device=device, dtype=torch.long)
            x_dict, x_target_gt, modality_mask = build_inputs_from_fixed_target(
                images=images,
                target_modality_id=target_modality_id,
            )
            with autocast(device_type=device.type, enabled=(device.type == "cuda" and autocast_enabled)):
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
            x_hat = outputs["x_hat"]  # [B,1,H,W,D]
            metrics = compute_samplewise_metrics(
                x_hat=x_hat,
                x_gt=x_target_gt,
                ssim3d=ssim3d,
            )
            for k in METRIC_KEYS:
                vals = metrics[k]
                overall_sum[k] += vals.sum().item()
                per_modality_sum[mod_name][k] += vals.sum().item()
            overall_count += B
            per_modality_count[mod_name] += B
            # Save per-case rows
            for i in range(B):
                case_id = case_ids[i] if case_ids is not None else f"case_{batch_idx:04d}_{i:02d}"
                row = {
                    "case_id": str(case_id),
                    "target_modality": mod_name,
                    "mae": float(metrics["mae"][i].item()),
                    "mse": float(metrics["mse"][i].item()),
                    "psnr": float(metrics["psnr"][i].item()),
                    "ssim": float(metrics["ssim"][i].item()),
                }
                per_case_rows.append(row)
                if save_case_predictions and prediction_dir is not None:
                    case_dir = os.path.join(prediction_dir, str(case_id))
                    os.makedirs(case_dir, exist_ok=True)
                    torch.save(
                        {
                            "target_modality": mod_name,
                            "x_hat": x_hat[i].detach().cpu(),
                            "x_gt": x_target_gt[i].detach().cpu(),
                            "inputs": {k: v[i].detach().cpu() for k, v in x_dict.items()},
                        },
                        os.path.join(case_dir, f"missing_{mod_name}.pt"),
                    )
        running_psnr = overall_sum["psnr"] / max(overall_count, 1)
        running_ssim = overall_sum["ssim"] / max(overall_count, 1)
        pbar.set_postfix({
            "cases": num_cases,
            "evals": overall_count,
            "psnr": f"{running_psnr:.3f}",
            "ssim": f"{running_ssim:.3f}",
        })
    results = {
        "overall": average_metric_dict(overall_sum, overall_count),
        "per_modality": {
            mod_name: average_metric_dict(per_modality_sum[mod_name], per_modality_count[mod_name])
            for mod_name in MODALITIES
        },
        "num_cases": int(num_cases),
        "num_modality_evals": int(overall_count),
        "num_infer_steps": int(num_infer_steps),
        "per_case": per_case_rows,
    }
    return results

# Multi-step evaluation helper
@torch.no_grad()
def evaluate_multiple_sampling_steps(
    model,
    loader,
    device,
    infer_steps_list: List[int],
    max_batches: Optional[int] = None,
):
    all_results = {}
    for steps in infer_steps_list:
        result = evaluate_missing_modality_generation(
            model=model,
            loader=loader,
            device=device,
            num_infer_steps=steps,
            max_batches=max_batches,
            autocast_enabled=True,
            save_case_predictions=False,
            prediction_dir=None,
        )
        all_results[str(steps)] = {
            "overall": result["overall"],
            "per_modality": result["per_modality"],
            "num_cases": result["num_cases"],
            "num_modality_evals": result["num_modality_evals"],
            "num_infer_steps": result["num_infer_steps"],
        }
    return all_results

# Main
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = "/home/nvsinh1/brats_segmentation/synthesis-model/model/checkpoints"
    save_root = "/home/nvsinh1/brats_segmentation/synthesis-model/model/eval_task"
    os.makedirs(save_root, exist_ok=True)
    use_ema = True
    num_infer_steps = 1000
    model = load_model_for_eval(
        checkpoint_dir=checkpoint_dir,
        device=device,
        latent_dim=128,
        num_timesteps=1000,
        num_modalities=4,
        num_domains=3,
        use_ema=use_ema,
    )
    results = evaluate_missing_modality_generation(
        model=model,
        loader=test_loader,
        device=device,
        num_infer_steps=num_infer_steps,
        max_batches=None,
        autocast_enabled=True,
        save_case_predictions=False,
        prediction_dir=None,
    )
    results["checkpoint_name"] = os.path.basename(checkpoint_dir.rstrip("/"))
    results["use_ema"] = use_ema
    json_path = os.path.join(save_root, f"generation_eval_{num_infer_steps}steps.json")
    csv_path = os.path.join(save_root, f"generation_eval_{num_infer_steps}steps_per_case.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_case_csv(results["per_case"], csv_path)
    print_summary(results)
    step_compare = evaluate_multiple_sampling_steps(
        model=model,
        loader=test_loader,
        device=device,
        infer_steps_list=[50, 100, 250, 500],
        max_batches=None,
    )
    compare_path = os.path.join(save_root, "sampling_steps_comparison.json")
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(step_compare, f, indent=2)
    print("\nSaved:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - {compare_path}")